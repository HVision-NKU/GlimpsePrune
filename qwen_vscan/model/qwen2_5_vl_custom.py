from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss
import numpy as np

from transformers.generation.utils import GenerationMixin
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLModel,
    Qwen2_5_VLCausalLMOutputWithPast
)
from transformers.modeling_outputs import BaseModelOutputWithPast

from qwen_vscan.model.qwen2_5_vl_utils import apply_rotary_pos_emb_vision, rotate_half, token_merging, window_selection, repeat_kv, apply_multimodal_rotary_pos_emb, apply_rotary_pos_emb_flashatt

from flash_attn import flash_attn_func, flash_attn_varlen_func
    
import sys
import time

class Qwen2_5_VLForConditionalGeneration_X(Qwen2_5_VLForConditionalGeneration):
    
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        output_masks=False,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        model_inputs = GenerationMixin.prepare_inputs_for_generation(
            self,            
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            use_cache=use_cache,
            **kwargs,
        )

        # Qwen2-5-VL position_ids are prepareed with rope_deltas in forward
        model_inputs["position_ids"] = None

        if cache_position is not None and cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None
            model_inputs["image_grid_thw"] = None
            model_inputs["video_grid_thw"] = None
            model_inputs["second_per_grid_ts"] = None

        # If the prompt was pruned in the prefill stage, the generation loop still carries the original
        # `attention_mask` length. Rebuild it from the pruned prompt mask + generated tokens so shapes match the cache.
        if past_key_values is not None and getattr(past_key_values, "get_seq_length", None) is not None:
            past_len = int(past_key_values.get_seq_length())
            if past_len > 0:
                base_attn = getattr(self.model, "_vscan_prefill_attention_mask", None)
                base_len = getattr(self.model, "_vscan_prefill_seq_len", None)
                if base_attn is not None and base_len is not None:
                    cur_input_ids = model_inputs.get("input_ids", None)
                    cur_inputs_embeds = model_inputs.get("inputs_embeds", None)
                    cur_len = int(cur_input_ids.shape[1]) if cur_input_ids is not None else int(cur_inputs_embeds.shape[1])

                    gen_so_far = past_len - int(base_len)
                    gen_so_far = max(gen_so_far, 0)
                    extra = torch.ones(
                        (base_attn.shape[0], gen_so_far + cur_len),
                        device=base_attn.device,
                        dtype=base_attn.dtype,
                    )
                    model_inputs["attention_mask"] = torch.cat([base_attn, extra], dim=1)
                    model_inputs["cache_position"] = torch.arange(
                        past_len, past_len + cur_len, device=base_attn.device, dtype=torch.long
                    )

                # Never forward vision inputs after prefill.
                model_inputs["pixel_values"] = None
                model_inputs["pixel_values_videos"] = None
                model_inputs["image_grid_thw"] = None
                model_inputs["video_grid_thw"] = None
                model_inputs["second_per_grid_ts"] = None
        
        model_inputs["output_masks"] = output_masks

        return model_inputs
    
    def llm_forward(self, 
                    input_ids: Optional[torch.LongTensor] = None,
                    attention_mask: Optional[torch.Tensor] = None,
                    position_ids: Optional[torch.LongTensor] = None,
                    past_key_values: Optional[List[torch.FloatTensor]] = None,
                    inputs_embeds: Optional[torch.FloatTensor] = None,
                    use_cache: Optional[bool] = None,
                    output_attentions: Optional[bool] = None,
                    output_hidden_states: Optional[bool] = None,
                    return_dict: Optional[bool] = None,
                    cache_position: Optional[torch.LongTensor] = None,
                    output_masks: bool = False):
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            output_masks=output_masks
        )
        hidden_states = outputs[0]
        return self.lm_head(hidden_states), outputs
    
    def llm_forward_prefilling(self, 
                    input_ids: Optional[torch.LongTensor] = None,
                    attention_mask: Optional[torch.Tensor] = None,
                    position_ids: Optional[torch.LongTensor] = None,
                    past_key_values: Optional[List[torch.FloatTensor]] = None,
                    inputs_embeds: Optional[torch.FloatTensor] = None,
                    use_cache: Optional[bool] = None,
                    output_attentions: Optional[bool] = None,
                    output_hidden_states: Optional[bool] = None,
                    return_dict: Optional[bool] = None,
                    cache_position: Optional[torch.LongTensor] = None,
                    output_masks: bool = False):
        if output_masks:
            self.model.image_token_bool_masks = []
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            output_masks=output_masks
        )
        hidden_states = outputs[0]
        return self.lm_head(hidden_states), outputs
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        output_masks: bool = False
    ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        >>> model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        

        # RoPE for pre-fill stage
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                )
                self.rope_deltas = rope_deltas
            else:
                pass

        if inputs_embeds is None:
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                visual_token_ratio = self.image_token_ratio
                image_embeds, attn_weights_local, attn_weights_global = self.visual(pixel_values, grid_thw=image_grid_thw, output_attentions=visual_token_ratio!=1)
                image_token_mask = (input_ids == self.config.image_token_id)
                n_image_tokens_total = int(image_token_mask.sum().item())
                n_image_features = int(image_embeds.shape[0])  # [n, 3584]
                if n_image_tokens_total != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens_total}, features {n_image_features}"
                    )

                spatial_merge_size = int(self.config.vision_config.spatial_merge_size)

                # Track per-sample multi-image token layout for later (LLM-layer pruning).
                # `image_grid_thw` is flattened across the whole batch in the same order as image tokens in `input_ids`.
                if image_grid_thw is None:
                    raise ValueError("`image_grid_thw` must be provided when `pixel_values` is not None.")

                per_image_lens_full: List[int] = []
                for grid in image_grid_thw:
                    t, h, w = int(grid[0].item()), int(grid[1].item()), int(grid[2].item())
                    per_image_lens_full.append(t * (h // spatial_merge_size) * (w // spatial_merge_size))
                if sum(per_image_lens_full) != n_image_tokens_total:
                    raise ValueError(
                        "Sum(image_grid_thw token lens) does not match the number of `<|image_pad|>` tokens in `input_ids`: "
                        f"sum(per_image_lens_full)={sum(per_image_lens_full)} vs n_image_tokens_total={n_image_tokens_total}"
                    )

                batch_size, seq_len = input_ids.shape
                per_sample_image_lens_full: List[List[int]] = []
                img_ptr = 0
                for b in range(batch_size):
                    n_b = int(image_token_mask[b].sum().item())
                    lens_b: List[int] = []
                    remain = n_b
                    while remain > 0:
                        if img_ptr >= len(per_image_lens_full):
                            raise ValueError("Ran out of `image_grid_thw` entries when mapping images to batch samples.")
                        l = int(per_image_lens_full[img_ptr])
                        lens_b.append(l)
                        img_ptr += 1
                        remain -= l
                    if remain != 0:
                        raise ValueError(
                            "Per-sample `<|image_pad|>` count is not compatible with `image_grid_thw` lens. "
                            f"batch={b} remain={remain}"
                        )
                    per_sample_image_lens_full.append(lens_b)
                if img_ptr != len(per_image_lens_full):
                    raise ValueError(
                        f"Unused `image_grid_thw` entries: used={img_ptr}, total={len(per_image_lens_full)}"
                    )

                self.model.keep_indices = None
                self.model.image_grid_thw = image_grid_thw

                # Phase-A visual pruning (token selection + token merging) + shrink `<|image_pad|>` spans in prompt.
                if visual_token_ratio != 1:
                    # Split visual tokens by images (flattened) and prune each image independently by ratio.
                    image_embeds_chunks: List[torch.Tensor] = []
                    per_sample_image_lens_kept: List[List[int]] = []
                    keep_mask_rows: List[torch.Tensor] = []
                    single_keep_indices = None

                    # We will build the kept image embeddings in the same order as image tokens appear in `input_ids`
                    # (batch-major, left-to-right), so we can scatter them back via `masked_scatter`.
                    img_token_offset_global = 0
                    img_grid_offset_global = 0
                    for b in range(batch_size):
                        img_pos_b = torch.nonzero(image_token_mask[b], as_tuple=False).squeeze(1)
                        lens_b_full = per_sample_image_lens_full[b]
                        pos_offset = 0
                        keep_mask_b = torch.ones(seq_len, dtype=torch.bool, device=input_ids.device)
                        lens_b_kept: List[int] = []
                        for l_full in lens_b_full:
                            grid_thw = image_grid_thw[img_grid_offset_global]
                            img_grid_offset_global += 1

                            embeds_img = image_embeds[img_token_offset_global : img_token_offset_global + l_full]
                            if attn_weights_local is None or attn_weights_global is None:
                                raise ValueError("Attention weights must be returned when visual_token_ratio != 1.")
                            attn_local_img = attn_weights_local[img_token_offset_global : img_token_offset_global + l_full]
                            attn_global_img = attn_weights_global[img_token_offset_global : img_token_offset_global + l_full]
                            img_token_offset_global += l_full

                            keep_len = int(visual_token_ratio * l_full)
                            keep_len = max(1, min(keep_len, l_full))
                            lens_b_kept.append(keep_len)

                            if keep_len == l_full:
                                keep_indices = torch.arange(l_full, device=embeds_img.device, dtype=torch.int)
                                merged_img = embeds_img
                            else:
                                local_keep = keep_len // 2
                                global_keep = keep_len - local_keep
                                keep_parts: List[torch.Tensor] = []
                                attn_global_work = attn_global_img
                                if local_keep > 0:
                                    keep_local = window_selection(
                                        attn_local_img,
                                        local_keep,
                                        grid_thw,
                                        window_size=4,
                                        spatial_merge_size=spatial_merge_size,
                                    )
                                    keep_parts.append(keep_local)
                                    attn_global_work = attn_global_img.clone()
                                    attn_global_work[keep_local] = 0  # avoid repetition
                                if global_keep > 0:
                                    keep_global = torch.topk(attn_global_work, global_keep).indices.to(torch.int)
                                    keep_parts.append(keep_global)
                                keep_indices = torch.cat(keep_parts, dim=0)
                                keep_indices = torch.sort(keep_indices, dim=0)[0]
                                merged_img = token_merging(embeds_img, keep_indices, scaling=1)

                            image_embeds_chunks.append(merged_img)
                            if batch_size == 1 and len(per_sample_image_lens_full[0]) == 1:
                                single_keep_indices = keep_indices.to(dtype=torch.long)

                            # Drop corresponding `<|image_pad|>` tokens in the prompt for this image.
                            pos_img = img_pos_b[pos_offset : pos_offset + l_full]
                            pos_offset += l_full
                            keep_mask_img = torch.zeros(l_full, device=pos_img.device, dtype=torch.bool)
                            keep_mask_img[keep_indices.to(device=pos_img.device, dtype=torch.long)] = True
                            drop_pos_img = pos_img[~keep_mask_img]
                            if drop_pos_img.numel() > 0:
                                keep_mask_b[drop_pos_img] = False

                        per_sample_image_lens_kept.append(lens_b_kept)
                        keep_mask_rows.append(keep_mask_b)

                    # Apply per-row keep masks and left-pad back to a rectangular batch.
                    input_ids_rows: List[torch.Tensor] = []
                    attn_rows: List[torch.Tensor] = []
                    pos_rows: List[torch.Tensor] = []
                    max_new_len = 0
                    for b in range(batch_size):
                        km = keep_mask_rows[b]
                        input_b = input_ids[b][km]
                        if attention_mask is None:
                            attn_b = torch.ones_like(input_b, dtype=torch.long)
                        else:
                            attn_b = attention_mask[b][km]
                        pos_b = position_ids[:, b, km]
                        input_ids_rows.append(input_b)
                        attn_rows.append(attn_b)
                        pos_rows.append(pos_b)
                        max_new_len = max(max_new_len, int(input_b.shape[0]))

                    pad_id = int(self.config.pad_token_id) if self.config.pad_token_id is not None else 0
                    input_ids_new = input_ids.new_full((batch_size, max_new_len), pad_id)
                    attention_mask_new = attn_rows[0].new_zeros((batch_size, max_new_len))
                    position_ids_new = position_ids.new_zeros((position_ids.shape[0], batch_size, max_new_len))
                    for b in range(batch_size):
                        cur_len = int(input_ids_rows[b].shape[0])
                        pad_left = max_new_len - cur_len
                        input_ids_new[b, pad_left:] = input_ids_rows[b]
                        attention_mask_new[b, pad_left:] = attn_rows[b]
                        position_ids_new[:, b, pad_left:] = pos_rows[b]

                    input_ids = input_ids_new
                    attention_mask = attention_mask_new
                    position_ids = position_ids_new

                    image_embeds = torch.cat(image_embeds_chunks, dim=0)
                    self.model.image_lens_list = per_sample_image_lens_kept
                    self.model.keep_indices = single_keep_indices
                else:
                    self.model.image_lens_list = per_sample_image_lens_full

                # Update image token bookkeeping on the (possibly) pruned prompt.
                image_token_mask = (input_ids == self.config.image_token_id)
                n_image_tokens_after = int(image_token_mask.sum().item())
                n_image_features_after = int(image_embeds.shape[0])
                self.model.n_image_tokens = n_image_tokens_after
                self.model.image_token_mask = image_token_mask
                # Cache the (possibly) pruned prompt mask for generation-time reconstruction.
                if attention_mask is None:
                    self.model._vscan_prefill_attention_mask = torch.ones_like(input_ids, dtype=torch.long)
                else:
                    self.model._vscan_prefill_attention_mask = attention_mask
                self.model._vscan_prefill_seq_len = int(input_ids.shape[1])
                if n_image_tokens_after != n_image_features_after:
                    raise ValueError(
                        f"Image features and image tokens do not match after pruning: tokens: {n_image_tokens_after}, features {n_image_features_after}"
                    )

                # Update rope deltas to match the (padded) pruned prompt length, so generation uses consistent positions.
                # (We can't call `get_rope_index` again because `image_grid_thw` still reflects the unpruned grids.)
                if position_ids is not None:
                    max_pos = position_ids.max(dim=0).values.max(dim=-1).values  # [batch]
                    self.rope_deltas = (max_pos + 1 - input_ids.shape[1]).to(dtype=position_ids.dtype, device=position_ids.device).unsqueeze(1)
                    cache_position = None  # let the decoder recompute based on the new prompt length

                inputs_embeds = self.model.embed_tokens(input_ids)
                image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                image_mask = image_mask.to(inputs_embeds.device)
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            else:
                inputs_embeds = self.model.embed_tokens(input_ids)
                self.model.n_image_tokens = 0

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds, attn_weights = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )

                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)

                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                pass
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        
        if ((cache_position is not None and cache_position[0] == 0)
            or self.rope_deltas is None
            or (past_key_values is None or past_key_values.get_seq_length() == 0)):
            logits, outputs = self.llm_forward_prefilling(
                input_ids=None,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                output_masks=output_masks
            )
            # Phase-B (LLM-layer) pruning may further change the effective prompt length, so refresh rope_deltas
            # from the decoder if it provided an updated value.
            maybe_deltas = getattr(self.model, "_vscan_rope_deltas", None)
            if maybe_deltas is not None:
                self.rope_deltas = maybe_deltas
                self.model._vscan_rope_deltas = None
        else:
            logits, outputs = self.llm_forward(
                input_ids=None,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                output_masks=output_masks
            )

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen2_5_VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )
        
class Qwen2_5_VisionPatchEmbed_X(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states


class Qwen2_5_VisionTransformerPretrainedModel_X(Qwen2_5_VisionTransformerPretrainedModel):
    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, output_attentions: bool=False) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        hidden_states = self.patch_embed(hidden_states) # [3552, 1176] -> [3552, 1280]
        rotary_pos_emb = self.rot_pos_emb(grid_thw) # [3552, 40]
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=hidden_states.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)
        
        seq_len, _ = hidden_states.size() # [3552, 1280]
        hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1) # [888, 4, 1280]
        hidden_states = hidden_states[window_index, :, :] # [888, 4, 1280]
        hidden_states = hidden_states.reshape(seq_len, -1) # [3552, 1280]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        self.gradient_checkpointing = False
        
        # We select a full attention layer here to get the attention weights
        attn_weights_all = []
        # selected_layer = 22
        num_blocks = len(self.blocks)
        selected_layer_local = self.fullatt_block_indexes[0]
        selected_layer_global = num_blocks - 1
        # print(self.fullatt_block_indexes) # [7, 15, 23, 31]
        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    blk.__call__, hidden_states, cu_seqlens_now, None, position_embeddings
                )
            else:
                if layer_num == selected_layer_local:
                    hidden_states, attn_weights_local = blk(hidden_states, cu_seqlens=cu_seqlens_now, position_embeddings=position_embeddings, output_attentions=output_attentions)
                elif layer_num == selected_layer_global:
                    hidden_states, attn_weights_global = blk(hidden_states, cu_seqlens=cu_seqlens_now, position_embeddings=position_embeddings, output_attentions=output_attentions)
                    attn_weights_all.append(attn_weights_global)
                else:
                    hidden_states, _ = blk(hidden_states, cu_seqlens=cu_seqlens_now, position_embeddings=position_embeddings, output_attentions=False)

        if output_attentions:
            # Sum across heads
            attn_weights_local = torch.sum(attn_weights_local, dim=0) # [3552, 3552]
            # Sum across different tokens
            attn_weights_local = torch.mean(attn_weights_local, dim=0) # [3552]
            # Reshape to [-1, 4]
            attn_weights_local = attn_weights_local.view(-1, self.spatial_merge_unit) # [888, 4]
            attn_weights_local = torch.mean(attn_weights_local, dim=1) # [888]
            
            # Sum across heads
            attn_weights_global = torch.sum(attn_weights_global, dim=0) # [3552, 3552]
            # Sum across different tokens
            attn_weights_global = torch.mean(attn_weights_global, dim=0) # [3552]
            # Reshape to [-1, 4]
            attn_weights_global = attn_weights_global.view(-1, self.spatial_merge_unit) # [888, 4]
            attn_weights_global = torch.mean(attn_weights_global, dim=1) # [888]
        
        # Sum across heads
        # attn_weights = torch.sum(attn_weights, dim=1) # [3552, 3552]
        # # Sum across different tokens
        # attn_weights = torch.mean(attn_weights, dim=1) # [3552]
        # # Reshape to [-1, 4]
        # attn_weights = attn_weights.view(attn_weights.shape[0], -1, self.spatial_merge_unit) # [888, 4]
        # attn_weights = torch.mean(attn_weights, dim=2) # [888]
        
        hidden_states = self.merger(hidden_states) # [3552, 1280] -> [888, 3584]
        reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states[reverse_indices, :] # [888, 3584]
        if output_attentions:
            attn_weights_local = attn_weights_local[reverse_indices] # [888]
            attn_weights_global = attn_weights_global[reverse_indices] # [888]
            return hidden_states, attn_weights_local, attn_weights_global
        return hidden_states, None, None
    
class Qwen2_5_VLVisionBlock_X(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: Optional[bool] = None,
    ) -> torch.Tensor:
        
        attn_output, attn_weights = self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + attn_output
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states, attn_weights

class Qwen2_5_VLVisionFlashAttention2_X(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: Optional[bool] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        else:
            cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_flashatt(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
        q = q.squeeze(0)
        k = k.squeeze(0)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
            seq_length, -1
        )
        attn_output = self.proj(attn_output)
        
        attn_weights = None
        if output_attentions:
            q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
            q = q.transpose(0, 1)
            k = k.transpose(0, 1)
            v = v.transpose(0, 1)
            attention_mask_X = torch.full(
                [1, seq_length, seq_length], torch.finfo(q.dtype).min, device=q.device, dtype=q.dtype
            )
            for i in range(1, len(cu_seqlens)):
                attention_mask_X[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0
            head_dim = q.size(-1)
            attn_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(head_dim)
            attn_weights = attn_weights + attention_mask_X
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        return attn_output, attn_weights

class Qwen2_5_VLVisionSdpaAttention_X(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: Optional[bool] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        else:
            cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
        
        # print(cu_seqlens) # [0, 64, 128, ...]
        attention_mask = torch.zeros([1, seq_length, seq_length], device=q.device, dtype=torch.bool)
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = True
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        
        # Calculate attention weights (reference: Qwen2_5_VLVisionAttention)
        attn_weights = None
        if output_attentions:
            attention_mask_X = torch.full(
                [1, seq_length, seq_length], torch.finfo(q.dtype).min, device=q.device, dtype=q.dtype
            )
            for i in range(1, len(cu_seqlens)):
                attention_mask_X[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0
            head_dim = q.size(-1)
            attn_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(head_dim)
            attn_weights = attn_weights + attention_mask_X
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)

        attn_output = F.scaled_dot_product_attention(
            q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), attention_mask, dropout_p=0.0
        )
        attn_output = attn_output.squeeze(0).transpose(0, 1)
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output, attn_weights
    
class Qwen2_5_VLModel_X(Qwen2_5_VLModel):
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_masks: bool = False
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        start_time = time.time()
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # torch.jit.trace() doesn't support cache objects in the output
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # the hard coded `3` is for temporal, height and width.
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.dim() == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        attention_mask_2d = attention_mask
        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        
        # Prompt-time image bookkeeping (set in `Qwen2_5_VLForConditionalGeneration_X.forward`)
        # - `image_token_mask`: [bs, seq_len] bool
        # - `image_lens_list`: List[List[int]] per-sample per-image lens (sum matches image_token_mask count)
        image_token_mask = getattr(self, "image_token_mask", None)
        image_lens_list = getattr(self, "image_lens_list", None)
        has_images = bool(image_token_mask is not None and int(image_token_mask.sum().item()) > 0)
        
        sum_visual_attention = []
        # 28 x Qwen2_5_VLDecoderLayer
        # print(self.layers)
        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
                
            # Modify here
            # rank & drop after specific layer
            # only drop in prefill stage when inference
            image_token_ratio_list = self.image_token_ratio_list
            rank_layer = layer_idx + 1
            if rank_layer in self.layer_list:
                if hidden_states.shape[1] != 1:  # prefill stage
                    if has_images:
                        stage = self.layer_list.index(rank_layer) # determine current stage
                        (
                            position_ids,
                            attention_mask_2d,
                            hidden_states,
                            sum_visual,
                            top_rank_index_x
                        ) = self.layer_prune(    
                            cur_num=stage,
                            rank_layer=rank_layer,
                            features=hidden_states,  
                            position_ids=position_ids,
                            attention_mask=attention_mask_2d,
                            position_embeddings=position_embeddings,
                            stage_ratio=float(image_token_ratio_list[stage]),
                            past_key_values=past_key_values,
                        )
                        # Update cached prompt state for generation (used by `prepare_inputs_for_generation`)
                        self._vscan_prefill_attention_mask = attention_mask_2d
                        self._vscan_prefill_seq_len = int(attention_mask_2d.shape[1])

                        # Refresh causal mask / RoPE for subsequent layers after sequence shortening.
                        cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device)
                        causal_mask = self._update_causal_mask(
                            attention_mask_2d, hidden_states, cache_position, None, output_attentions
                        )
                        position_embeddings = self.rotary_emb(hidden_states, position_ids)

                        if output_masks:
                            if (
                                top_rank_index_x is not None
                                and getattr(self, "keep_indices", None) is not None
                                and getattr(self, "image_grid_thw", None) is not None
                                and hidden_states.shape[0] == 1
                            ):
                                self.image_token_bool_masks.append(
                                    self._make_mask(self.keep_indices, top_rank_index_x, self.image_grid_thw[:, 1:] // 2)
                                )
                        # print(cur_image_tokens)
                        # sum_visual_attention.append(sum_visual)

        # if len(sum_visual_attention) > 0:
        #     sum_visual_attention = torch.cat(sum_visual_attention, dim=0)
        #     sum_visual_attention = sum_visual_attention.view(28, -1)
        #     print(sum_visual_attention.dtype) #bfloat16
        #     if os.path.exists('sum_visual_attention_qwen2_5_vl.pt'):
        #         prev_sum_visual_attention = torch.load('sum_visual_attention_qwen2_5_vl.pt')
        #         prev_sum_visual_attention = prev_sum_visual_attention.to(torch.float32)
        #         sum_visual_attention = sum_visual_attention.to(torch.float32)
        #         print(sum_visual_attention[0, :5])
        #         sum_visual_attention = sum_visual_attention + prev_sum_visual_attention
        #         print(sum_visual_attention[0, :5])
        #     sum_visual_attention = sum_visual_attention.to(torch.float32)
        #     torch.save(sum_visual_attention, 'sum_visual_attention_qwen2_5_vl.pt')
        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        # print(f"Elapsed time: {elapsed_time}")
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        
    def _make_mask(self, keep_indices, top_rank_index_x, attn_hw):
        """Create a boolean mask for the image tokens."""
        mask = torch.zeros(attn_hw[0].tolist(), dtype=torch.bool, device=keep_indices.device)
        mask = mask.flatten()
        remain_indices = keep_indices[top_rank_index_x]
        mask[remain_indices] = True
        return mask
        # mask[keep_indices][top_rank_index_x] = True
        
        
        
    def layer_prune(
        self,
        cur_num,
        rank_layer,
        features,
        position_ids,
        attention_mask,
        position_embeddings,
        stage_ratio: float,
        past_key_values=None,
    ):
        if rank_layer >= len(self.layers):
            return position_ids, attention_mask, features, None, None

        image_token_mask = getattr(self, "image_token_mask", None)
        image_lens_list = getattr(self, "image_lens_list", None)
        if image_token_mask is None or image_lens_list is None:
            return position_ids, attention_mask, features, None, None

        batch_size, seq_len, _ = features.shape

        # Use the next layer's attention projections to score image tokens (original implementation behavior).
        self_attn = self.layers[rank_layer].self_attn
        hidden_states = self.layers[rank_layer].input_layernorm(features)

        head_dim = self_attn.head_dim
        bsz, q_len, _ = hidden_states.size()
        query_states = self_attn.q_proj(hidden_states)
        key_states = self_attn.k_proj(hidden_states)
        value_states = self_attn.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, head_dim).transpose(1, 2)

        key_states = repeat_kv(key_states, self_attn.num_key_value_groups)
        value_states = repeat_kv(value_states, self_attn.num_key_value_groups)

        # Attention from the last token to all tokens.
        text_query_states = query_states[:, :, -1:, :]  # [bs, heads, 1, head_dim]
        attn_weights = torch.matmul(text_query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)  # [bs, heads, 1, seq]
        if text_query_states.dtype == torch.float16:
            attn_weights = torch.where(torch.isinf(attn_weights), torch.zeros_like(attn_weights), attn_weights)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # Average across heads -> [bs, seq]
        attn_scores = attn_weights.mean(dim=1).squeeze(1)

        # Build per-sample keep indices (keep all non-image tokens; prune image tokens per-image by stage_ratio).
        keep_indices_rows: List[torch.Tensor] = []
        new_image_lens_list: List[List[int]] = []
        new_image_token_mask_rows: List[torch.Tensor] = []
        new_features_rows: List[torch.Tensor] = []
        new_position_rows: List[torch.Tensor] = []
        new_attn_rows: List[torch.Tensor] = []
        max_new_len = 0
        top_rank_index_x = None  # kept for bs=1/single-image mask export

        for b in range(batch_size):
            img_pos = torch.nonzero(image_token_mask[b], as_tuple=False).squeeze(1)
            lens_b = list(image_lens_list[b])
            if sum(lens_b) != int(img_pos.numel()):
                # Fallback: treat as a single image span.
                lens_b = [int(img_pos.numel())] if int(img_pos.numel()) > 0 else []

            keep_mask_b = ~image_token_mask[b]
            new_lens_b: List[int] = []
            pos_offset = 0
            per_image_kept_rel: List[torch.Tensor] = []

            for l in lens_b:
                if l <= 0:
                    new_lens_b.append(0)
                    continue
                seg_pos = img_pos[pos_offset : pos_offset + l]
                pos_offset += l

                seg_scores = attn_scores[b, seg_pos]
                keep_l = int(stage_ratio * l)
                keep_l = max(1, min(keep_l, l))
                kept_rel = seg_scores.topk(keep_l).indices
                kept_rel = torch.sort(kept_rel).values  # keep original order
                keep_mask_b[seg_pos[kept_rel]] = True
                new_lens_b.append(keep_l)
                per_image_kept_rel.append(kept_rel)

            keep_idx_b = torch.nonzero(keep_mask_b, as_tuple=False).squeeze(1)

            # Save a bs=1/single-image compatible index for mask export.
            if batch_size == 1 and len(lens_b) == 1 and len(per_image_kept_rel) == 1:
                top_rank_index_x = per_image_kept_rel[0]

            keep_indices_rows.append(keep_idx_b)
            new_image_lens_list.append(new_lens_b)
            new_image_token_mask_rows.append(image_token_mask[b][keep_idx_b])
            new_features_rows.append(features[b][keep_idx_b])
            new_position_rows.append(position_ids[:, b, keep_idx_b])
            if attention_mask is None:
                new_attn_rows.append(torch.ones_like(keep_idx_b, dtype=torch.long, device=keep_idx_b.device))
            else:
                new_attn_rows.append(attention_mask[b][keep_idx_b])
            max_new_len = max(max_new_len, int(keep_idx_b.numel()))

        # Left-pad to a rectangular batch.
        new_features = features.new_zeros((batch_size, max_new_len, features.shape[-1]))
        new_position_ids = position_ids.new_zeros((position_ids.shape[0], batch_size, max_new_len))
        new_attention_mask = new_attn_rows[0].new_zeros((batch_size, max_new_len))
        new_image_token_mask = image_token_mask.new_zeros((batch_size, max_new_len))

        # Indices for cache gather: [bs, max_new_len] with left padding filled by 0.
        gather_indices = torch.zeros((batch_size, max_new_len), device=features.device, dtype=torch.long)
        for b in range(batch_size):
            keep_idx_b = keep_indices_rows[b].to(dtype=torch.long, device=features.device)
            cur_len = int(keep_idx_b.numel())
            pad_left = max_new_len - cur_len
            if cur_len > 0:
                new_features[b, pad_left:] = new_features_rows[b]
                new_position_ids[:, b, pad_left:] = new_position_rows[b]
                new_attention_mask[b, pad_left:] = new_attn_rows[b]
                new_image_token_mask[b, pad_left:] = new_image_token_mask_rows[b]
                gather_indices[b, pad_left:] = keep_idx_b

        # Update cached K/V for already-processed layers so all layers share the same (pruned) prompt length.
        if past_key_values is not None and hasattr(past_key_values, "key_cache"):
            # Layers < rank_layer have already been run (we prune caches for them).
            prune_layers = min(int(rank_layer), len(past_key_values.key_cache))
            for layer_idx in range(prune_layers):
                if layer_idx >= len(past_key_values.key_cache):
                    break
                key_cache = past_key_values.key_cache[layer_idx]
                value_cache = past_key_values.value_cache[layer_idx]
                if key_cache is None or (hasattr(key_cache, "numel") and key_cache.numel() == 0):
                    continue
                # key_cache: [bs, heads, seq, head_dim]
                idx = gather_indices.view(batch_size, 1, max_new_len, 1).expand(
                    batch_size, key_cache.shape[1], max_new_len, key_cache.shape[3]
                )
                past_key_values.key_cache[layer_idx] = torch.gather(key_cache, dim=2, index=idx)
                past_key_values.value_cache[layer_idx] = torch.gather(value_cache, dim=2, index=idx)
            if hasattr(past_key_values, "_seen_tokens"):
                past_key_values._seen_tokens = max_new_len

        # Update model state for later pruning / generation.
        self.image_token_mask = new_image_token_mask
        self.image_lens_list = new_image_lens_list
        self.n_image_tokens = int(new_image_token_mask.sum().item())
        # Provide updated rope deltas to the outer conditional generation module.
        # `get_rope_index` defines deltas as: max_position_id + 1 - seq_len.
        max_pos = new_position_ids.max(dim=0).values.max(dim=-1).values  # [batch]
        self._vscan_rope_deltas = (max_pos + 1 - max_new_len).to(
            dtype=new_position_ids.dtype, device=new_position_ids.device
        ).unsqueeze(1)

        return new_position_ids, new_attention_mask, new_features, None, top_rank_index_x

class Qwen2_5_VLDecoderLayer(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
