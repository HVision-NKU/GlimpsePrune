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

        if cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None
        
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
        return self.llm_forward(
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
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0] # [n, 3584]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                mask = (input_ids == self.config.image_token_id)
                # print(n_image_tokens)

                # print(position_ids.shape)
                # print("x1:", position_ids[0,0,:])
                # print("x2:", position_ids[1,0,:])
                # print("x3:", position_ids[2,0,:])

                self.model.keep_indices = None
                self.model.image_grid_thw = image_grid_thw
                if visual_token_ratio != 1:
                    num_keep_tokens = int(visual_token_ratio * n_image_tokens)
                    indices = torch.nonzero(mask)

                    keep_indices_local = window_selection(attn_weights_local, num_keep_tokens //2, image_grid_thw, window_size=4)

                    attn_weights_global[keep_indices_local] = 0 # avoid repetition
                    keep_indices_global = torch.topk(attn_weights_global, num_keep_tokens - num_keep_tokens // 2).indices
                    
                    keep_indices = torch.cat((keep_indices_local, keep_indices_global), dim=0)
                    keep_indices = torch.sort(keep_indices, dim=0)[0]
                    self.model.keep_indices = keep_indices

                    # Directly drop other tokens
                    # image_embeds = image_embeds[keep_indices, :]
                    # Token merging
                    image_embeds = token_merging(image_embeds, keep_indices, scaling=1)
                    
                    # Select the tokens with the lowest attention weights to remove
                    all_indices = torch.arange(n_image_tokens).to(keep_indices.device)
                    remove_indices = all_indices[~torch.isin(all_indices, keep_indices)]
                    indices_to_remove = indices[remove_indices]
                    # indices_to_remove = indices[:num_remove_tokens]
                    remove_mask = torch.ones_like(input_ids, dtype=torch.bool)
                    for index in indices_to_remove:
                        remove_mask[index[0], index[1]] = False
                    input_ids = input_ids[remove_mask].reshape(input_ids.shape[0], -1)
                    # Correctly apply the mask for position ids across all heads
                    position_ids = position_ids[remove_mask.unsqueeze(0).expand(position_ids.shape[0], -1, -1)].reshape(position_ids.shape[0], position_ids.shape[1], -1)

                n_image_tokens_after = (input_ids == self.config.image_token_id).sum().item()
                n_image_features_after = image_embeds.shape[0]
                self.model.n_image_tokens = n_image_tokens_after
                self.model.image_start_index = torch.nonzero(mask)[0, 1]
                if n_image_tokens_after != n_image_features_after:
                    raise ValueError(
                        f"Image features and image tokens do not match after pruning: tokens: {n_image_tokens_after}, features {n_image_features_after}"
                    )
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
                    hidden_states, _ = blk(hidden_states, cu_seqlens=cu_seqlens_now, position_embeddings=position_embeddings, output_attentions=output_attentions)

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
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
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
        
        cur_image_tokens = self.n_image_tokens
        
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
                    if cur_image_tokens > 0:
                        stage = self.layer_list.index(rank_layer) # determine current stage
                        next_image_tokens = int(image_token_ratio_list[stage] * cur_image_tokens)
                        (
                            position_ids,
                            attention_mask,
                            hidden_states,
                            sum_visual,
                            top_rank_index_x
                        ) = self.layer_prune(    
                            cur_num=stage,
                            rank_layer=rank_layer,
                            features=hidden_states,  
                            position_ids=position_ids,
                            attention_mask=causal_mask,
                            position_embeddings=position_embeddings,
                            cur_image_tokens=cur_image_tokens,
                            next_image_tokens=next_image_tokens,
                        )
                        position_embeddings = self.rotary_emb(hidden_states, position_ids)
                        cur_image_tokens = next_image_tokens

                        if output_masks:
                            self.image_token_bool_masks.append(self._make_mask(self.keep_indices, top_rank_index_x, self.image_grid_thw[:, 1:] // 2))
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
        self, cur_num, rank_layer, features, position_ids, attention_mask, position_embeddings, cur_image_tokens, next_image_tokens
    ):

        _position_ids = position_ids
        _attention_mask = attention_mask
        
        # print(features.shape) # [1, 357, 3584]
        # print(position_ids.shape) # [3, 1, 357]
        # print(attention_mask) # [1, 357, 3584]
        
        batch_size = features.shape[0] # 1
        seq_len = position_ids.shape[2] # 357
            
        hidden_states = features.clone().detach()
        self_attn = self.layers[rank_layer].self_attn
        hidden_states = self.layers[rank_layer].input_layernorm(hidden_states)
        
        num_heads = self_attn.num_heads # 28
        num_key_value_heads = self_attn.num_key_value_heads # 4
        head_dim = self_attn.head_dim # 128
        
        bsz, q_len, _ = hidden_states.size()

        query_states = self_attn.q_proj(hidden_states)
        key_states = self_attn.k_proj(hidden_states)
        value_states = self_attn.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, head_dim).transpose(1, 2)

        # cos, sin = position_embeddings
        # query_states, key_states = apply_multimodal_rotary_pos_emb(
        #     query_states, key_states, cos, sin, self_attn.rope_scaling["mrope_section"]
        # )

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self_attn.num_key_value_groups)
        value_states = repeat_kv(value_states, self_attn.num_key_value_groups)
        
        # obtain current states
        cur_key_states = key_states[0]
        cur_query_states = query_states[0]
        
        text_query_states = cur_query_states[:, seq_len - 1, :].unsqueeze(1)
        image_start_index = self.image_start_index
        image_end_index = image_start_index + cur_image_tokens

        attn_weights = torch.matmul(text_query_states, cur_key_states.transpose(1, 2)) / math.sqrt(head_dim)

        # Fix precision issues in Qwen2-VL float16 inference
        # Replace inf values with zeros in attention weights to prevent NaN propagation
        if text_query_states.dtype == torch.float16:
            attn_weights = torch.where(torch.isinf(attn_weights), torch.zeros_like(attn_weights), attn_weights)

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        sum_visual = torch.sum(attn_weights[:, :, image_start_index:image_end_index], dim=-1)
        sum_visual = sum_visual.squeeze(1) # (32)
        sum_visual, indices = torch.sort(sum_visual, descending=True) # (32)
        # print(rank_layer)
        # print(sum_visual)
        # print(attn_weights.shape) # [28, 1, 357]
        attention_avg_head = torch.mean(attn_weights, dim=0) # avg across heads
        attention_avg_head = attention_avg_head[:,image_start_index:image_end_index] # select image token as keys
        attention_avg_text = torch.mean(attention_avg_head, dim=0) # (576)
        
        # rank and drop by attention score
        top_rank_index = attention_avg_text.topk(next_image_tokens).indices
        top_rank_index_x = top_rank_index.clone()
        
        image_start_index = image_start_index.to(top_rank_index.device)
        top_rank_index = top_rank_index + image_start_index
        top_rank_index = top_rank_index.sort().values
        


        start_index = image_end_index
        new_input_embeds = torch.cat([features[0, :image_start_index, :], features[0,top_rank_index, :], features[0, start_index:, :]], dim=0)
        new_input_embeds = new_input_embeds.unsqueeze(0)
        top_rank_index = top_rank_index.to(position_ids.device)
        start_index = start_index.to(position_ids.device)
        new_position_ids = torch.cat([position_ids[:, :, :image_start_index], position_ids[:, :,top_rank_index], position_ids[:, :, start_index:]], dim=2)
        
        # print(new_input_embeds.shape)
        # print(new_position_ids.shape)
        # print(n_image_tokens)
        # print(int(n_keep_ratio * n_image_tokens))

        if _position_ids is None:
            position_ids = None
        
        return new_position_ids, attention_mask, new_input_embeds, sum_visual, top_rank_index_x

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
