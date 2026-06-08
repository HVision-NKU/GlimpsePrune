import os

import torch

from transformers_gp.models.internvl2_5 import InternVL2_5_GP_ForConditionalGeneration, InternVL2_5_GPConfig
from transformers_gp.models.internvl2_5.processing import InternVLGPProcessor

from .base import BaseInferModel


class InternVL2_5_GP(BaseInferModel):
    def _init_model(
        self,
        new_modules_dir=None,
        new_modules_config=None,
        use_ref_masks=False,
        use_zero_masks=False,
        reduce_layer=None,
        min_remain_num=None,
        max_remain_ratio=None,
        fixed_remain_ratio=None,
        **kwargs,
    ):
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        config = None
        if new_modules_config is not None:
            config = InternVL2_5_GPConfig.from_json_file(new_modules_config)
        use_flash_attn = self._attn_implementation == "flash_attention_2"
        model = InternVL2_5_GP_ForConditionalGeneration.from_pretrained(
            self._base_model,
            config=config,
            torch_dtype=self._torch_dtype,
            use_flash_attn=use_flash_attn,
        )
        if new_modules_dir is not None:
            model.load_new_modules(new_modules_dir)
        if use_ref_masks:
            model.config.use_ref_masks = True
        if use_zero_masks:
            assert not use_ref_masks, "use_ref_masks should be False when use_zero_masks is True"
            model.config.use_zero_masks = True
        if reduce_layer is not None:
            model.config.reduce_layer = reduce_layer
            model.config.selected_layers = (reduce_layer,)
        if min_remain_num is not None:
            model.config.min_remain_num = min_remain_num
        if max_remain_ratio is not None:
            model.config.max_remain_ratio = max_remain_ratio
        if fixed_remain_ratio is not None:
            model.config.fixed_remain_ratio = fixed_remain_ratio
        self._model = model.to(device).eval()

    def _init_processor(self, min_pixels=None, max_pixels=None, **kwargs):
        max_dynamic_patch = 12 if max_pixels is None else max(1, int((max_pixels + 448 * 448 - 1) // (448 * 448)))
        min_dynamic_patch = 1 if min_pixels is None else max(1, int((min_pixels + 448 * 448 - 1) // (448 * 448)))
        self._processor = InternVLGPProcessor.from_pretrained(
            self._base_model,
            max_dynamic_patch=max_dynamic_patch,
            min_dynamic_patch=min_dynamic_patch,
            use_thumbnail=True,
        )
        self._model.tokenizer = self._processor.tokenizer
        self._model._ensure_img_context_token_id()
        print(f"InternVL2.5 GP dynamic tiles: min={min_dynamic_patch}, max={max_dynamic_patch}")

    @property
    def model_config(self):
        return self._model.language_model.config

    def _do_generate(self, inputs, generation_config, do_selection):
        self._model.reset_image_tokens_cache()
        eos_token_id = self._processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        generation_config.eos_token_id = eos_token_id
        generation_config.pad_token_id = eos_token_id
        generate_ids = self._model.generate(
            **inputs,
            generation_config=generation_config,
            do_selection=do_selection,
        )
        prefix_ids = self._model.reduced_input_ids
        prefix_len = prefix_ids.shape[1] if prefix_ids is not None else inputs["input_ids"].shape[1]
        return generate_ids[:, prefix_len:]

    def _do_glimpse(self, inputs, generation_config):
        outputs = self._model(**inputs, do_selection=True, return_dict=True)
        return outputs.image_token_bool_masks

    def prepare_batch_inputs(self, batched_querys, batched_img_paths, batched_bboxes):
        pixel_values_list = []
        ref_token_masks = []
        num_patches_list = []
        prompts = []
        for query, image_path, bboxes in zip(batched_querys, batched_img_paths, batched_bboxes or [None] * len(batched_querys)):
            pixel_values, ref_mask = self._processor.load_image(image_path, normed_bboxes=bboxes)
            num_patches = pixel_values.shape[0]
            pixel_values_list.append(pixel_values)
            ref_token_masks.append(ref_mask)
            num_patches_list.append(num_patches)
            prompts.append(self._processor.build_prompt(query, num_patches, answer=None))

        model_inputs = self._processor.tokenizer(prompts, return_tensors="pt", padding=True)
        return {
            "input_ids": model_inputs["input_ids"].to(self._device),
            "attention_mask": model_inputs["attention_mask"].to(self._device),
            "pixel_values": torch.cat(pixel_values_list, dim=0).to(self._device, dtype=self._torch_dtype),
            "num_patches_list": num_patches_list,
            "ref_token_masks": [mask.to(self._device) for mask in ref_token_masks],
        }

    def batch_decode(self, generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True, **kwargs):
        return self._processor.tokenizer.batch_decode(
            generate_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )


__all__ = [
    "InternVL2_5_GP",
]
