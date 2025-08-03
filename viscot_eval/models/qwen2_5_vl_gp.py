import os
import torch
import yaml
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers_gp.models.qwen2_5_vl import (
    Qwen2_5_VL_GP_Processor,
    Qwen2_5_VL_GP_ForConditionalGeneration,
)

from .base import BaseInferModel
from warppers import time_logger_disabled


class Qwen2_5_VL_GP(BaseInferModel):
    def _init_model(self,
                    new_modules_dir=None,
                    use_ref_masks=False,
                    use_zero_masks=False,
                    reduce_layer=None,
                    anchor_positions=None,
                    min_remain_num=None,
                    max_remain_ratio=None,
                    adapter_dir=None,
                    adapter_merge=False,
                    new_modules_config=None,
                    **kwargs):
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        
        model = Qwen2_5_VL_GP_ForConditionalGeneration.from_pretrained(
            self._base_model,
            torch_dtype=self._torch_dtype,
            attn_implementation=self._attn_implementation,
            device_map={"": local_rank},
        )
        
        if new_modules_dir is not None:
            model.load_new_modules(new_modules_dir)
        elif new_modules_config is not None:
            config_dict = model.config.to_dict()
            new_config = model.config_class.from_json_file(new_modules_config)
            config_dict.update(new_config.to_dict())
            config = model.config_class.from_dict(config_dict)
            model._init_new_modules(config, re_init=True)
            
        if use_ref_masks:
            model.config.use_ref_masks = True
        if use_zero_masks:
            assert not use_ref_masks, "use_ref_masks should be False when use_zero_masks is True"
            model.config.use_zero_masks = True
        if reduce_layer is not None:
            model.config.reduce_layer = reduce_layer
        if anchor_positions is not None:
            model.config.anchor_positions = anchor_positions
        if min_remain_num is not None:
            model.config.min_remain_num = min_remain_num
        if max_remain_ratio is not None:
            model.config.max_remain_ratio = max_remain_ratio
        
        if adapter_dir is not None:
            model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
            if adapter_merge:
                model = model.merge_and_unload()
            print(f"Loaded adapter from {adapter_dir}, merge: {adapter_merge}")
        self._model = model
        self._model.eval()
    
    def _init_processor(self, **kwargs):
        self._processor = Qwen2_5_VL_GP_Processor.from_pretrained(self._base_model, padding_side="left")

    def _do_generate(self, inputs, generation_config, do_selection):
        self._model.reset_image_tokens_cache()
        if do_selection and isinstance(self._model, PeftModel):
            with self._model.disable_adapter():
                with time_logger_disabled():
                    output = self._model(**inputs, return_dict=True)
            inputs['use_ref_masks'] = True
            inputs['ref_token_masks'] = [one_image_token_masks[-1:].sigmoid() for one_image_token_masks in output.image_token_masks]
        generate_ids = self._model.generate(
            **inputs,
            do_selection=do_selection,
            generation_config=generation_config,
        )
        generated_ids_trimmed = generate_ids[:, inputs['input_ids'].shape[1]:]
        return generated_ids_trimmed
    
    def _do_glimpse(self, inputs, generation_config):
        if isinstance(self._model, PeftModel):
            with self._model.disable_adapter():
                output = self._model(**inputs, return_dict=True)
        else:
            output = self._model(**inputs, return_dict=True)
        # image_token_masks = output.image_token_masks
        # final_image_token_masks = [one_image_token_masks[-1].sigmoid() for one_image_token_masks in image_token_masks]
        # final_image_token_bool_masks = output.image_token_bool_masks
        # return final_image_token_masks, final_image_token_bool_masks
        return output.image_token_bool_masks
        
    def prepare_batch_inputs(self, batched_querys, batched_img_paths, batched_bboxes):
        batched_messages = []
        for one_query, one_img_path in zip(batched_querys, batched_img_paths):
            one_message = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": one_img_path},
                    {"type": "text", "text": one_query}
                ]
            }]
            batched_messages.append(one_message)
        text = self._processor.apply_chat_template(
                batched_messages, tokenize=False, add_generation_prompt=True
            )
        image_inputs, video_inputs = process_vision_info(batched_messages)
        inputs = self._processor(
                text=text,
                normed_bboxes=batched_bboxes,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        inputs = inputs.to(self._device)
        return inputs

    def batch_decode(self, generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True, **kwargs):
        return self._processor.batch_decode(generate_ids, 
                                            skip_special_tokens=skip_special_tokens, 
                                            clean_up_tokenization_spaces=clean_up_tokenization_spaces)


__all__ = [
    "Qwen2_5_VL_GP",
]