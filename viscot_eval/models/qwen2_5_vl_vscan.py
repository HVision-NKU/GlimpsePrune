import os
import ast
from qwen_vl_utils import process_vision_info

from transformers.models.qwen2_5_vl import (
    Qwen2_5_VLProcessor,
)

from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VLForConditionalGeneration,
        Qwen2_5_VisionTransformerPretrainedModel,
        Qwen2_5_VLVisionBlock,
        Qwen2_5_VLVisionSdpaAttention,
        Qwen2_5_VLVisionFlashAttention2,
        Qwen2_5_VisionPatchEmbed,
        Qwen2_5_VLModel,
    )
from qwen_vscan.model.qwen2_5_vl_custom import (
        Qwen2_5_VLForConditionalGeneration_X, 
        Qwen2_5_VisionTransformerPretrainedModel_X, 
        Qwen2_5_VLVisionBlock_X, 
        Qwen2_5_VLVisionSdpaAttention_X,
        Qwen2_5_VLVisionFlashAttention2_X,
        Qwen2_5_VisionPatchEmbed_X, 
        Qwen2_5_VLModel_X
    )


from .base import BaseInferModel
from warppers import time_logger


class Qwen2_5_VL_VScan(BaseInferModel):
    def _init_model(self,
                    layer_list,
                    image_token_ratio_list,
                    image_token_ratio,
                    **kwargs):
        Qwen2_5_VLForConditionalGeneration.forward = Qwen2_5_VLForConditionalGeneration_X.forward
        Qwen2_5_VLForConditionalGeneration.llm_forward = time_logger(Qwen2_5_VLForConditionalGeneration_X.llm_forward)
        Qwen2_5_VLForConditionalGeneration.llm_forward_prefilling = time_logger(Qwen2_5_VLForConditionalGeneration_X.llm_forward_prefilling)
        Qwen2_5_VLForConditionalGeneration.prepare_inputs_for_generation = Qwen2_5_VLForConditionalGeneration_X.prepare_inputs_for_generation
        
        Qwen2_5_VisionTransformerPretrainedModel.forward = Qwen2_5_VisionTransformerPretrainedModel_X.forward
        Qwen2_5_VLVisionBlock.forward = Qwen2_5_VLVisionBlock_X.forward
        Qwen2_5_VLVisionSdpaAttention.forward = Qwen2_5_VLVisionSdpaAttention_X.forward
        Qwen2_5_VLVisionFlashAttention2.forward = Qwen2_5_VLVisionFlashAttention2_X.forward
        Qwen2_5_VisionPatchEmbed.forward = Qwen2_5_VisionPatchEmbed_X.forward
        Qwen2_5_VLModel.forward = Qwen2_5_VLModel_X.forward
        Qwen2_5_VLModel.layer_prune = Qwen2_5_VLModel_X.layer_prune
        Qwen2_5_VLModel._make_mask = Qwen2_5_VLModel_X._make_mask
        
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self._base_model,
            torch_dtype=self._torch_dtype,
            attn_implementation=self._attn_implementation,
            device_map={"": local_rank},
        )
        self._model = model
        self._model.eval()
        
        self._model.model.layer_list = ast.literal_eval(layer_list)
        self._model.model.image_token_ratio_list = ast.literal_eval(image_token_ratio_list)
        self._model.image_token_ratio = image_token_ratio
        self._model.eval()
        
    def _init_processor(self, **kwargs):
        self._processor = Qwen2_5_VLProcessor.from_pretrained(self._base_model, padding_side="left")
    
    def _do_generate(self, inputs, generation_config, do_selection):
        generate_ids = self._model.generate(**inputs, generation_config=generation_config)
        generated_ids_trimmed = generate_ids[:, inputs['input_ids'].shape[1]:]
        return generated_ids_trimmed
        
    def _do_glimpse(self, inputs, generation_config):
        _ = self._model.generate(**inputs, generation_config=generation_config, output_masks=True)
        image_token_bool_masks = self._model.model.image_token_bool_masks
        return image_token_bool_masks
    
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
    "Qwen2_5_VL_VScan",
]