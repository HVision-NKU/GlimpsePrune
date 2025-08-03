import os

from qwen_vl_utils import process_vision_info
from transformers.models.qwen2_5_vl import (
    Qwen2_5_VLProcessor,
)
from qwen_visionzip.qwen2_5vl_visionzip import (
    Qwen2_5_VLForConditionalGeneration,
)

from .base import BaseInferModel


class Qwen2_5_VL_VisionZip(BaseInferModel):
    def _init_model(self, dominant_ratio, contextual_ratio, **kwargs):
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self._base_model,
            torch_dtype=self._torch_dtype,
            attn_implementation=self._attn_implementation,
            device_map={"": local_rank},
        )
        model.dominant_ratio = dominant_ratio
        model.contextual_ratio = contextual_ratio
        self._model = model
        self._model.eval()
    
    def _init_processor(self, **kwargs):
        self._processor = Qwen2_5_VLProcessor.from_pretrained(self._base_model, padding_side="left")
    
    def _do_generate(self, inputs, generation_config, do_selection):
        generate_ids = self._model.generate(**inputs, generation_config=generation_config)
        generated_ids_trimmed = generate_ids[:, inputs['input_ids'].shape[1]:]
        return generated_ids_trimmed
    
    def _do_glimpse(self, inputs, generation_config):
        raise NotImplementedError(
            "Glimpse is not supported for Qwen2.5-VL model. "
        )
    
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
    "Qwen2_5_VL_VisionZip",
]