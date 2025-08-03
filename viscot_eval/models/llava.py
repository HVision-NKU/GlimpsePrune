import os
import torch
from PIL import Image

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.mm_utils import (
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model

from .base import BaseInferModel


class LLaVA(BaseInferModel):
    def _init_model(self, **kwargs):
        model_name = get_model_name_from_path(self._base_model)
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        # assert self._torch_dtype == torch.float16, "LLaVA_1.5 only supports float16 dtype for now."
        llava_model_args = {
            "multimodal": True,
            "attn_implementation": self._attn_implementation,
            "torch_dtype": self._torch_dtype,
        }
        try:
            self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(self._base_model, None, model_name, 
                                                                                                          device_map=f"cuda:{local_rank}", **llava_model_args)
        except TypeError:
            llava_model_args.pop("multimodal")
            self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(self._base_model, None, model_name, 
                                                                                                          device_map=f"cuda:{local_rank}", **llava_model_args)
        self._model.eval()
        self._conv_mode = "vicuna_v1"
        
    def _init_processor(self, **kwargs):
        pass
    
    def _do_generate(self, inputs, generation_config, do_selection):
        generate_ids = self._model.generate(
            inputs=inputs['inputs'],
            images=inputs['images'],
            image_sizes=inputs['image_sizes'],
            generation_config=generation_config,
        )
        return generate_ids
        
    def _do_glimpse(self, inputs, generation_config):
        raise NotImplementedError(
            "Glimpse is not supported for LLaVA model. "
        )
    
    def batch_decode(self, generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True, **kwargs):
        return [self._tokenizer.batch_decode(generate_ids, 
                                            skip_special_tokens=skip_special_tokens,
                                            clean_up_tokenization_spaces=clean_up_tokenization_spaces)[0].strip()]
                                            
        
    
    def prepare_batch_inputs(self, batched_querys, batched_img_paths, batched_bboxes):
        # pass
        curr_batch_size = len(batched_querys)
        assert curr_batch_size == 1, "LLaVA_1.5 only supports batch size 1 for now."
        
        input_ids = []
        images = []
        image_sizes = []
        
        for one_query, one_img_path in zip(batched_querys, batched_img_paths):
            if self._model.config.mm_use_im_start_end:
                one_query = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + one_query
            else:
                one_query = DEFAULT_IMAGE_TOKEN + '\n' + one_query
            conv = conv_templates[self._conv_mode].copy()
            conv.append_message(conv.roles[0], one_query)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            
            image = Image.open(one_img_path).convert("RGB")
            image_tensor = process_images([image], self._image_processor, self._model.config)[0]
            
            input_id = tokenizer_image_token(prompt, self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
            input_ids.append(input_id)
            images.append(image_tensor)
            image_sizes.append(image.size)
            
        input_ids = torch.stack(input_ids, dim=0).to(device=self._device)
        image_tensor = torch.stack(images, dim=0).to(device=self._device, dtype=self._torch_dtype)
            
        return {
            "inputs": input_ids,
            "images": image_tensor,
            "image_sizes": image_sizes,
        }