import os
import torch
from PIL import Image

from llava_cdpruner.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava_cdpruner.conversation import conv_templates, SeparatorStyle
from llava_cdpruner.model.builder import load_pretrained_model
from llava_cdpruner.utils import disable_torch_init
from llava_cdpruner.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path


from .llava import LLaVA


class LLaVA_CDPruner(LLaVA):
    def _init_model(self, 
                    visual_token_num: int,
                    **kwargs):
        model_name = get_model_name_from_path(self._base_model)
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        # assert self._torch_dtype == torch.float16, "LLaVA_1.5 only supports float16 dtype for now."
        llava_model_args = {
            "multimodal": True,
            "attn_implementation": self._attn_implementation,
            "torch_dtype": self._torch_dtype,
            "visual_token_num": visual_token_num,
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
        
    def prepare_batch_inputs(self, batched_querys, batched_img_paths, batched_bboxes):
        # pass
        curr_batch_size = len(batched_querys)
        assert curr_batch_size == 1, "LLaVA_1.5 only supports batch size 1 for now."
        
        input_ids = []
        images = []
        image_sizes = []
        texts = []
        for one_query, one_img_path in zip(batched_querys, batched_img_paths):
            one_text = one_query.replace("\nAnswer the question using a single word or phrase.", "")
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
            texts.append(one_text)
            
        input_ids = torch.stack(input_ids, dim=0).to(device=self._device)
        image_tensor = torch.stack(images, dim=0).to(device=self._device, dtype=self._torch_dtype)
            
        return {
            "inputs": input_ids,
            "images": image_tensor,
            "image_sizes": image_sizes,
            "texts": texts[0],
        }
    
    def _do_generate(self, inputs, generation_config, do_selection):
        generate_ids = self._model.generate(
            inputs=inputs['inputs'],
            images=inputs['images'],
            image_sizes=inputs['image_sizes'],
            texts=inputs['texts'],
            generation_config=generation_config,
        )[0]  # [1] is visual_token_num
        return generate_ids