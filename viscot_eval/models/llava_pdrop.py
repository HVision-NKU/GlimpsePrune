import os
import ast
import torch

from llava_pdrop.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava_pdrop.conversation import conv_templates
from llava_pdrop.model.builder import load_pretrained_model
from llava_pdrop.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path


from .llava import LLaVA


class LLaVA_PDrop(LLaVA):
    def _init_model(self, 
                    layer_list: str, 
                    image_token_ratio_list: str, **kwargs):
        model_name = get_model_name_from_path(self._base_model)
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        # assert self._torch_dtype == torch.float16, "LLaVA_1.5 only supports float16 dtype for now."
        llava_model_args = {
            "multimodal": True,
            "pdrop_infer": True,
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
        
        self._model.model.layer_list = ast.literal_eval(layer_list)
        self._model.model.image_token_ratio_list = ast.literal_eval(image_token_ratio_list)
        self._model.model.image_token_ratio_list.insert(0, 1.0)
        self._model.eval()
        self._conv_mode = "vicuna_v1"
       