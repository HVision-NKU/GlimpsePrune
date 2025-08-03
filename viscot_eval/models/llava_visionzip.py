import os
import torch
from PIL import Image

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from visionzip import visionzip

from .llava import LLaVA


class LLaVA_VisionZip(LLaVA):
    def _init_model(self, 
                    dominant: int,
                    contextual: int,
                    **kwargs):
        model_name = get_model_name_from_path(self._base_model)
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
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

        self._model = visionzip(self._model, dominant=dominant, contextual=contextual)

        self._model.eval()
        self._conv_mode = "vicuna_v1"


__all__ = [
    "LLaVA_VisionZip",
]