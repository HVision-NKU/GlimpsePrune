import importlib
from typing import Type
from .base import BaseInferModel


AVAILABLE_MODELS = {
    "llava": "LLaVA",
    "llava_pdrop": "LLaVA_PDrop",
    "llava_divprune": "LLaVA_DivPrune",
    "llava_cdpruner": "LLaVA_CDPruner",
    "llava_vscan": "LLaVA_VScan",
    "llava_visionzip": "LLaVA_VisionZip",
    "llava_gp": "LLaVA_GP",
    "qwen2_5_vl": "Qwen2_5_VL",
    "qwen2_5_vl_sep": "Qwen2_5_VL_Sep",
    "qwen2_5_vl_gp": "Qwen2_5_VL_GP",
    "qwen2_5_vl_vscan": "Qwen2_5_VL_VScan",
    "qwen2_5_vl_visionzip": "Qwen2_5_VL_VisionZip",
}


# lazy load
def get_model(model_type) -> Type[BaseInferModel]:
    if model_type not in AVAILABLE_MODELS:
        raise ValueError(f"Model type '{model_type}' is not available, available models are: {list(AVAILABLE_MODELS.keys())}")
    path = f"viscot_eval.models.{model_type}"
    module = importlib.import_module(path)
    model_class = getattr(module, AVAILABLE_MODELS[model_type])
    return model_class
