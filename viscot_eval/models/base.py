import os
import torch
from abc import abstractmethod, ABCMeta
from warppers import (
    time_logger, time_logger_set_active,
    memory_logger, memory_logger_set_active
)


class BaseInferModel(metaclass=ABCMeta):
    def __init__(self, 
                 base_model,
                 torch_dtype,
                 attn_implementation,
                 enable_time_logger,
                 enable_memory_logger,
                 **kwargs):
        self._base_model = base_model
        self._torch_dtype = getattr(torch, torch_dtype)
        self._attn_implementation = attn_implementation
        self._enable_time_logger = enable_time_logger
        self._enable_memory_logger = enable_memory_logger
        self._init_model(**kwargs)
        self._init_processor(**kwargs)
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self._device = f"cuda:{local_rank}"
    
    @property
    def model_config(self):
        return self._model.config
    
    @memory_logger
    @time_logger
    def do_generate(self, inputs, generation_config, do_selection):
        with time_logger_set_active(self._enable_time_logger), memory_logger_set_active(self._enable_memory_logger):
            return self._do_generate(inputs, generation_config, do_selection)
    
    @memory_logger
    @time_logger
    def do_glimpse(self, inputs, generation_config):
        with time_logger_set_active(self._enable_time_logger), memory_logger_set_active(self._enable_memory_logger):
            return self._do_glimpse(inputs, generation_config)
    
    @abstractmethod       
    def prepare_batch_inputs(self, batched_querys, batched_img_paths, batched_bboxes):
        pass
    
    @abstractmethod
    def _init_model(self, **kwargs):
        """
        Initialize the model.
        """
        pass
    
    @abstractmethod
    def _init_processor(self, **kwargs):
        """
        Initialize the processor.
        """
        pass
        
    @abstractmethod
    def _do_generate(self, inputs, generation_config, do_selection):
        pass
    
    @abstractmethod
    def _do_glimpse(self, inputs, generation_config):
        pass
    
    @abstractmethod
    def batch_decode(self, *args, **kwargs):
        pass
    

        
    
__all__ = [
    "BaseInferModel",
]