import os
import re
import ast
import math
import yaml
import warnings
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Any, Callable, Optional, Union, Sized, Dict, Tuple, List, Literal, Type

import numpy as np
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import datasets

from PIL import Image

from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from trl.models import unwrap_model_for_generation

from transformers import (
    TrainingArguments, 
    Trainer,
    GenerationConfig,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    is_safetensors_available, 
    is_peft_available
)

if is_safetensors_available():
    import safetensors.torch
from peft import PeftConfig, get_peft_model, PeftModel
from accelerate.utils import is_peft_model, set_seed

from qwen_vl_utils import process_vision_info

from transformers_gp.models.qwen2_5_vl import (
    Qwen2_5_VL_GP_ForConditionalGeneration,
    Qwen2_5_VL_GP_Processor,
    Qwen2_5_VL_GPConfig,
)

from transformers.trainer import (
    logger,
    TRAINING_ARGS_NAME,
    CONFIG_NAME,
    ADAPTER_WEIGHTS_NAME,
    ADAPTER_SAFE_WEIGHTS_NAME,
    WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    FSDP_MODEL_NAME,
)

from warppers import debug_calls
from utils import (
    LLMClient, 
    norm_bboxes, 
    extract_one_bbox_from_str, 
    cal_paired_ious,
    print_rank0
)


# ---------- Datasets ----------


QUERY_KEY = "query"
IMG_PATH_KEY = "img_path"
ANSWER_KEY = "answer"
NORMED_BBOXES_KEY = "normed_bboxes"
SCORE_FUNCS_KEY = "score_funcs"


REMAIN_KEYS = [
    QUERY_KEY,
    IMG_PATH_KEY,
    NORMED_BBOXES_KEY,
    ANSWER_KEY,
    SCORE_FUNCS_KEY,
]


MAPPER_REGISTRY = {}

FILTER_REGISTRY = {}


def register_mappers():
    def wrapper(func):
        name = func.__name__.replace("_dataset_mapper", "")
        MAPPER_REGISTRY[name] = func
        return func
    return wrapper


def register_filters():
    def wrapper(func):
        name = func.__name__.replace("_dataset_filter", "")
        FILTER_REGISTRY[name] = func
        return func
    return wrapper



@register_mappers()
def cot_train_dataset_mapper(one_data, **kwargs):
    query = one_data['question']
    if 'prompt' in kwargs:
        query = kwargs['prompt'].format(query)
    answer = one_data['answer']
    image = one_data['image']
    dataset = one_data['dataset']
    img_path = os.path.join(kwargs['img_dir'], "cot", dataset, image)
    bboxes = one_data['bboxs']
    # normed_bboxes = norm_bboxes(bboxes, height, width, bbox_type=kwargs['bbox_type'])
    
    return {
        QUERY_KEY: query,
        ANSWER_KEY: answer,
        IMG_PATH_KEY: img_path,
        NORMED_BBOXES_KEY: bboxes,
        SCORE_FUNCS_KEY: kwargs['score_funcs']
    }    
    

@register_mappers()
def cot_train_fullmask_dataset_mapper(one_data, **kwargs):
    query = one_data['question']
    if 'prompt' in kwargs:
        query = kwargs['prompt'].format(query)
    answer = one_data['answer']
    image = one_data['image']
    dataset = one_data['dataset']
    img_path = os.path.join(kwargs['img_dir'], "cot", dataset, image)
    normed_bboxes = [[0.0, 0.0, 1.0, 1.0]]
    
    return {
        QUERY_KEY: query,
        ANSWER_KEY: answer,
        IMG_PATH_KEY: img_path,
        NORMED_BBOXES_KEY: normed_bboxes,
        SCORE_FUNCS_KEY: kwargs['score_funcs']
    }    
    
    
@register_mappers()
def norm_bboxes_dataset_mapper(one_data, **kwargs):
    bboxes = one_data.pop(NORMED_BBOXES_KEY)
    if 'width' in one_data:
        width = one_data['width']
        height = one_data['height']
    else:
        img_path = one_data[IMG_PATH_KEY]
        img_pil = Image.open(img_path)
        width, height = img_pil.size
        img_pil.close()
    normed_bboxes = norm_bboxes(bboxes, height, width, bbox_type=kwargs['bbox_type'])
    one_data[NORMED_BBOXES_KEY] = normed_bboxes
    return one_data

    
@register_filters()
def image_exist_dataset_filter(one_data, **kwargs):
    img_path = one_data[IMG_PATH_KEY]
    try:
        img = Image.open(img_path)
        img.close()  # Close the image to free resources
        return True  # Image exists and is valid
    except (FileNotFoundError, OSError) as e:
        print_rank0(f"Image not found or invalid: {img_path}. Error: {e}")
        return False
    except Exception as e:
        print_rank0(f"Unexpected error while checking image: {img_path}. Error: {e}")
        return False
    
@register_filters()
def inputs_seq_length_dataset_filter(one_data, **kwargs):
    processor = kwargs['processor']
    max_input_seq_length = kwargs.get('max_input_seq_length', None)
    max_input_remain_seq_length = kwargs.get('max_input_remain_seq_length', None)
    if max_input_seq_length is None and max_input_remain_seq_length is None:
        return True
    img_path = one_data[IMG_PATH_KEY]
    query = one_data[QUERY_KEY]
    normed_bboxes = [one_data[NORMED_BBOXES_KEY]] if max_input_remain_seq_length is not None else None
    messages = [[{"role": "user", "content": [{"type": "image", "image": img_path}, {"type": "text", "text": query}]}]]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            normed_bboxes=normed_bboxes,
            padding=True,
            return_tensors="pt",
        )
    seq_length = inputs.input_ids.shape[1]
    if max_input_seq_length is not None and seq_length > max_input_seq_length:
        # print_rank0(f"Input sequence length {seq_length} exceeds max limit {max_input_seq_length}. Filtering out.")
        return False
    
    if max_input_remain_seq_length is not None:
        ref_token_masks = inputs.ref_token_masks[0]
        reduced_num = ref_token_masks.numel() - ref_token_masks.sum().item()
        remain_seq_length = seq_length - reduced_num
        if remain_seq_length > max_input_remain_seq_length:
            # print_rank0(f"Remaining sequence length {remain_seq_length} exceeds max limit {max_input_remain_seq_length}. Filtering out.")
            return False
    return True


# ---------- Loss ----------

LOSS_REGISTRY = {}

def register_loss(loss_class):
    """
    Decorator to register a loss class in the LOSS
    registry. The class should inherit from torch.nn.Module.
    """
    name = loss_class.__name__
    if name in LOSS_REGISTRY:
        raise ValueError(f"Loss class '{name}' is already registered.")
    LOSS_REGISTRY[name] = loss_class
    return loss_class


@register_loss
class DiceLoss(nn.Module):
    def __init__(self, epsilon: float = 1e-6, **kwargs):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, 
                image_token_mask_logits: List[torch.Tensor], 
                ref_token_masks: List[torch.Tensor]
               ) -> torch.Tensor:
        if not isinstance(image_token_mask_logits, list) or not isinstance(ref_token_masks, list):
            raise TypeError("Inputs must be lists of tensors.")
            
        if len(image_token_mask_logits) != len(ref_token_masks):
            raise ValueError(f"Input lists must have the same length, but got "
                             f"{len(image_token_mask_logits)} and {len(ref_token_masks)}")
            
        if len(image_token_mask_logits) == 0:
            # Handle empty batch case if necessary, e.g., return 0 loss or raise error
            return torch.tensor(0.0, device=image_token_mask_logits[0].device if image_token_mask_logits else None) 
            # Or raise ValueError("Input lists cannot be empty") depending on desired behavior

        batch_size = len(image_token_mask_logits)
        total_dice_loss = 0.0

        for i in range(batch_size):
            pred_mask_1d = image_token_mask_logits[i].flatten().sigmoid() # Shape: (N_b,) float
            # Flatten the ground truth mask and convert to float
            # Ensure it's on the same device as the prediction
            gt_mask_1d = ref_token_masks[i].flatten().to(pred_mask_1d.device, dtype=torch.float) # Shape: (N_b,) float

            # Calculate Dice components
            intersection = (pred_mask_1d * gt_mask_1d).sum()
            pred_sum = pred_mask_1d.sum()
            gt_sum = gt_mask_1d.sum() # Already float

            # Calculate Dice coefficient for this sample
            dice_coefficient = (2.0 * intersection + self.epsilon) / (pred_sum + gt_sum + self.epsilon)

            # Calculate Dice loss for this sample
            dice_loss_sample = 1.0 - dice_coefficient

            # Accumulate loss
            total_dice_loss += dice_loss_sample

        # Average loss over the batch
        average_dice_loss = total_dice_loss / batch_size
        return average_dice_loss


@register_loss
class BCELoss(nn.Module):
    def ___init__(self, **kwargs):
        super(BCELoss, self).__init__()
        
    def forward(self, 
                image_token_mask_logits: List[torch.Tensor], 
                ref_token_masks: List[torch.Tensor]
               ) -> torch.Tensor:
        
        batch_size = len(image_token_mask_logits)
        total_bce_loss = 0.0
        for i in range(batch_size):
            pred_mask_1d = image_token_mask_logits[i].flatten()
            # Flatten the ground truth mask and convert to float
            gt_mask_1d = ref_token_masks[i].flatten().to(pred_mask_1d.device)
            # Calculate BCE loss
            bce_loss = F.binary_cross_entropy_with_logits(
                pred_mask_1d.float(),
                gt_mask_1d.float(),
            )
            # Accumulate loss
            total_bce_loss += bce_loss
        # Average loss over the batch
        average_bce_loss = total_bce_loss / batch_size
        return average_bce_loss


@register_loss
class MaskLoss(nn.Module):
    def __init__(self, 
                 dice_weight: float = 0.5,
                 bce_weight: float = 0.5,
                 epsilon: float = 1e-6,
                 **kwargs):
        super().__init__()
        self.dice_loss = DiceLoss(epsilon=epsilon)
        self.bce_loss = BCELoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
    
    def forward(self, image_token_mask_logits: List[torch.Tensor],
                ref_token_masks: List[torch.Tensor]
               ) -> torch.Tensor:
        """
        Combines Dice Loss and BCE Loss for image token masks.
        
        Args:
            image_token_mask_logits (List[torch.Tensor]): List of predicted masks (1D tensors).
            ref_token_masks (List[torch.Tensor]): List of ground truth masks (2D tensors).
        
        Returns:
            torch.Tensor: Combined loss value.
        """
        dice_loss = self.dice_loss(image_token_mask_logits, ref_token_masks)
        bce_loss = self.bce_loss(image_token_mask_logits, ref_token_masks)
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss



# ---------- Dataset & Collator & Sampler ----------

class GPDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset that loads and combines multiple datasets
    based on a YAML configuration file. It handles sampling
    and applies specified mapping functions.
    """
    @classmethod
    def _load_config(cls, config_path: str) -> Dict[str, Any]:
        """Loads configuration from a YAML file."""
        print_rank0(f"Loading configuration from: {config_path}")
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            if config is None or 'datasets' not in config:
                 raise ValueError("YAML config is empty or missing 'datasets' key.")
            print_rank0("Configuration loaded successfully.")
            return config
        except FileNotFoundError:
            print_rank0(f"Error: Configuration file not found at {config_path}")
            raise
        except yaml.YAMLError as e:
            print_rank0(f"Error: Could not parse YAML configuration: {e}")
            raise
        except Exception as e:
            print_rank0(f"An unexpected error occurred during config loading: {e}")
            raise

    @classmethod
    def _apply_sampling(cls, dataset: datasets.Dataset, strategy: Optional[str], seed: Optional[int] = None) -> datasets.Dataset:
        """Applies sampling strategy to a dataset."""
        if not strategy:
            print_rank0("No sampling strategy specified, using full dataset.")
            return dataset

        try:
            parts = strategy.split(':')
            if len(parts) != 2:
                raise ValueError(f"Invalid sampling strategy format: '{strategy}'. Expected 'type:value'.")

            strat_type, strat_value = parts[0].lower(), parts[1]
            num_samples = int(strat_value)
            total_size = len(dataset)

            if num_samples <= 0:
                 raise ValueError(f"Sampling value must be positive, got: {num_samples} [{strategy}]")
            # Ensure sample size isn't larger than dataset, prevents errors in select/slice
            num_samples = min(num_samples, total_size)


            print_rank0(f"Applying sampling: {strategy} ({num_samples} samples) to dataset of size {total_size}")

            if strat_type == "first":
                return dataset.select(range(num_samples))
            elif strat_type == "end":
                 # Ensure we don't request more than available from the end
                start_index = max(0, total_size - num_samples)
                return dataset.select(range(start_index, total_size))
            elif strat_type == "random":
                if seed is None:
                    print_rank0("Warning: Random sampling without a fixed seed. Results may not be reproducible.")
                shuffled_dataset = dataset.shuffle(seed=seed)
                return shuffled_dataset.select(range(num_samples))
            else:
                print_rank0(f"Warning: Unknown sampling strategy type: '{strat_type}'. Using full dataset.")
                return dataset
        except ValueError as e:
            print_rank0(f"Error parsing sampling strategy '{strategy}': {e}. Using full dataset.")
            return dataset
        except Exception as e:
            print_rank0(f"An unexpected error occurred during sampling: {e}. Using full dataset.")
            return dataset
        
    @classmethod
    def _all_processed_datasets(cls, config, processor, args):
        all_processed_datasets: Dict[str, datasets.Dataset] = {}
        for i, dataset_config in enumerate(config['datasets']):
            print_rank0(f"\nProcessing dataset entry {i+1}/{len(config['datasets'])}...")
            json_path = dataset_config.get('json_path')
            base_name = '.'.join(os.path.basename(json_path).split('.')[:-1])
            dataset_name = dataset_config.get('dataset_name', base_name)
            if not json_path:
                print_rank0(f"Warning: Skipping dataset entry {i+1} due to missing 'json_path'.")
                continue

            sampling_strategy = dataset_config.get('sampling_strategy', None)
            mapper_name = dataset_config.get('mapper')
            bbox_type = dataset_config.get('bbox_type')
            img_dir = dataset_config.get('img_dir', args.img_dir)
            additional_mappers = dataset_config.get('additional_mappers', [])
            score_funcs = dataset_config.get('score_funcs', [])
            prompt = dataset_config.get('prompt', None)
            max_input_seq_length = dataset_config.get('max_input_seq_length', args.max_input_seq_length)
            max_input_remain_seq_length = dataset_config.get('max_input_remain_seq_length', args.max_input_remain_seq_length)
            
            for score_func in score_funcs:
                assert score_func in SCORE_REGISTRY, f"Score function '{score_func}' not registered. Available: {list(SCORE_REGISTRY.keys())}"
            
            try:
                print_rank0(f"Loading raw data from: {json_path}")
                # Assuming JSON Lines format, common with `datasets`
                raw_dataset = datasets.load_dataset('json', data_files=json_path, split='train')
                print_rank0(f"Loaded {len(raw_dataset)} examples raw.")

                # Apply sampling
                sampled_dataset = cls._apply_sampling(raw_dataset, sampling_strategy, args.sampling_seed)
                if len(sampled_dataset) == 0:
                    print_rank0("Dataset is empty after sampling, skipping.")
                    continue
                print_rank0(f"Dataset size after sampling: {len(sampled_dataset)}")

                # Apply mapping
                mapper_func = MAPPER_REGISTRY[mapper_name]
                print_rank0(f"Applying mapper: '{mapper_name}'")
                # Prepare arguments for the mapper function
                mapper_kwargs = {
                    'img_dir': img_dir,
                    'score_funcs': score_funcs,
                }
                if prompt is not None:
                    mapper_kwargs['prompt'] = prompt
                print_rank0(f"Mapper arguments: {mapper_kwargs}")
                processed_dataset = sampled_dataset.map(
                    mapper_func,
                    num_proc=8,
                    fn_kwargs=mapper_kwargs,
                )

                processed_dataset = processed_dataset.remove_columns(
                    [col for col in processed_dataset.column_names if col not in REMAIN_KEYS]
                )
                    
                # Filtering
                print_rank0("Applying dataset filter: 'image_exist_dataset_filter'")
                processed_dataset = processed_dataset.filter(
                    image_exist_dataset_filter,
                    num_proc=8,
                    fn_kwargs={}
                )
                print_rank0(f"Processed dataset size after image_exist_dataset_filter: {len(processed_dataset)}")
                
                # Additional filtering
                if max_input_seq_length is not None or max_input_remain_seq_length is not None:
                    processed_dataset = processed_dataset.filter(
                        inputs_seq_length_dataset_filter,
                        num_proc=8,
                        fn_kwargs={
                            'processor': processor,
                            'max_input_seq_length': max_input_seq_length,
                            'max_input_remain_seq_length': max_input_remain_seq_length,
                        }
                    )
                    print_rank0(f"Processed dataset size after inputs_seq_length_dataset_filter: {len(processed_dataset)}")
                
                # Additional mapping
                for additional_mapper in additional_mappers:
                    mapper_func = MAPPER_REGISTRY[additional_mapper]
                    print_rank0(f"Applying additional mapper: '{additional_mapper}'")
                    processed_dataset = processed_dataset.map(
                        mapper_func,
                        num_proc=8,
                        fn_kwargs={
                            'bbox_type': bbox_type,
                        }
                    )
                print_rank0(f"Processed dataset size: {len(processed_dataset)}")
                if len(processed_dataset) == 0:
                    print_rank0(f"Warning: Processed dataset {dataset_name} is empty after mapping. Skipping.")
                    continue
                # Store the processed dataset
                if dataset_name in all_processed_datasets:
                    dataset_name_with_uuid = f"{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    print_rank0(f"Warning: Dataset name '{dataset_name}' already exists. Renaming to '{dataset_name_with_uuid}'")
                    all_processed_datasets[dataset_name_with_uuid] = processed_dataset
                else:
                    all_processed_datasets[dataset_name] = processed_dataset                

            except FileNotFoundError:
                print_rank0(f"Error: Data file not found for dataset entry {i+1}: {json_path}. Skipping.")
            except Exception as e:
                print_rank0(f"Error processing dataset entry {i+1} ({json_path}): {e}. Skipping.")
                
        return all_processed_datasets
        

    def __init__(self, config_path: str, processor: Qwen2_5_VL_GP_Processor, script_args: Optional[Any] = None):
        """
        Initializes the GPDataset.

        Args:
            config_path (str): Path to the YAML configuration file.
            processor (Qwen2_5_VL_GP_Processor): Processor for handling text and vision data.
            script_args (Any, optional): Additional arguments passed from the script
                                         (e.g., training args, could contain seed). Defaults to None.
        """
        super().__init__()
        self.args = script_args
        self.config = self._load_config(config_path)
        self.processor = processor
        all_processed_datasets = self._all_processed_datasets(self.config, self.processor, self.args)
        # Combine all processed datasets
        if all_processed_datasets:
            print_rank0(f"\nConcatenating {len(all_processed_datasets)} processed dataset(s)...")
            # Note: Concatenation works best if all datasets have the exact same features/columns.
            # The `map` function should ensure consistent output structure.
            # Consider using `features=...` argument in `concatenate_datasets` if schemas might differ slightly
            # and you know how to resolve them.
            self.final_dataset = datasets.concatenate_datasets(list(all_processed_datasets.values()))
            if len(self.final_dataset) == 0:
                raise ValueError("Final dataset is empty after concatenation.")
            print_rank0(f"Final combined dataset size: {len(self.final_dataset)}")
            # Optionally print final features/columns
            print_rank0(f"Final dataset features: {self.final_dataset.features}")
        else:
            # print_rank0("No datasets were successfully processed.")
            raise ValueError("No datasets were successfully processed. Please check your configuration.")
            self.final_dataset = None

    def __len__(self) -> int:
        """Returns the total number of samples in the combined dataset."""
        return len(self.final_dataset) if self.final_dataset else 0

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Retrieves a single sample from the combined dataset."""
        if self.final_dataset is None:
            raise IndexError("Dataset is not initialized or is empty.")
        if not 0 <= index < len(self.final_dataset):
             raise IndexError(f"Index {index} out of bounds for dataset of size {len(self.final_dataset)}")
        # `datasets` objects behave like lists/dicts for access
        return self.final_dataset[index]
    
    
    @classmethod
    def get_processed_dataset_dict(cls, config_path: str, processor: Qwen2_5_VL_GP_Processor, script_args: Optional[Any] = None) -> Dict[str, datasets.Dataset]:
        """
        Class method to get processed datasets based on the YAML configuration.

        Args:
            config_path (str): Path to the YAML configuration file.
            script_args (Any, optional): Additional arguments passed from the script
                                         (e.g., training args). Defaults to None.

        Returns:
            Dict[str, datasets.Dataset]: Dictionary of processed datasets.
        """
        config = cls._load_config(config_path)
        all_processed_datasets = cls._all_processed_datasets(config, processor, script_args)
        return all_processed_datasets



class GPCollator:
    def __init__(self, processor, is_sft):
        self.processor = processor
        self.is_sft = is_sft
        self.im_start_id = self.processor.tokenizer.encode("<|im_start|>")[0]
        
    def _prepare_labels_from_input_ids(self, input_ids):
        B, L = input_ids.shape
        labels = input_ids.clone()
        mask = input_ids == self.im_start_id
        flipped_mask = mask.flip(dims=(1,))  # Reverse the mask to find the last <|im_start|> token
        first_idx_in_flipped = torch.argmax(flipped_mask.int(), dim=1)
        last_pos = (L - 1) - first_idx_in_flipped
        mask_until_idx = last_pos + 3
        mask_until_idx = torch.clamp(mask_until_idx, max=L)
        
        arange_l = torch.arange(L, device=input_ids.device).expand(B, -1)
        modification_mask = arange_l < mask_until_idx.unsqueeze(1)
        
        labels[modification_mask] = -100   # ignore index of CrossEntropyLoss
        return labels
        
    
    def __call__(self, features):
        messages = []
        normed_bboxes = []
        answers = []
        querys = []
        score_funcs = []
        for feature in features:
            query = feature[QUERY_KEY]
            answer = feature[ANSWER_KEY]
            img_path = feature[IMG_PATH_KEY]
            if self.is_sft:
                messages.append([{"role": "user", "content": [{"type": "image", "image": img_path}, {"type": "text", "text": query}]}, {"role": "assistant", "content": [{"type": "text", "text": answer}]}])
            else:
                messages.append([{"role": "user", "content": [{"type": "image", "image": img_path}, {"type": "text", "text": query}]}])
            normed_bboxes.append(feature[NORMED_BBOXES_KEY])
            querys.append(query)
            answers.append(answer)
            score_funcs.append(feature[SCORE_FUNCS_KEY])
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=(not self.is_sft)
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=text,
            normed_bboxes=normed_bboxes,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        if self.is_sft:
            labels = self._prepare_labels_from_input_ids(inputs.input_ids)
            inputs["labels"] = labels
        
        inputs[QUERY_KEY] = querys
        inputs[ANSWER_KEY] = answers
        inputs[SCORE_FUNCS_KEY] = score_funcs
        return inputs
            

class RepeatRandomSampler(torch.utils.data.Sampler):
    """
    Sampler that repeats the indices of a dataset in a structured manner.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        mini_repeat_count (`int`):
            Number of times to repeat each index per batch.
        batch_size (`int`, *optional*, defaults to `1`):
            Number of unique indices per batch.
        repeat_count (`int`, *optional*, defaults to `1`):
            Number of times to repeat the full sampling process.
        seed (`int` or `None`, *optional*, defaults to `None`):
            Random seed for reproducibility.
    """

    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.seed = seed
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self):
        indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()
        indexes = [indexes[i : i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]

        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return self.num_samples * self.mini_repeat_count * self.repeat_count

# ---------- Client & Score Functions ----------
SCORE_REGISTRY = {}

def register_score():
    def wrapper(func):
        name = func.__name__.replace("_score", "")
        SCORE_REGISTRY[name] = func
        return func
    return wrapper


@register_score()
def llm_score(query, completion, answer, args):
    client = LLMClient(base_url=args.client_base_url, api_key=args.client_api_key, model_name=args.client_model_name)
    return client.score(query, completion, answer)


@register_score()
def precision_match_or_llm_score(query, completion, answer, args):
    """
    This score function first checks if the completion matches the answer.
    If it does, it returns the LLM score; otherwise, it returns 0.
    """
    client = LLMClient(base_url=args.client_base_url, api_key=args.client_api_key, model_name=args.client_model_name)
    scores = []
    for one_query, one_completion, one_answer in zip(query, completion, answer):
        if one_completion.strip().lower() == one_answer.strip().lower():
            scores.append(1.0)
        else:
            scores.append(client.score([one_query], [one_completion], [one_answer])[0])
    return scores
    
@register_score()
def precision_match_score(query, completion, answer, args):
    """
    This score function checks if the completion matches the answer.
    Returns 1.0 if they match, otherwise returns 0.0.
    """
    scores = []
    for one_query, one_completion, one_answer in zip(query, completion, answer):
        if one_completion.strip().lower() == one_answer.strip().lower():
            scores.append(1.0)
        else:
            scores.append(0.0)
    return scores

@register_score()
def one_box_iou_score(query, completion, answer, args):
    pred_bboxes = [extract_one_bbox_from_str(one_str) for one_str in completion]
    gt_bboxes = [ast.literal_eval(one_answer) for one_answer in answer]
    ious = cal_paired_ious(np.array(pred_bboxes), np.array(gt_bboxes))
    return ious.tolist()

@register_score()
def one_box_format_score(query, completion, answer, args):
    bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
    # Score=1 only if there exists only one bbox in the completion and it matches the pattern
    scores = []
    for one_completion in completion:
        matches = re.findall(bbox_pattern, one_completion)
        if len(matches) == 1:
            scores.append(1.0)  # Correct format
        else:
            scores.append(0.0)
    return scores

@register_score()
def single_choice_score(query, completion, answer, args):
    patterns = [
            r'(?:(?:the|my|the correct)\s+)?(?:answer|choice|option)\s*(?:is)?\s*[:：]?\s*([A-Z])',
            r'\(([A-Z])\)',
            r'\b([A-Z])[\.\)]',
            r'^([A-Z])\b',
            r'\b([A-Z])\b'
        ]
    scores = []
    for one_completion, one_answer in zip(completion, answer):
        one_answer = one_answer.strip().upper()
        extracted_completion = None
        for pattern in patterns:
            match = re.search(pattern, one_completion, re.IGNORECASE)
            if match:
                extracted_completion = match.group(1).strip().upper()
                break
        if extracted_completion and extracted_completion == one_answer:
            scores.append(1.0)
        else:
            scores.append(0.0)
    return scores
        

# ---------- Parameter Scheduler ----------
class BaseScheduler:
    def __init__(self, min_value: float, max_value: float, total_steps: int):
        if total_steps < 1:
            raise ValueError("total_steps must be at least 1.")
        self.min_value = min_value
        self.max_value = max_value
        self.total_steps = total_steps

    def get_value(self, current_step: int) -> float:
        raise NotImplementedError("This method should be implemented by subclasses.")

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"min_value={self.min_value}, "
                f"max_value={self.max_value}, "
                f"total_steps={self.total_steps})")

SCHEDULER_REGISTRY = {}

def register_scheduler(name: Optional[str] = None):
    def _decorator(cls: Type['BaseScheduler']) -> Type['BaseScheduler']:
        if name is not None:
            key = name
        else:
            # 'LinearScheduler' -> 'linear_scheduler'
            key = re.sub(r'(?<!^)(?=[A-Z])', '_', cls.__name__).lower()

        if key in SCHEDULER_REGISTRY:
            raise ValueError(f"Error: Scheduler '{key}' is already registered. ")
        SCHEDULER_REGISTRY[key] = cls
        return cls

    return _decorator
        
    
def create_scheduler(
    scheduler_name: str,
    min_value: float,
    max_value: float,
    total_steps: int
) -> BaseScheduler:
    if scheduler_name not in SCHEDULER_REGISTRY:
        raise ValueError(f"Scheduler '{scheduler_name}' is not registered. Available: {list(SCHEDULER_REGISTRY.keys())}")
    
    return SCHEDULER_REGISTRY[scheduler_name](min_value, max_value, total_steps)


@register_scheduler("linear")
class LinearScheduler(BaseScheduler):

    def get_value(self, current_step: int) -> float:
        current_step = min(current_step, self.total_steps - 1)
        if self.total_steps == 1:
            return self.min_value
        progress = current_step / (self.total_steps - 1)
        return self.max_value - (self.max_value - self.min_value) * progress


@register_scheduler("cosine")
class CosineAnnealingScheduler(BaseScheduler):

    def get_value(self, current_step: int) -> float:
        current_step = min(current_step, self.total_steps - 1)

        if self.total_steps == 1:
            return self.min_value
            
        cosine_progress = 0.5 * (1 + math.cos(math.pi * current_step / (self.total_steps - 1)))
        
        return self.min_value + (self.max_value - self.min_value) * cosine_progress


@register_scheduler("exponential")
class ExponentialScheduler(BaseScheduler):

    def __init__(self, min_value: float, max_value: float, total_steps: int):
        super().__init__(min_value, max_value, total_steps)
        if self.total_steps == 1 or self.max_value == 0:
            self.gamma = 0
        else:
            epsilon = 1e-9
            safe_min_value = max(self.min_value, epsilon)
            self.gamma = (safe_min_value / self.max_value) ** (1 / (self.total_steps - 1))

    def get_value(self, current_step: int) -> float:
        current_step = min(current_step, self.total_steps - 1)
        
        value = self.max_value * (self.gamma ** current_step)
        return max(value, self.min_value)
    
    
# ---------- Trainer ----------


def convert_to_left_padding(
    input_ids: torch.LongTensor,
    inputs_embeds: Optional[torch.FloatTensor],
    attention_mask: torch.LongTensor,
    position_ids: Optional[torch.LongTensor],
    completion_mask: torch.LongTensor,
    max_seq_length: Optional[int] = None,
) -> Tuple[torch.LongTensor, Optional[torch.FloatTensor], torch.LongTensor, Optional[torch.LongTensor], torch.LongTensor]:
    B = input_ids.shape[0]
    C = inputs_embeds.shape[2] if inputs_embeds is not None else 0 # Embedding dimension
    P = position_ids.shape[0] if position_ids is not None else 0 # First dimension of position_ids (e.g., 3)
    # max_completion_length = completion_mask.shape[1]  # Length of completion mask
    valid_completion_lens = completion_mask.sum(dim=1).long()
    
    device = input_ids.device # Use device of one of the inputs

    # 1. Calculate original effective lengths (number of non-padding tokens)
    original_effective_lengths = attention_mask.sum(dim=1).long() # Ensure long type for consistency

    # 2. Determine the length to keep for each sequence after applying max_seq_length
    if max_seq_length is not None:
        # Ensure max_seq_length is a tensor for broadcasting
        original_max_lengths = original_effective_lengths.max().item() # Get the maximum original length
        if max_seq_length < original_max_lengths:
            warnings.warn(
                f"max_seq_length ({max_seq_length}) is less than the maximum original effective length "
                f"({original_max_lengths}). Sequences will be truncated."
            )
            valid_completion_lens -= (original_max_lengths - max_seq_length)
            if torch.all(valid_completion_lens < 0):
                warnings.warn(
                    "All sequences will be truncated to zero length. Consider increasing max_seq_length."
                )
            elif torch.any(valid_completion_lens < 0):
                warnings.warn(
                    "Some sequences will be truncated to zero length. Consider increasing max_seq_length."
                )
            # valid_completion_lens = torch.maximum(valid_completion_lens, torch.zeros_like(valid_completion_lens))
            
        max_len_tensor = torch.tensor(max_seq_length, device=device, dtype=torch.long)
        lengths_to_keep = torch.minimum(original_effective_lengths, max_len_tensor)
    else:
        lengths_to_keep = original_effective_lengths
    
    if max_seq_length == 0:
        lengths_to_keep = torch.zeros_like(original_effective_lengths)

    if lengths_to_keep.numel() > 0: # Check if lengths_to_keep is not empty
        output_L = lengths_to_keep.max().item()
    else:
        output_L = 0
    
    new_input_ids = torch.zeros((B, output_L), dtype=input_ids.dtype, device=device)
    if inputs_embeds is not None:
        new_inputs_embeds = torch.zeros((B, output_L, C), dtype=inputs_embeds.dtype, device=device)
    else:
        new_inputs_embeds = None
    new_attention_mask = torch.zeros((B, output_L), dtype=attention_mask.dtype, device=device)
    if position_ids is not None:
        new_position_ids = torch.zeros((P, B, output_L), dtype=position_ids.dtype, device=device)
    else:
        new_position_ids = None
    new_completion_mask = torch.zeros((B, output_L), dtype=completion_mask.dtype, device=device)

    # 5. Fill the new tensors
    for i in range(B):
        # Number of tokens to keep for sequence i (after potential truncation)
        len_to_keep_i = lengths_to_keep[i].item()

        if len_to_keep_i == 0: # If sequence becomes empty, skip copying
            continue

        num_left_pads = output_L - len_to_keep_i
        num_completion_len = valid_completion_lens[i].item()
        
        # Create a boolean mask for valid tokens in the original sequence i
        # Assuming attention_mask[i] is 1D and has L_orig elements
        original_valid_token_mask_i = attention_mask[i].bool()

        # --- input_ids ---
        original_valid_ids_i = input_ids[i][original_valid_token_mask_i]
        ids_to_copy = original_valid_ids_i[:len_to_keep_i] # Take last 'len_to_keep_i' valid tokens
        new_input_ids[i, num_left_pads:] = ids_to_copy

        # --- inputs_embeds ---
        if new_inputs_embeds is not None:
            original_valid_embeds_i = inputs_embeds[i][original_valid_token_mask_i, :]
            embeds_to_copy = original_valid_embeds_i[:len_to_keep_i, :]
            new_inputs_embeds[i, num_left_pads:, :] = embeds_to_copy
        
        # --- attention_mask ---
        # The new attention mask is '1' for all kept tokens
        new_attention_mask[i, num_left_pads:] = 1

        # --- position_ids ---
        if new_position_ids is not None:
            # position_ids[:, i] gives shape (P, L_orig)
            original_valid_pos_ids_i = position_ids[:, i][:, original_valid_token_mask_i] # Shape (P, num_original_valid)
            pos_ids_to_copy = original_valid_pos_ids_i[:, :len_to_keep_i] # Shape (P, len_to_keep_i)
            new_position_ids[:, i, num_left_pads:] = pos_ids_to_copy
        
        # --- completion_mask ---
        if num_completion_len > 0:
            new_completion_mask[i, -num_completion_len:] = 1  # Set the last num_completion_len tokens to 1
            
    return new_input_ids, new_inputs_embeds, new_attention_mask, new_position_ids, new_completion_mask


class GPTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self._loc_conf_mat_list = []
        self._metrics = defaultdict(list)
        super().__init__(*args, **kwargs)
        self.reward_weight = self.args.reward_weight
        self.kd_weight = self.args.kd_weight
        self.loc_weight = self.args.loc_weight
        self.le_weight = self.args.le_weight
        self.max_completion_length = self.args.max_completion_length
        self.max_seq_length = self.args.max_seq_length
        self.gen_mask_usage_ratio = self.args.gen_mask_usage_ratio
        
        if self.reward_weight <= 0:
            self.num_iterations = 1
            self.num_generations = 1
        else:
            self.num_iterations = self.args.num_iterations
            self.num_generations = self.args.num_generations
            
        if self.num_iterations > 1:
            raise NotImplementedError()
            # self._buffered_inputs = [None] * self.args.gradient_accumulation_steps
        
        if self.loc_weight > 0:
            self.loc_criterion = LOSS_REGISTRY[self.args.loc_loss_class](dice_weight=self.args.loc_dice_weight, bce_weight=self.args.loc_bce_weight)
        
        if self.reward_weight > 0:
            num_processes = self.accelerator.num_processes
            global_batch_size = self.args.per_device_train_batch_size * num_processes
            possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
            if self.num_generations not in possible_values:
                raise ValueError(
                    f"The global train batch size ({num_processes} x {self.args.per_device_train_batch_size}) must be evenly "
                    f"divisible by the number of generations per prompt ({self.num_generations}). Given the current train "
                    f"batch size, the valid values for the number of generations are: {possible_values}."
                )
            if self.num_generations != global_batch_size:
                raise NotImplementedError("TODO: update score_dict gather")
        
        self._step = 0  # forward times
            
        set_seed(self.args.seed, device_specific=True)
        
    
    def _create_completion_mask(self, completion_ids: torch.Tensor, eos_token_id: int) -> torch.Tensor:
        B, L = completion_ids.shape
        device = completion_ids.device
        is_eos = (completion_ids == eos_token_id)
        padded_is_eos = torch.cat(
            [torch.zeros((B, 1), dtype=torch.int, device=device), is_eos.int()],
            dim=1
        )
        eos_cumsum = torch.cumsum(padded_is_eos, dim=1)
        cumulative_eos_before_current = eos_cumsum[:, :-1]
        completion_masks = (cumulative_eos_before_current == 0).long()
        return completion_masks

        
    def _extract_valid_completion_logps(self, per_token_logps, valid_completion_lens):
        """
        Extracts the valid completion log probabilities based on the provided lengths.
        Assumes that the completions are on the most right side of the tensor.

        Args:
            per_token_logps (torch.Tensor): Tensor of log probabilities. Shape: [B, L].
            valid_completion_lens (torch.Tensor): Tensor of valid lengths for each sequence. Shape: [B].

        Returns:
            completion_per_token_logps (List[torch.Tensor]): List of tensors, each containing the valid log probabilities for a sequence.
        """
        completion_per_token_logps = []
        for i, valid_len in enumerate(valid_completion_lens):
            # Extract the valid log probabilities for the current sequence
            if valid_len == 0:
                valid_logps = torch.zeros(0, dtype=per_token_logps.dtype, device=per_token_logps.device)
                completion_per_token_logps.append(valid_logps)
            else:
                valid_logps = per_token_logps[i, -valid_len:]
                completion_per_token_logps.append(valid_logps)
        return completion_per_token_logps
        

    def _update_ref_token_masks(self, model, inputs, image_token_mask_logits):
        # TODO: during training, the gen mask in all batch should be same
        original_ref_token_masks = inputs['ref_token_masks']
        use_gen_mask_by_necessity = [mask is None for mask in original_ref_token_masks]
        if sum(use_gen_mask_by_necessity) == 0 and self.gen_mask_usage_ratio <= 0:
            return [False] * len(original_ref_token_masks)
        use_gen_mask = [mask is None or torch.rand(1).item() < self.gen_mask_usage_ratio for mask in original_ref_token_masks]
        
        if image_token_mask_logits is None:
            with torch.no_grad():
                with self.accelerator.unwrap_model(model, keep_fp32_wrapper=False).disable_adapter():
                    rtn = model(
                        **inputs,
                        do_selection=True,
                        delay_selection=True,
                        use_cache=False,
                        return_dict=True,
                    )
                    image_token_mask_logits = rtn.image_token_mask_logits
                    del rtn
                    self.accelerator.unwrap_model(model).reset_image_tokens_cache()
        new_ref_token_masks = []
        for i, original_mask in enumerate(original_ref_token_masks):
            if use_gen_mask[i]:
                new_ref_token_masks.append(image_token_mask_logits[i][-1:].detach().sigmoid())
            else:
                new_ref_token_masks.append(original_mask)
        inputs['ref_token_masks'] = new_ref_token_masks
        return use_gen_mask
            
    
    def _generate_and_score_completions(self, model, inputs):
        queries = inputs.pop(QUERY_KEY)
        answers = inputs.pop(ANSWER_KEY)
        score_func_names = inputs.pop(SCORE_FUNCS_KEY)[0]  # datas in the same batch are same
        assert len(score_func_names) > 0, "No score functions provided in the batch."
        score_dict = {}
        for score_func_name in score_func_names:
            score_dict[score_func_name] = None
        
        temperature = 1.0
        generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            temperature=temperature,
        )
        num_generations = self.num_generations
        bsz = inputs['input_ids'].shape[0]
        
        if self.loc_weight > 0 or self.le_weight > 0:
            outputs = model(
                **inputs,
                do_selection=True,
                use_cache=False,
                return_dict=True,
                delay_selection=True,
            )
            image_token_mask_logits = outputs.image_token_mask_logits
            le_loss = outputs.le_loss
            self.accelerator.unwrap_model(model).reset_image_tokens_cache()
        else:
            image_token_mask_logits = None
            le_loss = None
            
        output_dict = {
            'le_loss': le_loss,
            'image_token_mask_logits': image_token_mask_logits,
        }
        if self.reward_weight <= 0:
            reduce_ratios = []
            for one_mask in outputs.image_token_mask_logits:
                one_mask = one_mask[-1].sigmoid() > 0.5
                reduce_ratios.append(one_mask.sum().item() / (one_mask.shape[0]))
            self._log_datas(queries, answers, reduce_ratios=reduce_ratios)
            return output_dict
        
        if self.kd_weight > 0:
            # get ref_first_token_logits
            with torch.no_grad():
                with self.accelerator.unwrap_model(model, keep_fp32_wrapper=False).disable_adapter():
                    ref_outputs = model(
                        **inputs,
                        do_selection=False,
                        use_cache=False,
                        return_dict=True,
                    )
            ref_first_token_log_probs = torch.log_softmax(ref_outputs.logits[:, -1, :], dim=-1)  # (B, V)
            del ref_outputs
            torch.cuda.empty_cache()
        else:
            ref_first_token_log_probs = None 
        
        use_gen_masks = self._update_ref_token_masks(model, inputs, image_token_mask_logits)
        
        outputs = model(
            **inputs,
            do_selection=True,
            use_cache=True,
            return_dict=True,
            delay_selection=True,
            use_ref_masks=True,
        )   # outputs before selection
        
        reduce_ratios = []
        for one_mask in outputs.image_token_mask_logits:
            one_mask = one_mask[-1].sigmoid() > 0.5
            reduce_ratios.append(one_mask.sum().item() / (one_mask.shape[0]))
        
        inputs['image_token_mask_logits'] = outputs.image_token_mask_logits
        
        try:
            outputs = model(
                **inputs,
                do_selection=True,
                use_cache=True,
                return_dict=True,
                delay_selection=True,
            )
        except torch.cuda.OutOfMemoryError:
            print(f"querys: {queries}")
            print(f"answers: {answers}")
            print(f"input_ids.shape: {inputs.input_ids.shape}")
            print(f"reduce_ratios: {reduce_ratios}")
            raise
        
        first_token_logits = outputs.logits[:, -1, :] # (B, V)
        probs = torch.softmax(first_token_logits / temperature, dim=-1)
        
        
        # generate samples
        with torch.no_grad():
            with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                # immitate the first token generation
                input_ids = outputs.input_ids
                inputs_embeds = outputs.inputs_embeds
                cache_position = torch.ones_like(input_ids[0, :], dtype=torch.int64).cumsum(0) - 1
                model_kwargs = {
                    'use_cache': True,
                    'cache_position': cache_position,
                }
                model_kwargs = unwrapped_model._update_model_kwargs_for_generation(outputs, model_kwargs)
                inputs.update(model_kwargs)
                first_token_ids = torch.multinomial(probs, num_samples=1)  # (B, 1)
                input_ids = torch.cat([input_ids, first_token_ids], dim=1)  # (B, L + 1)
                inputs_embeds = torch.cat([inputs_embeds, unwrapped_model.text_embed_forward(first_token_ids)], dim=1)  # (B, L + 1, C)
                inputs['input_ids'] = input_ids
                inputs['inputs_embeds'] = inputs_embeds
                generated_ids = unwrapped_model.generate(
                    **inputs, 
                    do_selection=True, 
                    delay_selection=True,
                    generation_config=generation_config
                )
            in_seq_len = outputs.input_ids.shape[1]
            # inputs.past_key_values.crop(in_seq_len)  # [0, in_seq_len) do not include the first token
            completion_ids_from_first = generated_ids[:, in_seq_len:]  # right padding
            completion_texts = self.processing_class.batch_decode(completion_ids_from_first, skip_special_tokens=True)
            for score_name in score_dict:
                score_dict[score_name] = SCORE_REGISTRY[score_name](queries, completion_texts, answers, self.args)
            
        completion_mask = self._create_completion_mask(completion_ids_from_first, self.processing_class.eos_token_id)
        max_completion_seq_len = completion_ids_from_first.shape[1]
        # assert max_completion_seq_len == completion_mask.sum(dim=1).max().item(), f"max_completion_seq_len ({max_completion_seq_len}) does not match the maximum valid completion length ({completion_mask.sum(dim=1).max().item()})"
        if max_completion_seq_len != completion_mask.sum(dim=1).max().item():
            warnings.warn(
                f"max_completion_seq_len ({max_completion_seq_len}) does not match the maximum valid completion length ({completion_mask.sum(dim=1)}).\ncompletion_ids_from_first: {completion_ids_from_first}\ncompletion_mask: {completion_mask}"
            )
            # TODO: why?
        
        completion_position_ids = outputs.position_ids[:, :, -1:].expand(-1, -1, max_completion_seq_len)
        add_position_ids = torch.arange(1, max_completion_seq_len + 1, dtype=completion_position_ids.dtype, device=completion_position_ids.device).view(1, 1, -1)
        completion_position_ids = completion_position_ids + add_position_ids
        # completion_inputs_embeds = model.model.embed_tokens(completion_ids)
        completion_inputs_embeds = self.accelerator.unwrap_model(model).text_embed_forward(completion_ids_from_first)
        all_inputs_ids = torch.cat([outputs.input_ids, completion_ids_from_first], dim=1)
        all_inputs_embeds = torch.cat([outputs.inputs_embeds, completion_inputs_embeds], dim=1)
        all_attention_mask = torch.cat([outputs.attention_mask, completion_mask], dim=1)
        all_position_ids = torch.cat([outputs.position_ids, completion_position_ids], dim=2)
        del completion_ids_from_first, completion_inputs_embeds, completion_position_ids
        
        self._metrics["completion_length"].append(self.accelerator.gather_for_metrics(completion_mask.sum(dim=1).float()).mean().item())
        
        all_inputs_ids, all_inputs_embeds, all_attention_mask, all_position_ids, completion_mask = convert_to_left_padding(
            all_inputs_ids, all_inputs_embeds, all_attention_mask, all_position_ids, completion_mask, self.max_seq_length
        )
        
        valid_completion_lens = completion_mask.sum(dim=1)
        max_completion_seq_len = valid_completion_lens.max().item()
        if max_completion_seq_len <= 0:
            completion_ids_from_first = torch.zeros((bsz, 0), dtype=all_inputs_ids.dtype, device=all_inputs_ids.device)  # left padding
        else:
            completion_ids_from_first = all_inputs_ids[:, -max_completion_seq_len:]
        # for debug
        completion_ids_for_debug = [one_input_ids[one_mask.bool()] for one_input_ids, one_mask in zip(all_inputs_ids, completion_mask)]
        completion_texts_for_debug = self.processing_class.batch_decode(completion_ids_for_debug, skip_special_tokens=True)
        
        
        self.accelerator.unwrap_model(model).reset_image_tokens_cache()
        torch.cuda.empty_cache()
        
        self._log_datas(
            queries, answers, 
            completion_texts=completion_texts, 
            completion_texts_for_debug=completion_texts_for_debug,
            scores=score_dict,
            all_inputs_ids_shape=all_inputs_ids.shape,
            reduce_ratios=reduce_ratios,
            use_gen_masks=use_gen_masks,
        )
        self._log_cuda_memeory()
        
        
        completion_logits = model(
            inputs_embeds=all_inputs_embeds,
            position_ids=all_position_ids,
            attention_mask=all_attention_mask,
            use_cache=False,
            do_selection=False,
        )[0]
        
        if max_completion_seq_len > 1:
            completion_logits = completion_logits[:, -max_completion_seq_len:-1]  # logits from the second token to the last eos token， left-padding
            completion_log_probs = torch.log_softmax(completion_logits, dim=-1)  # (B, L, V)
            completion_ids_from_second = all_inputs_ids[:, -max_completion_seq_len+1:]  # tokens from the second token to the last eos token, left-padding
            completion_logps = torch.gather(completion_log_probs, dim=-1, index=completion_ids_from_second.unsqueeze(-1)).squeeze(-1)
            completion_logps = self._extract_valid_completion_logps(completion_logps, valid_completion_lens - 1)
        else:
            completion_logps = [torch.zeros(0, dtype=first_token_logits.dtype, device=first_token_logits.device) for _ in range(bsz)]
            
        if ref_first_token_log_probs is not None:
            with torch.no_grad():
                ref_first_token_logps = torch.gather(ref_first_token_log_probs, dim=-1, index=first_token_ids)  # (B, 1)
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_completion_logits = model(
                        inputs_embeds=all_inputs_embeds,
                        position_ids=all_position_ids,
                        attention_mask=all_attention_mask,
                        use_cache=False,
                        do_selection=False)[0]
                
                if max_completion_seq_len > 1:
                    ref_completion_logits = ref_completion_logits[:, -max_completion_seq_len:-1]
                    ref_completion_log_probs = torch.log_softmax(ref_completion_logits, dim=-1)  # (B, L, V)
                    ref_completion_logps = torch.gather(ref_completion_log_probs, dim=-1, index=completion_ids_from_second.unsqueeze(-1)).squeeze(-1)  # (B, L)
                    ref_completion_logps = self._extract_valid_completion_logps(ref_completion_logps, valid_completion_lens - 1)
                else:
                    ref_completion_logps = [torch.zeros(0, dtype=first_token_logits.dtype, device=first_token_logits.device) for _ in range(bsz)]
                for b in range(bsz):
                    ref_completion_logps[b] = torch.cat([ref_first_token_logps[b], ref_completion_logps[b]], dim=0)
                
                # for debug
                # ref_first_token_ids = ref_first_token_log_probs.argmax(dim=-1, keepdim=True)  # (B, 1)
                # ref_completion_ids_from_second = ref_completion_logits.argmax(dim=-1)
                # ref_completion_ids_from_second = [torch.cat([first_id, one_ids[-valid_len+1:]], dim=-1) for first_id, one_ids, valid_len in zip(ref_first_token_ids, ref_completion_ids_from_second, valid_completion_lens)]
                # ref_completion_texts = self.processing_class.batch_decode(ref_completion_ids_from_second, skip_special_tokens=True)
        else:
            ref_completion_logps = None
            ref_completion_texts = None
            
        # get lopgs of first token
        # first_token_ids # (B, 1)
        first_token_log_probs = torch.log_softmax(first_token_logits, dim=-1)  # (B, V)
        first_token_logps = torch.gather(first_token_log_probs, dim=-1, index=first_token_ids)  # (B, 1)
        
        # concat logps
        for b in range (bsz):
            completion_logps[b] = torch.cat(
                [first_token_logps[b], completion_logps[b]], dim=0
            )

        # convert scores to advantages
        scores = list(score_dict.values())
        
        scores = torch.tensor(scores, dtype=first_token_log_probs.dtype, device=first_token_log_probs.device).transpose(0, 1)  # (B, N)
        scores_by_sample = self.accelerator.gather(scores)  # GxB, N
        
        mean_scores_by_names = scores_by_sample.mean(dim=0)  # N
        std_scores_by_names = scores_by_sample.std(dim=0)  # N
        for score_name, one_mean_score, one_std_score in zip(score_dict.keys(), mean_scores_by_names, std_scores_by_names):
            self._metrics[f"{score_name}_score_mean"].append(self.accelerator.gather_for_metrics(one_mean_score).mean().item())
            self._metrics[f"{score_name}_score_std"].append(self.accelerator.gather_for_metrics(one_std_score).mean().item())
        
        scores = scores.sum(dim=1)  # (B, )
        mean_scores = scores_by_sample.sum(dim=1).mean()
        std_scores = scores_by_sample.sum(dim=1).std()
        
        advantages = (scores - mean_scores) / (std_scores + 1e-4)

        # self._log_datas(
        #     query_texts, answer_texts, 
        #     completion_texts=completion_texts, 
        #     ref_completion_texts=ref_completion_texts,
        #     scores=scores,
        #     # advantages=advantages,
        #     # mean_scores=mean_scores,
        #     # std_scores=std_scores,
        # )


        output_dict.update({
            # "scores_by_sample": scores_by_sample,  # List(G) of tensors (bsz,)
            "advantages": advantages,  # (G, bsz)
            "completion_logps": completion_logps,  # List(G) of List(bsz) of tensors (L,)
            "ref_completion_logps": ref_completion_logps,  # List(G) of List(bsz) of tensors (L,) or None
        })
        return output_dict
    
    # def _score_completions(self, query_texts, completion_texts, answer_texts):
    #     return self.client.score(query_texts, completion_texts, answer_texts)
    
        
    def _calculate_kd_loss(self, additional_inputs):
        if self.args.kd_weight <= 0:
            return 0
        
        ref_completion_logps = additional_inputs["ref_completion_logps"]  # List(B) of tensors (L,)
        completion_logps = additional_inputs["completion_logps"]  # List(B) of tensors (L,)
        
        B = len(ref_completion_logps)
        mean_kl_loss = 0.0

        for b in range(B):
            ref_per_token_logps = ref_completion_logps[b]  # (L,)
            per_token_lopgs = completion_logps[b]  # (L,)
            per_token_kl = torch.exp(ref_per_token_logps - per_token_lopgs) - (ref_per_token_logps - per_token_lopgs) - 1
            mean_kl_loss += per_token_kl.mean()
        mean_kl_loss /= B
        mean_kl_loss *= self.args.kd_weight
        
        self._metrics["kd_loss"].append(self.accelerator.gather_for_metrics(mean_kl_loss).mean().item())
        return mean_kl_loss
        
    
    def _calculate_reward_loss(self, additional_inputs):
        if self.reward_weight <= 0:
            return 0
        
        if self.num_iterations > 1:
            raise NotImplementedError()
        
        completion_logps = additional_inputs["completion_logps"]  # List(B) of tensors (L,)
        advantages = additional_inputs["advantages"]  # (B,)
        
        B = advantages.shape[0]
        
        mean_reward_loss = 0.0
        for b in range(B):
            per_token_logps = completion_logps[b]  # (L,)
            per_token_loss = -torch.exp(per_token_logps - per_token_logps.detach()) * advantages[b]
            mean_reward_loss += per_token_loss.mean()
        mean_reward_loss /= B
        
        mean_reward_loss *= self.reward_weight
        self._metrics["reward_loss"].append(self.accelerator.gather_for_metrics(mean_reward_loss).mean().item())
        return mean_reward_loss
        
        
    
    @torch.no_grad()
    def _update_loc_conf_mat(self, image_token_mask_logits, ref_token_masks):
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        for one_pred_masks, one_ref_masks in zip(image_token_mask_logits, ref_token_masks):
            pred_mask = one_pred_masks[-1].sigmoid() > 0.5
            ref_mask = one_ref_masks.bool()
            pred_mask = pred_mask.view(-1).int()
            ref_mask = ref_mask.view(-1).int()
            tp += (pred_mask * ref_mask).sum().item()
            fp += (pred_mask * (1 - ref_mask)).sum().item()
            fn += ((1 - pred_mask) * ref_mask).sum().item()
            tn += ((1 - pred_mask) * (1 - ref_mask)).sum().item()
        self._loc_conf_mat_list.append(np.array([[tp, fp], [fn, tn]]))
        
    
    def _calculate_loc_loss(self, inputs, outputs):
        if self.loc_weight <= 0:
            return 0.0
        
        image_token_mask_logits = outputs["image_token_mask_logits"]  # List(B) of float tensors (0-1 value)
        ref_token_masks = inputs["ref_token_masks"]  # List(B) of Bool Tensors (H, W)
        pred_layers = image_token_mask_logits[0].shape[0]
        loc_loss = 0
        for layer_id in range(pred_layers):
            image_token_mask_logits_per_layer = [one_mask[layer_id] for one_mask in image_token_mask_logits]
            loc_loss_per_layer = self.loc_criterion(image_token_mask_logits_per_layer, ref_token_masks)
            loc_loss += loc_loss_per_layer
            self._metrics[f"loc_loss_{layer_id}"].append(self.accelerator.gather_for_metrics(loc_loss_per_layer).mean().item())
        self._update_loc_conf_mat(image_token_mask_logits, ref_token_masks)
        return loc_loss * self.loc_weight
        
            
    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        # gather box_conf_mat
        # local_rank = int(os.environ.get("LOCAL_RANK", 0))
        # print(f"Rank {local_rank}: box_conf_mat: {self._box_conf_mat}")
        box_conf_mat_list = self.accelerator.gather_for_metrics(self._loc_conf_mat_list)
        if len(box_conf_mat_list) > 0:
            box_conf_mat = np.sum(np.stack(box_conf_mat_list), axis=0)
            tp = box_conf_mat[0, 0].item()
            fp = box_conf_mat[0, 1].item()
            fn = box_conf_mat[1, 0].item()
            tn = box_conf_mat[1, 1].item()
            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
            iou = tp / (tp + fp + fn) if tp + fp + fn > 0 else 0
            box_metrics = {
                "box/precision": precision,
                "box/recall": recall,
                "box/f1": f1,
                "box/iou": iou
            }
        else:
            box_metrics = {
            }
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics, **box_metrics}
        super().log(logs, start_time)
        self._metrics.clear()
        self._loc_conf_mat_list.clear()
        
    @debug_calls(only_rank0=True)
    def _log_cuda_memeory(self):
        mem = torch.cuda.memory_allocated()
        print(f"CUDA memory allocated: {mem / (1024 ** 2):.2f} MB")
        
    
    @debug_calls(only_rank0=False)
    def _log_datas(self, query_texts, answer_texts, **kwargs):
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        forward_times = self._step
        backward_times = self.state.global_step
        info = f"\nB({backward_times} F({forward_times}) R[{local_rank}]): {answer_texts}"
        for key, value in kwargs.items():
            info += f"\n{key}: {value}"
        print(info)
            
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        
        
        self.accelerator.unwrap_model(model).reset_image_tokens_cache()
        # 1. Get original completion logits without do_selection (no_grad)
        if self.state.global_step % self.num_iterations == 0:
            additional_inputs = self._generate_and_score_completions(model, inputs)
            # self._buffered_inputs[self._step % self.args.gradient_accumulation_steps] = additional_inputs
        else:
            raise NotImplementedError("Multiple iterations are not supported yet.")
            # additional_inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
        self._step += 1
        
        loc_loss = self._calculate_loc_loss(inputs, additional_inputs)
        kd_loss = self._calculate_kd_loss(additional_inputs)
        reward_loss = self._calculate_reward_loss(additional_inputs)
        le_loss = additional_inputs.get("le_loss", None)
        if le_loss is None:
            le_loss = 0
        else:
            le_loss = self.le_weight * le_loss
            self._metrics["le_loss"].append(self.accelerator.gather_for_metrics(le_loss).mean().item())
        return kd_loss + loc_loss + reward_loss + le_loss


    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # super()._save(output_dir=output_dir, state_dict=state_dict)
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        # self.model.save_new_modules(output_dir)
        
        model_to_save = self.accelerator.unwrap_model(self.model)
        if self.loc_weight > 0 or self.le_weight > 0:
            model_to_save.save_new_modules(output_dir)
        
        if self.reward_weight > 0:
            model_to_save.config.save_pretrained(output_dir)
            
            supported_classes = (PreTrainedModel,) if not is_peft_available() else (PreTrainedModel, PeftModel)
            # Save a trained model and configuration using `save_pretrained()`.
            # They can then be reloaded using `from_pretrained()`
            if not isinstance(self.model, supported_classes):
                if state_dict is None:
                    state_dict = self.model.state_dict()

                if isinstance(self.accelerator.unwrap_model(self.model), supported_classes):
                    self.accelerator.unwrap_model(self.model).save_pretrained(
                        output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                    )
                else:
                    logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                    if self.args.save_safetensors:
                        safetensors.torch.save_file(
                            state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME), metadata={"format": "pt"}
                        )
                    else:
                        torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
            else:
                self.model.save_pretrained(
                    output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                )
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))


    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        if self.reward_weight > 0:
            # call super
            super()._load_from_checkpoint(resume_from_checkpoint, model=model)
            
        if model is None:
            model = self.model
        model.load_new_modules(resume_from_checkpoint)
        
    def _get_train_sampler(self) -> torch.utils.data.Sampler:
        effective_batch_size = (
            self.args.per_device_train_batch_size
            * self.accelerator.num_processes
            * self.args.gradient_accumulation_steps
        )
        
        return RepeatRandomSampler(
            data_source=self.train_dataset,
            mini_repeat_count=self.num_generations,
            batch_size=effective_batch_size // self.num_generations,
            repeat_count=self.num_iterations,
            seed=self.args.seed,
        )

# ---------- Arguments ----------

@dataclass
class GPScriptArguments:
    train_dataset: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the training dataset config."},
    )
    eval_dataset: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the evaluation dataset config."},
    )
    img_dir: str = field(
        default="datas",
        metadata={"help": "Path to the image directory."},
    )    
    sampling_seed: int = field(
        default=42,
        metadata={"help": "Random seed for sampling."},
    )
    max_pixels: int = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image."},
    )
    max_input_seq_length: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum sequence length for the input."},
    )
    max_input_remain_seq_length: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum remaining sequence length for the input."},
    )


@dataclass
class GPTrainingArguments(TrainingArguments):
    kd_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for knowledge distillation loss."},
    )
    
    loc_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for loc loss."},
    )
    
    loc_loss_class: str = field(
        default="DiceLoss",
        metadata={"help": "Class for loc loss."},
    )
    
    loc_dice_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for Dice loss in MaskLoss."},
    )
    
    loc_bce_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for BCE loss in MaskLoss."},
    )
    
    le_weight: float = field(
        default=0.0,
        metadata={"help": "Weight for learnable embeddings sft loss."},
    )
    
    reward_weight: float = field(
        default=0.0,
        metadata={"help": "Weight for reward loss."},
    )
    
    num_generations: int = field(
        default=4,
        metadata={"help": "Number of generations of RL sampling for each batch."},
    )
    
    num_iterations: int = field(
        default=1,
        metadata={"help": "Number of iterations of RL sampling."},
    )
    
    max_completion_length: int = field(
        default=128,
        metadata={"help": "Maximum length of the completion of generation."},
    )
    
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum sequence length when training the model."},
    )
    
    gen_mask_usage_ratio: float = field(
        default=0.0,
        metadata={"help": "Ratio of image token mask usage in generation."},
    )
        
    min_ratio: float = field(
        default=0.1,
        metadata={"help": "Minimum ratio for image token mask."},
    )
    
    load_new_modules: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the directory to load new modules from."},
    )

    load_adapter: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the adapter to load."},
    )
    
    client_base_url: str = field(
        default="http://localhost:8000/v1",
        metadata={"help": "Base URL for the client."},
    )
    
    client_api_key: str = field (
        default="dummy",
        metadata={"help": "API Key for the client."},
    )
    
    client_model_name: str = field(
        default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8",
        metadata={"help": "Model name for the client."},
    )


@dataclass
class GPModelConfig:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Model checkpoint for weights initialization."},
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override the default `torch.dtype` and load the model under this dtype.",
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    attn_implementation: Optional[str] = field(
        default=None,
        metadata={
            "help": "Which attention implementation to use. You can run `--attn_implementation=flash_attention_2`, in "
            "which case you must install this manually by running `pip install flash-attn --no-build-isolation`."
        },
    )
    selected_layers: Tuple[int] = field(
        default=(),
        metadata={"help": "Selected layers for the LLM Decoder."},
    )
    reduce_layer: int = field(
        default=1000,
        metadata={"help": "Layer to do reduce during prefilling."},
    )
    anchor_positions: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Anchor positions for reduction. If None, use the default behavior."}
    )
    use_attention_logits: bool = field(
        default=False,
        metadata={"help": "Whether to use attention logits or logps."},
    )
    attn_fuse_type: str = field(
        default="AttnFuserWinAttnV1",
        metadata={"help": "Attention fusion type."},
    )
    attn_fuse_size: int = field(
        default=256,
        metadata={"help": "Attention fusion size."},
    )
    attn_fuse_global: bool = field(
        default=False,
        metadata={"help": "Whether to use global attention fusion."},
    )
    selected_visual_layers: Tuple[int] = field(
        default=(),
        metadata={"help": "Selected layers for the vision encoder."},
    )
    visual_cond_size: int = field(
        default=256,
        metadata={"help": "Visual condition size."},
    )
    attn_fuse_num_heads: int = field(
        default=4,
        metadata={"help": "Number of attention heads for the fusion."},
    )
    attn_fuse_hidden_act: str = field(
        default="silu",
        metadata={"help": "Activation function for the fusion."},
    )
    ori_attn_supervision: bool = field(
        default=True,
        metadata={"help": "Whether to use original attention supervision."},
    )
    deep_supervision: bool = field(
        default=True,
        metadata={"help": "Whether to use deep supervision of loc loss."},
    )
    le_layers: Tuple[int] = field(
        default=(0,),
        metadata={"help": "Layers for the learnable embeddings."},
    )
    le_length: int = field(
        default=1,
        metadata={"help": "Length of the learnable embeddings."},
    )
    le_norm_type: str = field(
        default="rmsnorm",
        metadata={"help": "Normalization type for the learnable embeddings."},
    )
    le_dropout_prob: float = field(
        default=0.1,
        metadata={"help": "Dropout probability for the learnable embeddings."},
    )
    
    use_peft: bool = field(
        default=False,
        metadata={"help": "Whether to use PEFT for training."},
    )
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA R value."},
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha."},
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout."},
    )
    lora_target_modules: Optional[list[str]] = field(
        default=None,
        metadata={"help": "LoRA target modules."},
    )
    lora_modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={"help": "Model layers to unfreeze & train."},
    )
    lora_task_type: str = field(
        default="CAUSAL_LM",
        metadata={"help": "Task type to pass for LoRA (use 'SEQ_CLS' for reward modeling)."},
    )
    use_rslora: bool = field(
        default=False,
        metadata={
            "help": "Whether to use Rank-Stabilized LoRA, which sets the adapter scaling factor to `lora_alpha/√r`, "
            "instead of the original default value of `lora_alpha/r`."
        },
    )
    use_dora: bool = field(
        default=False,
        metadata={
            "help": "Enable Weight-Decomposed Low-Rank Adaptation (DoRA). This technique decomposes the updates of "
            "the weights into two parts, magnitude and direction. Direction is handled by normal LoRA, whereas the "
            "magnitude is handled by a separate learnable parameter. This can improve the performance of LoRA, "
            "especially at low ranks. Right now, DoRA only supports linear and Conv2D layers. DoRA introduces a "
            "bigger overhead than pure LoRA, so it is recommended to merge weights for inference."
        },
    )

# ---------- Main ----------

def patch_processor(processor):
    if not hasattr(processor.tokenizer, "eos_token_id"):
        eos_token = getattr(processor.tokenizer, "eos_token")
        eos_token_id = processor.tokenizer.convert_tokens_to_ids(eos_token)
        processor.tokenizer.eos_token_id = eos_token_id
    print("eos_token_id:", processor.tokenizer.eos_token_id)
    
    if not hasattr(processor.tokenizer, "pad_token_id"):
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    print("pad_token_id:", processor.tokenizer.pad_token_id)
    processor.tokenizer.padding_side = "left"

def main():
    parser = TrlParser((GPScriptArguments, GPTrainingArguments, GPModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    
    
    processor = Qwen2_5_VL_GP_Processor.from_pretrained(
        model_args.model_name_or_path,
        max_pixels=script_args.max_pixels,
    )
    patch_processor(processor)
    
    train_dataset = GPDataset(script_args.train_dataset, processor, script_args) if script_args.train_dataset else None
    
    data_collator = GPCollator(processor, is_sft=training_args.le_weight > 0)
    
    model_class = Qwen2_5_VL_GP_ForConditionalGeneration
    
    model_args_dict = vars(model_args)
    
    model_init_kwargs = {}
    for key in model_args_dict:
        if 'peft' in key or 'lora' in key or 'dora' in key:
            continue
        model_init_kwargs[key] = model_args_dict[key]
    
    model_name_or_path = model_init_kwargs.pop("model_name_or_path")
    
    set_seed(training_args.seed)
    
    model = model_class.from_pretrained(
        model_name_or_path,
        **model_init_kwargs)
    
    if training_args.load_adapter:
        model = PeftModel.from_pretrained(model, training_args.load_adapter)
        model = model.merge_and_unload()
        print("========= The adapter is merged. =========")
    if training_args.load_new_modules:
        model.load_new_modules(training_args.load_new_modules)
    
    
    for param in model.parameters():
        param.requires_grad = False
        
    if training_args.loc_weight > 0 or training_args.le_weight > 0:
        for module in model.new_modules_to_be_saved().values():
            if isinstance(module, nn.Parameter):
                module.requires_grad = True
            else:
                for param in module.parameters():
                    param.requires_grad = True
    
    peft_config = get_peft_config(model_args)
    if peft_config is not None:
        if hasattr(model, "peft_target_modules"):
            module_names = model.peft_target_modules().keys()
            peft_config.target_modules = list(module_names)
        model = get_peft_model(model, peft_config)
    
    trainer = GPTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        processing_class=processor.tokenizer,
    )
            
    train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    
    
    
if __name__ == '__main__':
    main()
    
    
    
    
    
