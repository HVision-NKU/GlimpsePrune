import os
import yaml
import warnings
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Any, Callable, Optional, Union, Sized, Dict, Tuple, List, Literal, Type

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import datasets

from PIL import Image

from utils_llava import MyTrlParser as TrlParser

from transformers import (
    TrainingArguments, 
    Trainer,
)
from transformers.utils import (
    is_safetensors_available, 
)


from accelerate.utils import set_seed


from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava_gp.mm_utils import (
    get_model_name_from_path,
    process_images,
    process_bboxes,
    tokenizer_image_token,
)
from llava_gp.model.builder import load_pretrained_model


from transformers.trainer import (
    logger,
    TRAINING_ARGS_NAME,
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
# SCORE_FUNCS_KEY = "score_funcs"


REMAIN_KEYS = [
    QUERY_KEY,
    IMG_PATH_KEY,
    NORMED_BBOXES_KEY,
    ANSWER_KEY,
    # SCORE_FUNCS_KEY,
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
        # SCORE_FUNCS_KEY: kwargs['score_funcs']
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
        # SCORE_FUNCS_KEY: kwargs['score_funcs']
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
    tokenizer = kwargs['tokenizer']
    max_input_seq_length = kwargs.get('max_input_seq_length', None)
    max_input_remain_seq_length = kwargs.get('max_input_remain_seq_length', None)
    if max_input_seq_length is None and max_input_remain_seq_length is None:
        return True
    img_path = one_data[IMG_PATH_KEY]
    query = one_data[QUERY_KEY]
    normed_bboxes = [one_data[NORMED_BBOXES_KEY]] if max_input_remain_seq_length is not None else None
    messages = [[{"role": "user", "content": [{"type": "image", "image": img_path}, {"type": "text", "text": query}]}]]
    
    raise NotImplementedError()
    
    seq_length = inputs.input_ids.shape[1]
    if max_input_seq_length is not None and seq_length > max_input_seq_length:
        print_rank0(f"Input sequence length {seq_length} exceeds max limit {max_input_seq_length}. Filtering out.")
        return False
    
    if max_input_remain_seq_length is not None:
        ref_token_masks = inputs.ref_token_masks[0]
        reduced_num = ref_token_masks.numel() - ref_token_masks.sum().item()
        remain_seq_length = seq_length - reduced_num
        if remain_seq_length > max_input_remain_seq_length:
            print_rank0(f"Remaining sequence length {remain_seq_length} exceeds max limit {max_input_remain_seq_length}. Filtering out.")
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
                image_token_masks: List[torch.Tensor], 
                ref_token_masks: List[torch.Tensor]
               ) -> torch.Tensor:
        if not isinstance(image_token_masks, list) or not isinstance(ref_token_masks, list):
            raise TypeError("Inputs must be lists of tensors.")
            
        if len(image_token_masks) != len(ref_token_masks):
            raise ValueError(f"Input lists must have the same length, but got "
                             f"{len(image_token_masks)} and {len(ref_token_masks)}")
            
        if len(image_token_masks) == 0:
            # Handle empty batch case if necessary, e.g., return 0 loss or raise error
            return torch.tensor(0.0, device=image_token_masks[0].device if image_token_masks else None) 
            # Or raise ValueError("Input lists cannot be empty") depending on desired behavior

        batch_size = len(image_token_masks)
        total_dice_loss = 0.0

        for i in range(batch_size):
            pred_mask_1d = image_token_masks[i].flatten().sigmoid() # Shape: (N_b,) float
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
                image_token_masks: List[torch.Tensor], 
                ref_token_masks: List[torch.Tensor]
               ) -> torch.Tensor:
        
        batch_size = len(image_token_masks)
        total_bce_loss = 0.0
        for i in range(batch_size):
            pred_mask_1d = image_token_masks[i].flatten()
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
    
    def forward(self, image_token_masks: List[torch.Tensor],
                ref_token_masks: List[torch.Tensor]
               ) -> torch.Tensor:
        """
        Combines Dice Loss and BCE Loss for image token masks.
        
        Args:
            image_token_masks (List[torch.Tensor]): List of predicted masks (1D tensors).
            ref_token_masks (List[torch.Tensor]): List of ground truth masks (2D tensors).
        
        Returns:
            torch.Tensor: Combined loss value.
        """
        dice_loss = self.dice_loss(image_token_masks, ref_token_masks)
        bce_loss = self.bce_loss(image_token_masks, ref_token_masks)
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss




# ---------- Dataset & Collator & Sampler ----------

class LlavaGPDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset that loads and combines multiple datasets
    based on a YAML configuration file. It handles sampling
    and applies specified mapping functions.
    """
    
    def __init__(self, config_path: str, tokenizer, grid_hw: Tuple[int, int], script_args: Optional[Any] = None):
        """
        Initializes the LlavaGPDataset.

        Args:
            config_path (str): Path to the YAML configuration file.
            tokenizer (Any): Tokenizer instance used for processing text.
            script_args (Any, optional): Additional arguments passed from the script
                                         (e.g., training args, could contain seed). Defaults to None.
        """
        super().__init__()
        self.args = script_args
        self.config = self._load_config(config_path)
        self.tokenizer = tokenizer
        self.grid_hw = grid_hw
        all_processed_datasets = self._all_processed_datasets(self.config, self.tokenizer, self.grid_hw, self.args)
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
    def _all_processed_datasets(cls, config, tokenizer, grid_hw, args):
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
            # score_funcs = dataset_config.get('score_funcs', [])
            prompt = dataset_config.get('prompt', None)
            max_input_seq_length = dataset_config.get('max_input_seq_length', args.max_input_seq_length)
            max_input_remain_seq_length = dataset_config.get('max_input_remain_seq_length', args.max_input_remain_seq_length)
            
            # for score_func in score_funcs:
            #     assert score_func in SCORE_REGISTRY, f"Score function '{score_func}' not registered. Available: {list(SCORE_REGISTRY.keys())}"
            
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
                    # 'score_funcs': score_funcs,
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
                            'tokenizer': tokenizer,
                            'grid_hw': grid_hw,
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
                all_processed_datasets[dataset_name] = processed_dataset                

            except FileNotFoundError:
                print_rank0(f"Error: Data file not found for dataset entry {i+1}: {json_path}. Skipping.")
            except Exception as e:
                print_rank0(f"Error processing dataset entry {i+1} ({json_path}): {e}. Skipping.")
                
        return all_processed_datasets
        

    
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
    def get_processed_dataset_dict(cls, config_path: str, tokenizer, grid_hw: Tuple[int, int], script_args: Optional[Any] = None) -> Dict[str, datasets.Dataset]:
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
        all_processed_datasets = cls._all_processed_datasets(config, tokenizer, grid_hw, script_args) 
        return all_processed_datasets


class LlavaGPCollator:
    def __init__(self, conv_mode, tokenizer, image_processor, model_config, grid_hw, torch_dtype, is_sft):
        self.conv_mode = conv_mode
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.grid_hw = grid_hw
        self.torch_dtype = torch_dtype
        self.is_sft = is_sft

    def _prepare_label(self, input_id: torch.Tensor, input_id_wo_answer: torch.Tensor):
        label = input_id.clone()
        label[:input_id_wo_answer.shape[0]] = IMAGE_TOKEN_INDEX
        return label
    
    def __call__(self, features):
        answers = []
        querys = []
        ref_token_masks = []
        images = []
        image_sizes = []
        input_ids = []
        labels = [] if self.is_sft else None
        grid_h, grid_w = self.grid_hw
        for feature in features:
            query = feature[QUERY_KEY]
            answer = feature[ANSWER_KEY]
            img_path = feature[IMG_PATH_KEY]
            bboxes = feature[NORMED_BBOXES_KEY]
            querys.append(query)
            answers.append(answer)
            
            if self.model_config.mm_use_im_start_end:
                query_with_image = f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}\n{query}"
            else:
                query_with_image = f"{DEFAULT_IMAGE_TOKEN}\n{query}"
            
            
            conv_wo_answer = conv_templates[self.conv_mode].copy()
            conv_wo_answer.append_message(conv_wo_answer.roles[0], query_with_image)
            conv_wo_answer.append_message(conv_wo_answer.roles[1], None)
            prompt_wo_answer = conv_wo_answer.get_prompt()
            if self.is_sft:
                conv = conv_templates[self.conv_mode].copy()
                conv.append_message(conv.roles[0], query_with_image)
                conv.append_message(conv.roles[1], answer)
                prompt = conv.get_prompt()
            else:
                prompt = prompt_wo_answer
                
            image = Image.open(img_path).convert("RGB")
            image_tensor = process_images([image], self.image_processor, self.model_config)[0]
            images.append(image_tensor)
            image_sizes.append(image.size)
            ref_token_masks.append(process_bboxes(bboxes, (grid_h, grid_w)))
            input_id = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
            input_ids.append(input_id)
            
            if self.is_sft:
                input_id_wo_answer = tokenizer_image_token(prompt_wo_answer, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
                label = self._prepare_label(input_id, input_id_wo_answer)
                labels.append(label)
            
        # left padding
        bsz = len(input_ids)
        max_len = max(input_id.shape[0] for input_id in input_ids)
        attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
        position_ids = torch.ones((bsz, max_len), dtype=torch.long)
        for i, input_id in enumerate(input_ids):
            input_len = input_id.shape[0]
            attention_mask[i, -input_len:] = 1
            position_ids[i, -input_len:] = torch.arange(input_len, dtype=torch.long)
        
        
        padded_input_ids = torch.zeros((bsz, max_len), dtype=torch.long)
        padded_input_ids[attention_mask.bool()] = torch.cat(input_ids, dim=0)

        if labels is not None:
            padded_labels = torch.full((bsz, max_len), IMAGE_TOKEN_INDEX ,dtype=torch.long)
            padded_labels[attention_mask.bool()] = torch.cat(labels, dim=0)


        inputs = {
            "input_ids": padded_input_ids,
            "images": torch.stack(images, dim=0).to(dtype=self.torch_dtype),
            "image_sizes": image_sizes,
            "ref_token_masks": ref_token_masks,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        
        if self.is_sft:
            inputs["labels"] = padded_labels
        
        inputs[QUERY_KEY] = querys
        inputs[ANSWER_KEY] = answers
        # inputs[SCORE_FUNCS_KEY] = score_funcs
        
        return inputs
            
# ---------- Trainer ----------


class LlavaGPTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self._loc_conf_mat_list = []
        self._metrics = defaultdict(list)
        # print(f"model init device: {kwargs['model'].device}")
        super().__init__(*args, **kwargs)
        # print(f"model after init device: {self.model.device}")
        
        self.loc_weight = self.args.loc_weight
        self.le_weight = self.args.le_weight
        
        if self.loc_weight > 0:
            self.loc_criterion = LOSS_REGISTRY[self.args.loc_loss_class](dice_weight=self.args.loc_dice_weight, bce_weight=self.args.loc_bce_weight)
        self._step = 0  # forward times
        set_seed(self.args.seed, device_specific=True)
        
    
    @torch.no_grad()
    def _update_loc_conf_mat(self, image_token_masks, ref_token_masks):
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        for one_pred_masks, one_ref_masks in zip(image_token_masks, ref_token_masks):
            pred_mask = one_pred_masks[-1].sigmoid() > 0.5
            ref_mask = one_ref_masks.bool()
            pred_mask = pred_mask.view(-1).int()
            ref_mask = ref_mask.view(-1).int()
            tp += (pred_mask * ref_mask).sum().item()
            fp += (pred_mask * (1 - ref_mask)).sum().item()
            fn += ((1 - pred_mask) * ref_mask).sum().item()
            tn += ((1 - pred_mask) * (1 - ref_mask)).sum().item()
        self._loc_conf_mat_list.append(torch.tensor([[tp, fp], [fn, tn]], device=self.accelerator.device, dtype=torch.int64))


    def _calculate_loc_loss(self, inputs, outputs):
        if self.loc_weight <= 0:
            return 0.0
        
        image_token_mask_logits = outputs["image_token_mask_logits"]  # List(B) of float tensors (0-1 value)
        ref_token_masks = inputs["ref_token_masks"]  # List(B) of Bool Tensors (H, W)
        pred_layers = image_token_mask_logits[0].shape[0]
        loc_loss = 0
        for layer_id in range(pred_layers):
            image_token_masks_per_layer = [one_mask[layer_id] for one_mask in image_token_mask_logits]
            loc_loss_per_layer = self.loc_criterion(image_token_masks_per_layer, ref_token_masks)
            loc_loss += loc_loss_per_layer
            self._metrics[f"loc_loss_{layer_id}"].append(self.accelerator.gather_for_metrics(loc_loss_per_layer).mean().item())
        self._update_loc_conf_mat(image_token_mask_logits, ref_token_masks)
        return loc_loss * self.loc_weight
        
    
            
    def log(self, logs: dict[str, float]) -> None:
        # gather box_conf_mat
        # local_rank = int(os.environ.get("LOCAL_RANK", 0))
        # print(f"Rank {local_rank}: box_conf_mat: {self._box_conf_mat}")
        box_conf_mat_list = self.accelerator.gather_for_metrics(self._loc_conf_mat_list)
        if len(box_conf_mat_list) > 0:
            box_conf_mat = torch.sum(torch.stack(box_conf_mat_list, dim=0), dim=0)
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
        super().log(logs)
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
        query = inputs.pop(QUERY_KEY)
        answer = inputs.pop(ANSWER_KEY)
        
        self.accelerator.unwrap_model(model).reset_image_tokens_cache()
        outputs = model(
            **inputs,
            do_selection=True,
            use_cache=False,
            return_dict=True,
            delay_selection=True,
        )
        le_loss = outputs.le_loss

        loc_loss = self._calculate_loc_loss(inputs, outputs)
        le_loss = outputs.get("le_loss", None)
        if le_loss is None:
            le_loss = 0
        else:
            le_loss = self.le_weight * le_loss
            self._metrics["le_loss"].append(self.accelerator.gather_for_metrics(le_loss).mean().item())
        return loc_loss + le_loss


    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # super()._save(output_dir=output_dir, state_dict=state_dict)
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        model_to_save = self.accelerator.unwrap_model(self.model)
        model_to_save.save_new_modules(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))


    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        if model is None:
            model = self.model
        model.load_new_modules(resume_from_checkpoint)
        

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
    loc_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for box loss."},
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
    
    
    load_new_modules: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the directory to load new modules from."},
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
        metadata={"help": "Whether to use deep supervision of box loss."},
    )
    le_layers: Tuple[int] = field(
        default=(0,),
        metadata={"help": "Layers for the learnable embeddings."},
    )
    le_length: int = field(
        default=1,
        metadata={"help": "Length of the learnable embeddings."},
    )
    le_dropout_prob: float = field(
        default=0.1,
        metadata={"help": "Dropout probability for the learnable embeddings."},
    )
    

# ---------- Main ----------


def main():
    parser = TrlParser((GPScriptArguments, GPTrainingArguments, GPModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    
    set_seed(training_args.seed)
    
    model_args_dict = vars(model_args)
    
    model_init_kwargs = {}
    for key in model_args_dict:
        if 'peft' in key or 'lora' in key or 'dora' in key:
            continue
        model_init_kwargs[key] = model_args_dict[key]
    
    base_model = model_init_kwargs.pop("model_name_or_path")

    model_name = get_model_name_from_path(base_model)

    # device = f"cuda:{local_rank}"
    print(f"Local rank: {training_args.local_rank}, Device: {training_args.device}")
    model_init_kwargs["device_map"] = {"": training_args.device}
    model_init_kwargs["device"] = training_args.device
    torch_dtype = getattr(torch, model_args.torch_dtype)
    model_init_kwargs["torch_dtype"] = torch_dtype

    tokenizer, model, image_processor, max_length = load_pretrained_model(
        base_model,
        None,
        model_name,
        **model_init_kwargs,
    )
    
    # vision_tower = model.get_vision_tower()
    # vision_tower.to(device=training_args.device, dtype=torch_dtype)
    # print(f"Local rank: {training_args.local_rank}, Vision tower device: {vision_tower.device}, dtype: {vision_tower.dtype}")
    
    model.config.tokenizer_padding_side = "left"
    tokenizer.padding_side = "left"
    
    grid_h = grid_w = model.get_vision_tower().num_patches_per_side
    grid_hw = (grid_h, grid_w)
    
    train_dataset = LlavaGPDataset(script_args.train_dataset, tokenizer, grid_hw, script_args)
                                   
    is_sft = training_args.le_weight > 0
    conv_mode = "vicuna_v1"
    data_collator = LlavaGPCollator(conv_mode, tokenizer, image_processor, model.config, 
                                    grid_hw, torch_dtype, is_sft=is_sft)


    if training_args.load_new_modules:
        model.load_new_modules(training_args.load_new_modules)
    
    
    for param in model.parameters():
        param.requires_grad = False
        
    for module in model.new_modules_to_be_saved().values():
        if isinstance(module, nn.Parameter):
            module.requires_grad = True
        else:
            for param in module.parameters():
                param.requires_grad = True
    
    
    trainer = LlavaGPTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )
            
    train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    
    
    
if __name__ == '__main__':
    main()
    
    
    
    
    
