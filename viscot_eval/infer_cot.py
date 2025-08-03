import os
import json
import warnings
import ast
from collections import defaultdict
from typing import List, Dict, Any, Optional, Literal, Union
from tqdm import tqdm
from dataclasses import dataclass, field
import datasets
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import HfArgumentParser, GenerationConfig
from PIL import Image
from accelerate.utils import set_seed

from warppers import (
    time_logger_set_active,
    memory_logger_set_active,
    get_all_time_logger_stats,
    reset_all_time_logger_stats,
    get_all_memory_logger_stats,
    reset_all_memory_logger_stats
)

from .models import BaseInferModel, get_model

COT_BRIEF_PROMPT = "{}\n\nAnswer the question using a single word or phrase."
CHOICE_BRIEF_PROMPT = "\nAnswer with the option's letter from the given choices directly."


def setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank) 
    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    return local_rank, world_size, rank


@dataclass
class ScriptArgs:
    model_type: str = field(
        default="qwen2_5_vl_gp",
        metadata={"help": "Version of the new modules."}
    )
    base_model: str = field(
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        metadata={"help": "Path to the base model."}
    )
    attn_implementation: Literal["flash_attention_2", "eager", "sdpa"] = field(
        default="flash_attention_2",
        metadata={"help": "Attention implementation to use."}
    )
    torch_dtype: Literal["auto", "bfloat16", "float16", "float32"] = field(
        default="bfloat16",
        metadata={"help": "Torch dtype to use for the model."}
    )
    use_cache: bool = field(
        default=True,
        metadata={"help": "Whether to use cache for the model."}
    )
    seed: bool = field(
        default=42,
        metadata={"help": "Random seed for reproducibility."}
    )
    enable_time_logger: bool = field(
        default=False,
        metadata={"help": "Whether to enable time logger."}
    )
    enable_memory_logger: bool = field(
        default=False,
        metadata={"help": "Whether to enable memory logger."}
    )
    new_modules_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the new modules directory."}
    )
    new_modules_config: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the new modules config file."}
    )
    adapter_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the adapter directory."}
    )
    adapter_merge: bool = field(
        default=False,
        metadata={"help": "Whether to merge the adapter."}
    )
    do_selection: bool = field(
        default=True,
        metadata={"help": "Whether to do selection."}
    )
    datasets: str = field(
        default=("gqa,flickr30k,vsr,visual7w,cub,dude,infographicsvqa,openimages,sroie,textcap,textvqa,docvqa"),
        metadata={"help": "List of dataset names to use."}
    )
    num_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Number of samples to use from each dataset. If None, use all samples."}
    )
    img_dir: str = field(
        default="datas",
        metadata={"help": "Path to the image directory."}
    )
    cot_bench_dir: str = field(
        default="viscot_benchmark/benchmark",
        metadata={"help": "Path to the cot bench directory."}
    )
    batch_size_per_device: int = field(
        default=1,
        metadata={"help": "Batch size per device."}
    )
    max_new_tokens: int = field(
        default=1024,
        metadata={"help": "Maximum number of new tokens to generate."}
    )
    output_dir: str = field(
        default="result/cot_bench/debug",
        metadata={"help": "Path to the output directory."}
    )
    use_box: bool = field(
        default=False,
        metadata={"help": "Whether to use bounding box."}
    )
    use_ref_masks: bool = field(
        default=False,
        metadata={"help": "Whether to use reference masks."}
    )
    use_zero_masks: bool = field(
        default=False,
        metadata={"help": "Whether to use no masks."}
    )
    reduce_layer: Optional[int] = field(
        default=None,
        metadata={"help": "After this layer in LLM when decoding, the visual tokens are reduced."}
    )
    anchor_positions: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Anchor positions for reduction. If None, use the default behavior."}
    )
    min_remain_num: Optional[int] = field(
        default=None,
        metadata={"help": "Minimum number of visual tokens to remain after reduction. If None, use the default behavior."}
    )
    max_remain_ratio: Optional[float] = field(
        default=None,
        metadata={"help": "Maximum ratio of visual tokens to remain after reduction. If None, use the default behavior."}
    )
    do_func_name: Literal["generate", "glimpse"] = field(
        default="generate",
        metadata={"help": "Function to use for processing."}
    )
    brief: bool = field(
        default=False,
        metadata={"help": "Whether to use brief prompt."}
    )
    save_masks: bool = field(
        default=False,
        metadata={"help": "Whether to save masks."}
    )
    
    layer_list: Optional[str] = field(
        default=None,
        metadata={"help": "List of layers to use for VScan/PDrop model."}
    )
    image_token_ratio_list: Optional[str] = field(
        default=None,
        metadata={"help": "List of image token ratios for VScan/PDrop model."}
    )
    image_token_list: Optional[str] = field(
        default=None,
        metadata={"help": "List of image tokens for VScan/CDPruner model."}
    )
    image_token_ratio: Optional[float] = field(
        default=None,
        metadata={"help": "Image token ratio for VScan model."}
    )
    visual_token_num: Optional[int] = field(
        default=None,
        metadata={"help": "Number of visual tokens for VScan/CDPruner model."}
    )
    dominant_ratio: Optional[float] = field(
        default=None,
        metadata={"help": "Dominant ratio for VisionZip model."}
    )
    contextual_ratio: Optional[float] = field(
        default=None,
        metadata={"help": "Contextual ratio for VisionZip model."}
    )
    dominant: Optional[int] = field(
        default=None,
        metadata={"help": "Dominant visual token number for VisionZip model."}
    )
    contextual: Optional[int] = field(
        default=None,
        metadata={"help": "Contextual visual token number for VisionZip model."}
    )
    
        
def norm_bbox(bbox, width, height):
    return [bbox[0]/width, bbox[1]/height, bbox[2]/width, bbox[3]/height]


QUERY_KEY = "query"
IMG_PATH_KEY = "img_path"

BOX_KEY = "normed_bboxes"
RESPONSE_KEY = "response"
CONF_MAT_KEY = "conf_mat"
IOU_KEY = "iou"
RATIO_KEY = "ratio"
NUM_GEN_TOKENS_KEY = "num_gen_tokens"

MIDDLE_KEYS = [
    QUERY_KEY,
    IMG_PATH_KEY,
    BOX_KEY,
]

def cot_bench_dataset_mapper(one_data, args):
    query = one_data["conversations"][0]["value"].replace("Please provide the bounding box coordinate of the region that can help you answer the question better.", "").strip()
    query = query.replace("<image>\n", "")
    if args.brief:
        query = COT_BRIEF_PROMPT.format(query)
    img_path = os.path.join(args.img_dir, one_data["image"][0])
    # messages = [{"role": "user", "content": [{"type": "image", "image": img_path}, {"type": "text", "text": query}]}]
    # one_data[MESSAGE_KEY] = messages
    one_data[QUERY_KEY] = query
    if not os.path.isfile(img_path):
        img_path_list = list(img_path.split('/'))
        img_path_list.insert(3, "val")
        img_path = os.path.join(*img_path_list)
    assert os.path.isfile(img_path), f"Image file {img_path} does not exist."
    one_data[IMG_PATH_KEY] = img_path
    if args.use_box:
        bbox = one_data["image"][1].split('###')[1]
        bbox = ast.literal_eval(bbox)
        img = Image.open(img_path)
        width, height = img.size
        one_data[BOX_KEY] = [norm_bbox(bbox, width, height)]
    return one_data

def vstar_bench_dataset_mapper(one_data, args):
    query = one_data["text"]
    if not args.brief:
        query = query.replace(CHOICE_BRIEF_PROMPT, "")
    img_path = os.path.join(args.img_dir, one_data["image"])
    one_data[QUERY_KEY] = query
    one_data[IMG_PATH_KEY] = img_path
    if args.use_box:
        pass
        # raise NotImplementedError("use_box not implemented for vstar")
    return one_data


def refcoco_dataset_mapper(one_data, args):
    one_data[QUERY_KEY] = one_data["problem"] + " Output the final answer in JSON format."
    one_data[IMG_PATH_KEY] = os.path.join(args.img_dir, one_data["image"])
    assert not args.brief
    if args.use_box:
        box = one_data["normalized_solution"]
        box = [d / 1000.0 for d in box]  # convert from millimeters to meters
        one_data[BOX_KEY] = [box]
    return one_data


def scienceqa_img_mapper(one_data, args):
    hint = one_data["hint"]
    if hint:
        hint = f"Context: {hint}\n"
    question = one_data["question"]
    
    choices_list = one_data["choices"]
    choices_str = "\n".join([f"{chr(ord('A') + i)}. {choice}" for i, choice in enumerate(choices_list)])
    query = f"{hint}{question}\n{choices_str}"
    
    if args.brief:
        query = query + CHOICE_BRIEF_PROMPT
    
    one_data[QUERY_KEY] = query
    img_path = os.path.join(args.img_dir, "ScienceQA", one_data["split"], one_data["id"], one_data["image"])
    one_data[IMG_PATH_KEY] = img_path
    assert not args.use_box, "use_box should be False for scienceqa"
    return one_data


def batched_obj_to_item(batch_objs):
    batch_data = {}
    for key in batch_objs[0]:
        batch_data[key] = [obj[key] for obj in batch_objs]
    return batch_data


def save_results(dataset: datasets.Dataset, all_outputs: Dict[str, List[Any]], result_path: str):
    if len(all_outputs) == 0:
        # not rank 0
        return
    saved_dataset = dataset
    for key in MIDDLE_KEYS:
        if key in dataset.column_names:
            saved_dataset = saved_dataset.remove_columns(key)
    for key, value in all_outputs.items():
        if key in MIDDLE_KEYS:
            continue
        saved_dataset = saved_dataset.add_column(key, value)
    with open(result_path, "w") as f:
        for item in saved_dataset:
            f.write(json.dumps(item) + "\n")
    print(f"Results saved to {result_path}")
    
    
def gather_and_save_info(do_func, args, extra_infos: Dict[str, Any], info_path: str, world_size: int, local_rank: int):
    avg_time = do_func.get_average_time() / args.batch_size_per_device
    call_count = do_func.get_call_count()
    all_avg_time = [None] * world_size
    all_call_count = [None] * world_size
    dist.all_gather_object(all_avg_time, avg_time)
    dist.all_gather_object(all_call_count, call_count)
    
    if extra_infos and world_size > 1:
        # TODO: gather extra infos from all ranks
        warnings.warn("Only rank 0 will save the extra infos, other ranks will ignore them.")
    
    if local_rank == 0:
        final_avg_time = 0.0
        final_call_count = 0
        for rank_count, rank_time in zip(
            all_call_count, all_avg_time
        ):
            final_call_count += rank_count
            final_avg_time += rank_time * rank_count
        if final_call_count == 0:
            final_avg_time = 0.0
        else:
            final_avg_time /= final_call_count
        info = {
            "args": vars(args),
            "avg_time": final_avg_time,
            "call_count": final_call_count,
        }
        info.update(extra_infos)
        with open(info_path, "w") as f:
            json.dump(info, f, indent=4)
        print(f"Info saved to {info_path}")
    

def cal_box_metrics(image_token_bool_masks: List[Union[List[torch.Tensor], torch.Tensor]],
                    ref_token_masks: Optional[List[torch.Tensor]]):
    metrics = defaultdict(list)
    if ref_token_masks is not None:
        assert len(image_token_bool_masks) == len(ref_token_masks)
        for one_image_token_bool_mask, one_ref_token_mask in zip(image_token_bool_masks, ref_token_masks):
            if isinstance(one_image_token_bool_mask, list):
                one_image_token_bool_mask = one_image_token_bool_mask[0]
            one_image_token_mask = one_image_token_bool_mask.view(-1).cpu().int().numpy()
            one_ref_token_mask = one_ref_token_mask.view(-1).cpu().int().numpy()
            tp = ((one_image_token_mask == 1) & (one_ref_token_mask == 1)).sum()
            fp = ((one_image_token_mask == 1) & (one_ref_token_mask == 0)).sum()
            fn = ((one_image_token_mask == 0) & (one_ref_token_mask == 1)).sum()
            tn = ((one_image_token_mask == 0) & (one_ref_token_mask == 0)).sum()
            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
            conf_mat = np.array([[tp, fp], [fn, tn]])
            metrics[CONF_MAT_KEY].append(conf_mat)
            metrics[IOU_KEY].append(iou)
    for one_image_token_bool_mask in image_token_bool_masks:
        if isinstance(one_image_token_bool_mask, list):
            one_image_token_bool_mask = one_image_token_bool_mask[0]
        ratio = one_image_token_bool_mask.sum().item() / one_image_token_bool_mask.numel()
        metrics[RATIO_KEY].append(ratio)
    return metrics    


def gather_output(rank_outputs: Dict[str, List[Any]], total_samples: int, rank_start: int, world_size: int, local_rank: int):
    all_outputs = {}
    for key, rank_list in rank_outputs.items():
        rank_list_with_index = [(rank_start + i, item) for i, item in enumerate(rank_list)]
        all_rank_list_with_index = [None] * world_size
        dist.all_gather_object(all_rank_list_with_index, rank_list_with_index)
        assert all_rank_list_with_index[-1][-1][0] == total_samples - 1
        
        if local_rank == 0:
            all_outputs[key] = [None] * total_samples
            for rank_list in all_rank_list_with_index:
                for idx, item in rank_list:
                    assert idx < total_samples
                    all_outputs[key][idx] = item
            assert all_outputs[key][-1] is not None
    return all_outputs



def gather_extra_infos(all_outputs: Dict[str, List[Any]]) -> Dict[str, Any]:
    extra_infos = {}
    
    conf_mats = all_outputs.pop(CONF_MAT_KEY, None)
    if conf_mats is not None:
        conf_mats = np.stack(conf_mats)
        conf_mats = conf_mats.sum(axis=0)
        tp = conf_mats[0][0]
        fp = conf_mats[0][1]
        fn = conf_mats[1][0]
        tn = conf_mats[1][1]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        ratio = np.mean(all_outputs[RATIO_KEY])
        extra_infos.update({
            "mPrecision": precision,
            "mRecall": recall,
            "mF1": f1,
            "mIoU": iou,
            "mRatio": ratio,
        })
        
    num_gen_tokens = all_outputs.pop(NUM_GEN_TOKENS_KEY, None)
    if num_gen_tokens is not None:
        avg_num_gen_tokens = np.mean(num_gen_tokens)
        extra_infos["avgNumGenTokens"] = avg_num_gen_tokens
        
    all_time_logger_stats = get_all_time_logger_stats(called_only=True)
    if all_time_logger_stats:
        # extra_infos.update(all_time_logger_stats)
        for k, v in all_time_logger_stats.items():
            if k in extra_infos:
                extra_infos[k].update(v)
            else:
                extra_infos[k] = v
    all_memory_logger_stats = get_all_memory_logger_stats(called_only=True)
    if all_memory_logger_stats:
        for k, v in all_memory_logger_stats.items():
            if k in extra_infos:
                extra_infos[k].update(v)
            else:
                extra_infos[k] = v
    return extra_infos
    

def save_masks(data_ids: List[int], image_token_bool_masks: List[Union[List[torch.Tensor], torch.Tensor]], grid_hw: torch.Tensor, mask_dir: str):
    for data_id, one_image_token_masks, one_grid_hw in zip(data_ids, image_token_bool_masks, grid_hw):
        if isinstance (one_image_token_masks, list):
            for seq_id, one_seq_image_token_masks in enumerate(one_image_token_masks):
                one_seq_image_token_masks = one_seq_image_token_masks.cpu().int().numpy()
                one_h = one_grid_hw[0].item()
                one_w = one_grid_hw[1].item()
                one_seq_image_token_masks = one_seq_image_token_masks.reshape(one_h, one_w)
                one_seq_image_token_masks = Image.fromarray(one_seq_image_token_masks.astype(np.uint8) * 255)
                if seq_id == 0:
                    save_path = os.path.join(mask_dir, f"{data_id}.png")
                else:
                    save_path = os.path.join(mask_dir, f"{data_id}_seq-{seq_id}.png")
                one_seq_image_token_masks.save(save_path)
        else:
            one_image_token_masks = one_image_token_masks.cpu().int().numpy()
            one_h = one_grid_hw[0].item()
            one_w = one_grid_hw[1].item()
            one_image_token_masks = one_image_token_masks.reshape(one_h, one_w)
            one_image_token_masks = Image.fromarray(one_image_token_masks.astype(np.uint8) * 255)
            save_path = os.path.join(mask_dir, f"{data_id}.png")
            one_image_token_masks.save(save_path)


def process_one_dataset(dataset, dataset_name, world_size, local_rank, infer_model: BaseInferModel, device, args):
    # Split data for distributed evaluation
    rank_size = len(dataset) // world_size
    st = local_rank * rank_size
    ed = st + rank_size if local_rank != world_size - 1 else len(dataset)
    rank_datas = dataset.select(range(st, ed))
    rank_loader = DataLoader(rank_datas, batch_size=args.batch_size_per_device, collate_fn=batched_obj_to_item)
    rank_outputs = defaultdict(list)
    
    task_name = args.do_func_name
    if args.save_masks:
        mask_dir = os.path.join(args.output_dir, f"{dataset_name}_{task_name}_masks")
        os.makedirs(mask_dir, exist_ok=True)
    else:
        mask_dir = None
    
    
    generation_config = GenerationConfig.from_model_config(infer_model.model_config)
    generation_config.max_new_tokens = args.max_new_tokens
    generation_config.do_sample = False
    generation_config.top_p = None
    generation_config.top_k = None
    generation_config.temperature = None
    generation_config.use_cache = args.use_cache
    
    
    with tqdm(range(len(rank_datas)), disable=local_rank != 0) as pbar:
        for bid, batched_datas in enumerate(rank_loader):
            batched_querys = batched_datas[QUERY_KEY]
            batched_img_paths = batched_datas[IMG_PATH_KEY]
            batched_bboxes = batched_datas.get(BOX_KEY, None)
            curr_batch_size = len(batched_querys)
            inputs = infer_model.prepare_batch_inputs(
                batched_querys, batched_img_paths, batched_bboxes
            )
            if args.do_func_name == "glimpse":
                image_token_bool_masks = infer_model.do_glimpse(inputs, generation_config)
                ref_token_masks = inputs.get('ref_token_masks', None)
                metric_dict = cal_box_metrics(image_token_bool_masks, ref_token_masks)
                if args.save_masks:
                    curr_data_ids = [st + bid * args.batch_size_per_device + i for i in range(curr_batch_size)]
                    save_masks(curr_data_ids, image_token_bool_masks, inputs["image_grid_thw"][:, 1:]//2, mask_dir)
                for key in metric_dict:
                    rank_outputs[key].extend(metric_dict[key])
                avg_time = infer_model.do_glimpse.get_average_time() / curr_batch_size
                pbar.set_postfix_str(f"Avg glimpse time: {avg_time/1000:.4f}s")
            else:
                generated_ids = infer_model.do_generate(inputs, generation_config, args.do_selection)
                generated_length = [len(generated_id) for generated_id in generated_ids]
                batch_output_text = infer_model.batch_decode(
                    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                rank_outputs[RESPONSE_KEY].extend(batch_output_text)
                rank_outputs[NUM_GEN_TOKENS_KEY].extend(generated_length)
                avg_time = infer_model.do_generate.get_average_time() / curr_batch_size
                pbar.set_postfix_str(f"Avg generate time: {avg_time/1000:.4f}s")
            pbar.update(curr_batch_size)
            pbar.refresh()
        
    print(f"Rank {local_rank} finished processing {len(rank_datas)} samples.")
    
    all_outputs = gather_output(rank_outputs, len(dataset), st, world_size, local_rank)
    extra_infos = gather_extra_infos(all_outputs)    
    
    result_path = os.path.join(args.output_dir, f"{dataset_name}_{task_name}.jsonl")
    save_results(dataset, all_outputs, result_path)
    
    info_path = os.path.join(args.output_dir, f"{dataset_name}_{task_name}_info.json")
    if args.do_func_name == "glimpse":
        do_func = infer_model.do_glimpse            
    else:
        do_func = infer_model.do_generate
    gather_and_save_info(do_func, args, extra_infos, info_path, world_size, local_rank)
    dist.barrier()


def parse_dataset_names(dataset_names: str) -> List[str]:
    dataset_name_list = dataset_names.split(",")
    valid_names = []
    for name in dataset_name_list:
        name = name.strip()
        if name:
            valid_names.append(name)
    return valid_names
                     
              
def main():
    parser = HfArgumentParser(dataclass_types=[ScriptArgs])
    args = parser.parse_args_into_dataclasses()[0]
    
    set_seed(args.seed)
    
    if args.num_samples is not None:
        num_samples = args.num_samples
        assert num_samples > 0, "num_samples should be a positive integer"
    else:
        num_samples = None
    
    if args.do_func_name == 'glimpse' or args.use_ref_masks:
        # must use box
        if not args.use_box:
            warnings.warn("use_box should be True when do glimpse or use_ref_masks")
            args.use_box = True
        if not args.do_selection:
            warnings.warn("do_selection should be True when do glimpse or use_ref_masks")
            args.do_selection = True
                
    local_rank, world_size, _ = setup_distributed()
    device = f"cuda:{local_rank}"
    print(f"Process {local_rank} using {device}")
    
    # print(args.datasets)
    processed_datasets = {}
    dataset_names = parse_dataset_names(args.datasets)
    for name in dataset_names:
        if name == "vstar":
            dataset_path = os.path.join(args.cot_bench_dir, "test_questions.jsonl")
            mapper_func = vstar_bench_dataset_mapper
        elif name.startswith("scienceqa_img"):
            split = name.split("_")[-1]
            assert split in ["train", "val", "test"], f"Invalid split {split} for scienceqa_img dataset"
            dataset_path = f"datas/ScienceQA/problems_{split}_img.json"
            mapper_func = scienceqa_img_mapper
        elif name.startswith("refcoco"):
            dataset_path = os.path.join(args.cot_bench_dir, name + '.json')
            mapper_func = refcoco_dataset_mapper
        else:
            dataset_path = os.path.join(args.cot_bench_dir, name + '.json')
            mapper_func = cot_bench_dataset_mapper
        try:
            processed_dataset = datasets.load_dataset('json', data_files=dataset_path, split='train')
            if num_samples is not None:
                if num_samples < len(processed_dataset):
                    processed_dataset = processed_dataset.select(range(num_samples))
                    name = f"{name}_{num_samples}"
            
            processed_dataset = processed_dataset.map(mapper_func, fn_kwargs={"args": args})
            if local_rank == 0:
                print(f"Loaded {name} dataset with {len(processed_dataset)} samples.")
        except Exception as e:
            if local_rank == 0:
                print(f"Error loading {name} dataset: {e}")
            processed_dataset = None
            continue
        processed_datasets[name] = processed_dataset

    infer_model = get_model(args.model_type)(**vars(args))

    if local_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)

    with torch.inference_mode():
        for dataset_name, dataset in processed_datasets.items():
            task_name = args.do_func_name
            result_path = os.path.join(args.output_dir, f"{dataset_name}_{task_name}.jsonl")
            if os.path.exists(result_path) and not args.save_masks:
                print(f"Result file {result_path} already exists. Skipping...")
                continue
            print(f"Processing {dataset_name} dataset...")
            with time_logger_set_active(True), \
                 memory_logger_set_active(args.enable_memory_logger):
                process_one_dataset(dataset, dataset_name, world_size, local_rank, infer_model, device, args)
            reset_all_time_logger_stats()
            reset_all_memory_logger_stats()
            torch.cuda.empty_cache()
  
    # destroy the process group
    dist.destroy_process_group()        

if __name__ == "__main__":
    main()