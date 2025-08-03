import os
import ast
import json
import time
import random
import math
import argparse
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from my_transformers.models.qwen2_5_vl.processing import find_indices_of_bbox_on_grid
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

MAPPERS = {}

def register_mappers():
    def wrapper(func):
        name = func.__name__.replace("_dataset_mapper", "")
        MAPPERS[name] = func
        return func
    return wrapper

        
def norm_bbox(bbox, width, height):
    return [bbox[0]/width, bbox[1]/height, bbox[2]/width, bbox[3]/height]
    

@register_mappers()
def cot_train_dataset_mapper(one_data, args):
    image = one_data['image']
    dataset = one_data['dataset']
    img_path = os.path.join(args.img_dir, "cot", dataset, image)
    assert os.path.exists(img_path), f"Image not found: {img_path}"
    # if args.use_bbox:
    width = one_data['width']
    height = one_data['height']
    bboxes = one_data['bboxs']
    normed_bboxes = [norm_bbox(bbox, width, height) for bbox in bboxes]
    one_data['normed_bboxes'] = normed_bboxes
    return one_data

@register_mappers()
def cot_bench_dataset_mapper(one_data, args):
    img_path = os.path.join(args.img_dir, one_data["image"][0])
    img = Image.open(img_path)
    width, height = img.size
    one_data['width'] = width
    one_data['height'] = height
    # if args.use_bbox:
    bbox = one_data["image"][1].split('###')[1]
    bbox = ast.literal_eval(bbox)
    one_data['normed_bboxes'] = [norm_bbox(bbox, width, height)]
    return one_data

def process_one_data(one_data, args) -> int:
    width = one_data['width']
    height = one_data['height']
    if args.pool_size > 0 and args.version == 1:
        grid_size = (args.pool_size, args.pool_size)
    else:
        resized_height, resized_width = smart_resize(height, width, factor=28, min_pixels=args.min_pixels, max_pixels=args.max_pixels)
        grid_size = (resized_height // 28, resized_width // 28)
        if args.pool_size > 0:
            grid_size = (math.ceil(grid_size[0] / args.pool_size), math.ceil(grid_size[1] / args.pool_size))
    normed_bboxes = one_data['normed_bboxes']
    indices = find_indices_of_bbox_on_grid(normed_bboxes, grid_size)
    return grid_size[0] * grid_size[1], len(indices)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, default=2, help="Version of the vlpart")
    parser.add_argument("--dataset", type=str, help="File path", default="viscot_benchmark/benchmark/cub.json")
    parser.add_argument("--mapper", type=str, help="Mapper function", default="cot_bench")
    parser.add_argument("--split", type=str, help="Split", default="train")
    parser.add_argument("--img-dir", type=str, help="Root path of images", default="datas")
    parser.add_argument("--level", type=int, default=-1, help="If < 0, use the original image.")
    parser.add_argument("--max-pixels", type=int, default=16384*28*28)
    parser.add_argument("--min-pixels", type=int, default=4*28*28)
    parser.add_argument("--update-freq", type=int, default=10)
    args = parser.parse_args()
    
    args.pool_size = 2 ** args.level if args.level >= 0 else 0
    
    len_dataset = len(load_dataset("json", data_files=args.dataset, split=args.split))
    dataset = load_dataset("json", data_files=args.dataset, streaming=True, split=args.split)
    dataset = dataset.map(lambda x: MAPPERS[args.mapper](x, args))
    
    
    # avg_num_tokens = 0
    # min_num_tokens = 1e9
    # max_num_tokens = 0
    avg_remaining_ratio = 0
    num = 0

    with tqdm(total=len_dataset, postfix={"avg": avg_remaining_ratio}) as pbar:
        for data in dataset:
            num_tokens_ori, num_tokens_remain = process_one_data(data, args)
            remain_ratio = num_tokens_remain / num_tokens_ori
            num += 1
            # avg_num_tokens += (num_tokens - avg_num_tokens) / num
            # min_num_tokens = min(min_num_tokens, num_tokens)
            # max_num_tokens = max(max_num_tokens, num_tokens)
            avg_remaining_ratio += (remain_ratio - avg_remaining_ratio) / num
            pbar.update(1)
            if num % args.update_freq == 0:
                # pbar.set_postfix(avg=avg_num_tokens, min=min_num_tokens, max=max_num_tokens)
                pbar.set_postfix(avg=avg_remaining_ratio)
    
    if args.version == 1:
        print(f"Average remaining ratio: {avg_remaining_ratio} when pool_size={args.pool_size}")
    else:
        print(f"Average remaining ratio: {avg_remaining_ratio} when stride_size={args.pool_size}")


if __name__ == '__main__':
    main()