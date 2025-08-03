import argparse
import torch
import os
import json
import pandas as pd
from tqdm import tqdm
import shortuuid

from llava_vscan.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava_vscan.conversation import conv_templates, SeparatorStyle
from llava_vscan.model.builder import load_pretrained_model
from llava_vscan.utils import disable_torch_init
from llava_vscan.mm_utils import tokenizer_image_token, process_images, load_image_from_base64, get_model_name_from_path

from PIL import Image
import math

import os.path
from torch.nn import functional as F
from clip_prs.utils.factory import create_model_and_transforms, get_tokenizer
from hook import hook_prs_logger
import numpy as np
import sys
import ast


all_options = ['A', 'B', 'C', 'D']


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def is_none(value):
    if value is None:
        return True
    if type(value) is float and math.isnan(value):
        return True
    if type(value) is str and value.lower() == 'nan':
        return True
    if type(value) is str and value.lower() == 'none':
        return True
    return False

def get_options(row, options):
    parsed_options = []
    for option in options:
        option_value = row[option]
        if is_none(option_value):
            break
        parsed_options.append(option_value)
    return parsed_options

def get_clip_model(clip_model_name = "ViT-L-14-336", layer_index = 23): # "ViT-L-14", "ViT-B-32"
    ## Hyperparameters
    device = 'cuda'
    pretrained = 'openai'

    ## Loading Model
    clip_model, _, clip_preprocess = create_model_and_transforms(clip_model_name, pretrained=pretrained)
    clip_model.to(device)
    clip_model.eval()
    context_length = clip_model.context_length
    vocab_size = clip_model.vocab_size
    clip_tokenizer = get_tokenizer(clip_model_name)

    prs = hook_prs_logger(clip_model, device, layer_index)

    return clip_model, prs, clip_preprocess, device, clip_tokenizer

def get_sorted_indices_clip(clip_model, prs, clip_preprocess, device, clip_tokenizer, image_folder, row):
    images = []
    image_pils = []
    
    line = row
    image_pil = load_image_from_base64(line['image'])
    # image_pil = Image.open(image_file)
    image = clip_preprocess(image_pil)[np.newaxis, :, :, :]
    images.append(image)
    image_pils.append(image_pil)
    image = torch.cat(images, dim=0).to(device)

    prs.reinit()
    with torch.no_grad():
        representation = clip_model.encode_image(image, 
                                            attn_method='head', 
                                            normalize=False) 
        attentions, mlps = prs.finalize(representation) 
        
    qs = line["question"]

    texts = clip_tokenizer(qs).to(device)
    class_embeddings = clip_model.encode_text(texts)
    class_embedding = F.normalize(class_embeddings, dim=-1)
    
    attention_map = attentions[:, 0, 1:, :]
    attention_map = torch.einsum('bnd,bd->bn', attention_map, class_embedding)
    # Sort the attentions
    sorted_indices = torch.argsort(attention_map, descending=True)
    return sorted_indices

def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    if eval(args.layer_list) is not None:
        llm_pruning = True  # whether to use middle layer pruning at LLM stage
    else:
        llm_pruning = False
    
    selected_indices = []

    questions = pd.read_table(os.path.expanduser(args.question_file))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, 
                                                                           args.model_base, 
                                                                           model_name,
                                                                           llm_pruning,
                                                                           visual_token_num=args.visual_token_num,
                                                                           selected_indices=selected_indices
                                                                           )

    model_class_name = type(model).__name__
    if model_class_name == "LlavaLlamaForCausalLM_X":
        model.model.layer_list = eval(args.layer_list)
        model.model.image_token_list = eval(args.image_token_list)
        model.model.image_token_list.insert(0, args.visual_token_num)

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    counter = 0
    for index, row in tqdm(questions.iterrows(), total=len(questions)):
        counter += 1
        options = get_options(row, all_options)
        cur_option_char = all_options[:len(options)]

        if args.all_rounds:
            num_rounds = len(options)
        else:
            num_rounds = 1

        for round_idx in range(num_rounds):
            idx = row['index']
            question = row['question']
            hint = row['hint']
            image = load_image_from_base64(row['image'])
            if not is_none(hint):
                question = hint + '\n' + question
            for option_char, option in zip(all_options[:len(options)], options):
                question = question + '\n' + option_char + '. ' + option
            qs = cur_prompt = question
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            if args.single_pred_prompt:
                if args.lang == 'cn':
                    qs = qs + '\n' + "请直接回答选项字母。"
                else:
                    qs = qs + '\n' + "Answer with the option's letter from the given choices directly."

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            image_tensor = process_images([image], image_processor, model.config)[0]

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.half().cuda(), #.unsqueeze(0).half().cuda(),
                    image_sizes=[image.size],
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=1024,
                    use_cache=True)

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({"question_id": idx,
                                    "round_id": round_idx,
                                    "prompt": cur_prompt,
                                    "text": outputs,
                                    "options": options,
                                    "option_char": cur_option_char,
                                    "answer_id": ans_id,
                                    "model_id": model_name,
                                    "metadata": {}}) + "\n")
            ans_file.flush()

            # rotate options
            options = options[1:] + options[:1]
            cur_option_char = cur_option_char[1:] + cur_option_char[:1]
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--all-rounds", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--layer_list", type=str, default= None)
    parser.add_argument("--image_token_list", type=str, default= None)
    parser.add_argument("--llm_pruning", action='store_true')
    parser.add_argument("--visual_token_num", type=int, default=576)
    args = parser.parse_args()

    eval_model(args)
