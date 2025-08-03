import os
import argparse
import datasets
from transformers.models.qwen2_5_vl import (
    Qwen2_5_VLProcessor,
    Qwen2_5_VLConfig,
)

from transformers_gp.models.qwen2_5_vl.configuration import Qwen2_5_VL_GPConfig
from qwen_vl_utils import process_vision_info

def self_attn_flops(
    seq_len: int,
    hidden_size: int,
    num_heads: int,
    num_key_value_heads: int,
    head_dim: int,
):
    assert hidden_size == num_heads * head_dim, "hidden_size must be equal to num_heads * head_dim"
    assert num_heads % num_key_value_heads == 0, "num_heads must be divisible by num_key_value_heads"
    
    q_proj_flops = 2 * seq_len * hidden_size * hidden_size
    k_proj_flops = 2 * seq_len * hidden_size * num_key_value_heads * head_dim
    v_proj_flops = 2 * seq_len * hidden_size * num_key_value_heads * head_dim
    
    attn_scores_flops = 2 * seq_len * seq_len * hidden_size
    weight_sum_flops = 2 * seq_len * seq_len * hidden_size
    
    out_proj_flops = 2 * seq_len * hidden_size * hidden_size
    
    total_flops = (
        q_proj_flops
        + k_proj_flops
        + v_proj_flops
        + attn_scores_flops
        + weight_sum_flops
        + out_proj_flops
    )
    return total_flops
    

def self_attn_flops_decoding(
    seq_len: int,
    hidden_size: int,
    num_heads: int,
    num_key_value_heads: int,
    head_dim: int,
):
    assert hidden_size == num_heads * head_dim, "hidden_size must be equal to num_heads * head_dim"
    assert num_heads % num_key_value_heads == 0, "num_heads must be divisible by num_key_value_heads"
    
    q_proj_flops = 2 * hidden_size * hidden_size
    k_proj_flops = 2 * hidden_size * num_key_value_heads * head_dim
    v_proj_flops = 2 * hidden_size * num_key_value_heads * head_dim
    attn_scores_flops = 2 * seq_len * hidden_size
    weight_sum_flops = 2 * seq_len * hidden_size
    out_proj_flops = 2 * hidden_size * hidden_size
    total_flops = (
        q_proj_flops
        + k_proj_flops
        + v_proj_flops
        + attn_scores_flops
        + weight_sum_flops
        + out_proj_flops
    )
    return total_flops
    

def cond_attn_flops(
    seq_len: int,
    hidden_size: int,
    cond_size: int,
    num_heads: int,
    num_key_value_heads: int,
    head_dim: int,
):
    qk_size = hidden_size + cond_size
    assert qk_size == num_heads * head_dim, "hidden_size must be equal to num_heads * head_dim"
    assert num_heads % num_key_value_heads == 0, "num_heads must be divisible by num_key_value_heads"

    
    q_proj_flops = 2 * seq_len * qk_size * qk_size
    k_proj_flops = 2 * seq_len * qk_size * num_key_value_heads * head_dim
    v_proj_flops = 2 * seq_len * hidden_size * num_key_value_heads * head_dim
    attn_scores_flops = 2 * seq_len * seq_len * qk_size
    weight_sum_flops = 2 * seq_len * seq_len * hidden_size

    out_proj_flops = 2 * hidden_size * hidden_size
    total_flops = (
        q_proj_flops
        + k_proj_flops
        + v_proj_flops
        + attn_scores_flops
        + weight_sum_flops
        + out_proj_flops
    )
    return total_flops


def qwen2_5_mlp_flops(
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
):
    gate_proj_flops = 2 * seq_len * hidden_size * intermediate_size
    up_proj_flops = 2 * seq_len * hidden_size * intermediate_size
    down_proj_flops = 2 * seq_len * intermediate_size * hidden_size
    return gate_proj_flops + up_proj_flops + down_proj_flops


def qwen2_5_mlp_flops_decoding(
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
):
    gate_proj_flops = 2 * hidden_size * intermediate_size
    up_proj_flops = 2 * hidden_size * intermediate_size
    down_proj_flops = 2 * intermediate_size * hidden_size
    return gate_proj_flops + up_proj_flops + down_proj_flops


def qwen2_5_prefilling_flops(
    seq_len: int,
    hidden_size: int,
    num_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    intermediate_size: int,
    num_layers: int,
):
    attn_flops = self_attn_flops(
        seq_len, hidden_size, num_heads, num_key_value_heads, head_dim
    )
    mlp_flops = qwen2_5_mlp_flops(seq_len, hidden_size, intermediate_size)
    total_flops = num_layers * (attn_flops + mlp_flops)
    return total_flops


def vip_flops(
    seq_len: int,
    hidden_size: int,
    cond_size: int,
    num_heads: int,
    num_layers: int,
):
    num_key_value_heads = num_heads
    head_dim = (hidden_size + cond_size) // num_heads
    intermediate_size = hidden_size * 2
    attn_flops = cond_attn_flops(
        seq_len, hidden_size, cond_size, num_heads, num_key_value_heads, head_dim
    )
    mlp_flops = qwen2_5_mlp_flops(seq_len, hidden_size, intermediate_size)
    total_flops = (attn_flops + mlp_flops) * num_layers
    return total_flops


def qwen2_5_gp_prefilling_flops(
    seq_len: int,
    hidden_size: int,
    num_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    intermediate_size: int,
    vip_hidden_size: int,
    vip_num_heads: int,
    vip_cond_size: int,
    vip_num_layers: int,
    vip_seq_len: int,
    num_layers: int,
    prune_layer: int,
    prune_seq_len: int,
):
    attn_flops_bef_prune = self_attn_flops(
        seq_len, hidden_size, num_heads, num_key_value_heads, head_dim
    )
    mlp_flops_bef_prune = qwen2_5_mlp_flops(seq_len, hidden_size, intermediate_size)
    the_vip_flops = vip_flops(
        vip_seq_len, vip_hidden_size, vip_cond_size, vip_num_heads, vip_num_layers
    )
    attn_flops_aft_prune = self_attn_flops(
        prune_seq_len, hidden_size, num_heads, num_key_value_heads, head_dim    
    )
    mlp_flops_aft_prune = qwen2_5_mlp_flops(
        prune_seq_len, hidden_size, intermediate_size
    )
    layers_bef_prune = prune_layer
    layers_aft_prune = num_layers - prune_layer
    total_flops = (
        layers_bef_prune * (attn_flops_bef_prune + mlp_flops_bef_prune)
        + layers_aft_prune * (attn_flops_aft_prune + mlp_flops_aft_prune)
        + the_vip_flops
    )
    return total_flops
    

def qwen2_5_decoding_flops(
    seq_len: int,
    hidden_size: int,
    num_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    intermediate_size: int,
    num_layers: int,
    gen_len: int,
):
    all_flops = 0
    for i in range(gen_len):
        cache_len = seq_len + i
        attn_flops = self_attn_flops_decoding(
            cache_len, hidden_size, num_heads, num_key_value_heads, head_dim
        )
        mlp_flops = qwen2_5_mlp_flops_decoding(cache_len, hidden_size, intermediate_size)
        total_flops = (attn_flops + mlp_flops) * num_layers
        all_flops += total_flops
    return all_flops


def qwen2_5_llm_7b_flops(
    seq_len: int,
):
    hidden_size = 3584
    num_heads = 28
    num_key_value_heads = 4
    head_dim = 128
    intermediate_size = 18944
    num_layers = 28
    
    return qwen2_5_prefilling_flops(
        seq_len, hidden_size, num_heads, num_key_value_heads, head_dim, intermediate_size, num_layers
    )
    
def qwen2_5_llm_7b_flops_decoding(
    seq_len: int,
    gen_len: int,
):
    hidden_size = 3584
    num_heads = 28
    num_key_value_heads = 4
    head_dim = 128
    intermediate_size = 18944
    num_layers = 28
    
    return qwen2_5_decoding_flops(
        seq_len, hidden_size, num_heads, num_key_value_heads, head_dim, intermediate_size, num_layers, gen_len
    )
    

def _format_flops(flops: int|float) -> str:
    if flops >= 1e12:
        return f"{flops / 1e12:.2f} T"
    elif flops >= 1e9:
        return f"{flops / 1e9:.2f} B"
    elif flops >= 1e6:
        return f"{flops / 1e6:.2f} M"
    else:
        return f"{flops} F"


COT_BRIEF_PROMPT = "{}\n\nAnswer the question using a single word or phrase."
QUERY_KEY = "query"
IMG_PATH_KEY = "img_path"
RESP_KEY = "response"
RATIO_KEY = "ratio"


def cot_bench_dataset_mapper(one_data, args):
    query = one_data["conversations"][0]["value"].replace("Please provide the bounding box coordinate of the region that can help you answer the question better.", "").strip()
    query = query.replace("<image>\n", "")
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
    return one_data


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate FLOPs for Qwen2.5 models")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="Base model name or path")
    parser.add_argument("--config", type=str, help="Path to the model configuration file")
    parser.add_argument("--generate", type=str, help="Path to generate result file")
    parser.add_argument("--glimpse", type=str, help="Path to glimpse result file")
    parser.add_argument("--num_samples", type=int, default=100, help="Upper limit of samples")
    parser.add_argument("--img_dir", type=str, default="datas", help="Directory containing images")
    return parser.parse_args()

def get_seq_lens(query, img_path, resp, ratio, processor, config):
    messages = [[{"role": "user", "content": [{"type": "image", "image": img_path}, {"type": "text", "text": query}]}]]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
                text=text,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
    input_seq_len = inputs.input_ids.shape[1]
    image_token_id = config.image_token_id
    visual_seq_len = (inputs.input_ids == image_token_id).sum().item()
    assert visual_seq_len < input_seq_len, "Visual sequence length should be less than input sequence length"
    remain_visual_seq_len = int(visual_seq_len * ratio)
    remain_input_seq_len = input_seq_len - visual_seq_len + remain_visual_seq_len
    

    # full_messages = [[{"role": "user", "content": [{"type": "image", "image": img_path}, {"type": "text", "text": query}]},
    #                 {"role": "assistant", "content": [{"type": "text", "text": resp}]}]]
    # full_text = processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
    # full_inputs = processor(
    #             text=full_text,
    #             images=image_inputs,
    #             videos=video_inputs,
    #             padding=True,
    #             return_tensors="pt",
    #         )
    # gen_seq_len = full_inputs.input_ids.shape[1] - input_seq_len
    get_seq_len = processor.tokenizer.encode(resp, add_special_tokens=False)
    gen_seq_len = len(get_seq_len)
    assert gen_seq_len > 0, "Generated sequence length should be greater than 0"
    return (input_seq_len, visual_seq_len, remain_input_seq_len, remain_visual_seq_len, gen_seq_len)
    

def main():
    args = parse_args()
    config = Qwen2_5_VL_GPConfig.from_json_file(args.config)

    assert os.path.exists(args.generate)
    assert os.path.exists(args.glimpse)
    generate_datas = datasets.load_dataset("json", data_files=args.generate, split="train")
    glimpse_datas = datasets.load_dataset("json", data_files=args.glimpse, split="train")
    if args.num_samples is not None:
        generate_datas = generate_datas.select(range(args.num_samples))
        glimpse_datas = glimpse_datas.select(range(args.num_samples))
    
    
    
    generate_datas = generate_datas.map(cot_bench_dataset_mapper, fn_kwargs={"args": args})
    glimpse_datas = glimpse_datas.map(cot_bench_dataset_mapper, fn_kwargs={"args": args})
    
    processor = Qwen2_5_VLProcessor.from_pretrained(args.base_model, padding_side="left")
    
    avg_qwen2_5_prefilling_flops = 0
    avg_qwen2_5_decoding_flops = 0
    avg_qwen2_5_gp_prefilling_flops = 0
    avg_qwen2_5_gp_decoding_flops = 0
    avg_input_seq_len = 0
    avg_remain_input_seq_len = 0
    avg_gen_seq_len = 0
    n = 0
    
    for one_generate_data, one_glimpse_data in zip(generate_datas, glimpse_datas):
        query = one_generate_data[QUERY_KEY]
        img_path = one_generate_data[IMG_PATH_KEY]
        resp = one_generate_data[RESP_KEY]
        ratio = one_glimpse_data[RATIO_KEY]
        
        (
            input_seq_len,
            visual_seq_len,
            remain_input_seq_len,
            remain_visual_seq_len,
            gen_seq_len
        ) = get_seq_lens(
            query, img_path, resp, ratio, processor, config
        )
        print(f"Query: {query}")
        print(f"Image Path: {img_path}")
        print(f"Response: {resp}")
        print(f"Ratio: {ratio}")
        print(f"Input Sequence Length: {input_seq_len}")
        print(f"Visual Sequence Length: {visual_seq_len}")
        print(f"Remaining Input Sequence Length: {remain_input_seq_len}")
        print(f"Remaining Visual Sequence Length: {remain_visual_seq_len}")
        print(f"Generated Sequence Length: {gen_seq_len}")
        
        qwen2_5_prefilling_flops_value = qwen2_5_prefilling_flops(
            input_seq_len,
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.hidden_size // config.num_attention_heads,
            config.intermediate_size,
            config.num_hidden_layers
        )
        print(f"Qwen2.5 Prefilling FLOPs: {_format_flops(qwen2_5_prefilling_flops_value)}")
        
        qwen2_5_decoding_flops_value = qwen2_5_decoding_flops(
            input_seq_len,
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.hidden_size // config.num_attention_heads,
            config.intermediate_size,
            config.num_hidden_layers,
            gen_seq_len
        )
        print(f"Qwen2.5 Decoding FLOPs: {_format_flops(qwen2_5_decoding_flops_value)}")
        
        qwen2_5_gp_prefilling_flops_value = qwen2_5_gp_prefilling_flops(
            input_seq_len,
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.hidden_size // config.num_attention_heads,
            config.intermediate_size,
            config.attn_fuse_size,
            config.attn_fuse_num_heads,
            config.visual_cond_size,
            len(config.selected_visual_layers),
            visual_seq_len,
            config.num_hidden_layers,
            config.reduce_layer + 1,
            remain_visual_seq_len
        )
        print(f"Qwen2.5 GP Prefilling FLOPs: {_format_flops(qwen2_5_gp_prefilling_flops_value)}")
        
        qwen2_5_gp_decoding_flops_value = qwen2_5_decoding_flops(
            remain_input_seq_len,
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.hidden_size // config.num_attention_heads,
            config.intermediate_size,
            config.num_hidden_layers,
            gen_seq_len
        )
        print(f"Qwen2.5 GP Decoding FLOPs: {_format_flops(qwen2_5_gp_decoding_flops_value)}")
        
        # update averages
        n += 1
        avg_qwen2_5_prefilling_flops += (qwen2_5_prefilling_flops_value - avg_qwen2_5_prefilling_flops) / n
        avg_qwen2_5_decoding_flops += (qwen2_5_decoding_flops_value - avg_qwen2_5_decoding_flops) / n
        avg_qwen2_5_gp_prefilling_flops += (qwen2_5_gp_prefilling_flops_value - avg_qwen2_5_gp_prefilling_flops) / n
        avg_qwen2_5_gp_decoding_flops += (qwen2_5_gp_decoding_flops_value - avg_qwen2_5_gp_decoding_flops) / n
        avg_input_seq_len += (input_seq_len - avg_input_seq_len) / n
        avg_remain_input_seq_len += (remain_input_seq_len - avg_remain_input_seq_len) / n
        avg_gen_seq_len += (gen_seq_len - avg_gen_seq_len) / n

        print("=============================================")

    
    print(f"Base Model: {args.base_model}")
    print(f"Config: {args.config}")
    print(f"Generate File: {args.generate}")
    print(f"Glimpse File: {args.glimpse}")
    print(f"Number of Samples: {n}") 
        
    print(f"Average Qwen2.5 Prefilling FLOPs: {_format_flops(avg_qwen2_5_prefilling_flops)}")
    print(f"Average Qwen2.5 Decoding FLOPs: {_format_flops(avg_qwen2_5_decoding_flops)}")
    print(f"Average Qwen2.5 GP Prefilling FLOPs: {_format_flops(avg_qwen2_5_gp_prefilling_flops)}")
    print(f"Average Qwen2.5 GP Decoding FLOPs: {_format_flops(avg_qwen2_5_gp_decoding_flops)}")
    print(f"Average input seq len: {avg_input_seq_len}")
    print(f"Average remain input seq len: {avg_remain_input_seq_len}")
    print(f"Average generate seq len: {avg_gen_seq_len}")

if __name__ == "__main__":
    main()