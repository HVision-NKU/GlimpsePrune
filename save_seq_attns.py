import os
import json
import argparse
import torch
from PIL import Image
import numpy as np
import matplotlib.cm as cm
from qwen_vl_utils import process_vision_info

from my_transformers.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration_Sep
from transformers.models.qwen2_5_vl import Qwen2_5_VLProcessor

COT_BRIEF_PROMPT = "{}\nAnswer the question using a single word or phrase."

def parse_args():
    parser = argparse.ArgumentParser(description="Save attention weights from Qwen2.5 VL model.")
    parser.add_argument("vision_info_path", type=str, help="Path to the vision info file.")
    parser.add_argument("output_dir", type=str, help="Directory to save the attention weights.")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--layer", type=int, default=23, help="Layer index to save attention weights from.")
    parser.add_argument("--head", type=int, default=-1, help="Head index to save attention weights from.")
    parser.add_argument("--color_map", type=str, default="Reds", help="Color map for attention visualization.")
    parser.add_argument("--alpha", type=float, default=0.3, help="Alpha for blending attention maps with images.")
    parser.add_argument("--brief", action="store_true", help="If set, use brief prompt")
    return parser.parse_args()


def prepare_labels_from_input_ids(input_ids, im_start_id):
    B, L = input_ids.shape
    labels = input_ids.clone()
    mask = input_ids == im_start_id
    flipped_mask = mask.flip(dims=(1,))  # Reverse the mask to find the last <|im_start|> token
    first_idx_in_flipped = torch.argmax(flipped_mask.int(), dim=1)
    last_pos = (L - 1) - first_idx_in_flipped
    mask_until_idx = last_pos + 3
    mask_until_idx = torch.clamp(mask_until_idx, max=L)
    
    arange_l = torch.arange(L, device=input_ids.device).expand(B, -1)
    modification_mask = arange_l < mask_until_idx.unsqueeze(1)
    
    labels[modification_mask] = -100   # ignore index of CrossEntropyLoss
    return labels

def reduce_tensor(src: torch.Tensor, select_idx, dim, keepdim=False):
    if select_idx == -1:
        return src.mean(dim=dim, keepdim=keepdim)
    else:
        rtn = src.select(dim, select_idx)
        if keepdim:
            rtn = rtn.unsqueeze(dim)
        return rtn

def save_attentions_on_image(image_path, attentions, label_ids, grid_hw, output_dir, color_map="jet", alpha=0.5):
    
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    grid_h, grid_w = grid_hw
    target_w, target_h = grid_w * 28, grid_h * 28
    image = image.resize((target_w, target_h), Image.LANCZOS)
    for idx, (one_attn, one_label_id) in enumerate(zip(attentions, label_ids)):
        save_path = os.path.join(output_dir, f"{idx}_{one_label_id.item()}.png")
        attention_img = one_attn.reshape(grid_h, grid_w)
        # minmax norm
        attention_img = (attention_img - attention_img.min()) / (attention_img.max() - attention_img.min())
        # interpolate to image size (nearest neighbor)
        attention_img = attention_img.unsqueeze(0).unsqueeze(0)  # (1, 1, grid_h, grid_w)
        attention_img = torch.nn.functional.interpolate(attention_img, size=(target_h, target_w), mode='nearest')
        attention_img = attention_img.squeeze(0).squeeze(0)  # (height, width)
        attention_img = (attention_img * 255).byte().cpu().numpy()

        heatmap_rgba = cm.get_cmap(color_map)(attention_img)
        heatmap_rgb = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)
        heatmap_img = Image.fromarray(heatmap_rgb, 'RGB')
        blended_img = Image.blend(image, heatmap_img, alpha=alpha)
        blended_img.save(save_path)
        print(f"Saved attention map to {save_path}")


def generate_answer(question, image_path, model, processor):
    messages = [[
        {"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": question}]}
    ]]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
    inputs = inputs.to(model.device)
    with torch.inference_mode():
        generate_ids = model.generate(**inputs, max_new_tokens=512)
        generate_ids = generate_ids[:, inputs.input_ids.shape[1]:]  # remove input_ids part
    answer = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
    return answer


def main():
    args = parse_args()
    model = Qwen2_5_VLForConditionalGeneration_Sep.from_pretrained(args.model_path,
                                                                   torch_dtype=torch.bfloat16,
                                                                   attn_implementation="flash_attention_2",
                                                                   device_map="auto")
    model.eval()
    processor = Qwen2_5_VLProcessor.from_pretrained(args.model_path, padding_side="left")
    im_start_id = processor.tokenizer.encode("<|im_start|>")[0]
    print(f"Using <|im_start|> token ID: {im_start_id}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(args.vision_info_path, 'r') as f:
        vision_info = json.load(f)
    
    for info_idx, one_info in enumerate(vision_info):
        question = one_info['question']
        image_path = one_info['image_path']
        answer = one_info.get('answer', None)
        
        if args.brief:
            question = COT_BRIEF_PROMPT.format(question)
        
        if answer is None:
            answer = generate_answer(question, image_path, model, processor)
            print(f"Generated answer: {answer}")
        
        messages = [[
            {"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": question}]},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]}
        ]]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        print(text)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
                text=text,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        inputs = inputs.to(model.device)
        labels = prepare_labels_from_input_ids(inputs.input_ids, im_start_id)
        # print(f"input_ids: {inputs.input_ids[0].tolist()}")
        # print(f"labels: {labels[0].tolist()}")
        with torch.inference_mode():
            output = model(**inputs, labels=labels, output_attentions=True, return_dict=True)
        attentions = output.attentions  # tuple(layers) of list(bsz) of (k_select_len, nheads, q_len)
        attentions = [one_layer[0] for one_layer in attentions]  # tuple(layers) of (k_select_len, nheads, q_len)
        attentions = torch.stack(attentions, dim=0)   # (layers, k_select_len, nheads, q_len)
        attentions = reduce_tensor(attentions, args.layer, 0)  # (k_select_len, nheads, q_len)
        attentions = reduce_tensor(attentions, args.head, 1)  # (k_select_len, q_len)
        attentions = attentions.transpose(0, 1)  # (q_len, k_select_len)
        attentions = attentions[:-1]
        label_mask = labels != -100
        label_ids = inputs.input_ids[label_mask]  # (q_len, )
        assert len(label_ids) == attentions.shape[0], f"Label IDs length {len(label_ids)} does not match attentions length {attentions.shape[0]}"
        grid_hw = inputs["image_grid_thw"][0, 1:] // 2
        grid_hw = grid_hw.tolist()
        assert grid_hw[0] * grid_hw[1] == attentions.shape[1], f"Grid size {grid_hw} does not match attentions width {attentions.shape[1]}"
        image_name = os.path.basename(image_path).split('.')[0]
        dir_suffix = f"_layer{args.layer}_head{args.head}"
        if args.brief:
            dir_suffix += "_brief"
        save_dir = os.path.join(args.output_dir, f"{info_idx}_" + image_name + dir_suffix)
        os.makedirs(save_dir, exist_ok=True)
        save_attentions_on_image(image_path, attentions, label_ids, grid_hw, save_dir,
                                 color_map=args.color_map, alpha=args.alpha)
        
        id2str_path = os.path.join(save_dir, "id2str.json")
        id2str = {}
        for idx, one_label_id in enumerate(label_ids):
            one_label_str = processor.tokenizer.decode(one_label_id)
            id2str[idx] = (one_label_id.item(), one_label_str)
        with open(id2str_path, 'w') as f:
            json.dump(id2str, f, indent=4, ensure_ascii=False)
        print(f"Saved label IDs to {id2str_path}")
            
        
        
if __name__ == "__main__":
    main()
