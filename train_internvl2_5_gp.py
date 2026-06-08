import math
import os
from dataclasses import dataclass
from typing import Optional

import torch
from PIL import Image
from peft import get_peft_model
from torch import nn
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer
from trl import TrlParser, get_peft_config

from train_qwen_gp import (
    ANSWER_KEY,
    GPModelConfig,
    GPScriptArguments,
    GPDataset,
    GPTrainer,
    GPTrainingArguments,
    IMG_PATH_KEY,
    NORMED_BBOXES_KEY,
    QUERY_KEY,
    SCORE_FUNCS_KEY,
)
from transformers_gp.models.internvl2_5 import InternVL2_5_GP_ForConditionalGeneration


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = 448
TOKENS_PER_SIDE = 16
IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
SYSTEM_MESSAGE = "你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大模型。"
SEP = "<|im_end|>\n"


def build_transform(input_size=IMAGE_SIZE):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess_with_boxes(image, min_num=1, max_num=12, image_size=IMAGE_SIZE, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    scale_x = orig_width / target_width
    scale_y = orig_height / target_height

    processed = []
    for i in range(blocks):
        x0 = (i % (target_width // image_size)) * image_size
        y0 = (i // (target_width // image_size)) * image_size
        x1 = x0 + image_size
        y1 = y0 + image_size
        tile = resized_img.crop((x0, y0, x1, y1))
        orig_box = (x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y)
        processed.append((tile, orig_box))

    if use_thumbnail and len(processed) != 1:
        processed.append((image.resize((image_size, image_size)), (0.0, 0.0, float(orig_width), float(orig_height))))
    return processed


def bbox_mask_for_tile(normed_bboxes, tile_box, image_size):
    x0, y0, x1, y1 = tile_box
    tile_w = max(1e-6, x1 - x0)
    tile_h = max(1e-6, y1 - y0)
    mask = torch.zeros((TOKENS_PER_SIDE, TOKENS_PER_SIDE), dtype=torch.bool)
    if normed_bboxes is None:
        return mask
    for bbox in normed_bboxes:
        bx0, by0, bx1, by1 = bbox
        bx0 *= image_size[0]
        bx1 *= image_size[0]
        by0 *= image_size[1]
        by1 *= image_size[1]
        ix0 = max(bx0, x0)
        iy0 = max(by0, y0)
        ix1 = min(bx1, x1)
        iy1 = min(by1, y1)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        gx0 = int(math.floor((ix0 - x0) / tile_w * TOKENS_PER_SIDE))
        gy0 = int(math.floor((iy0 - y0) / tile_h * TOKENS_PER_SIDE))
        gx1 = int(math.ceil((ix1 - x0) / tile_w * TOKENS_PER_SIDE))
        gy1 = int(math.ceil((iy1 - y0) / tile_h * TOKENS_PER_SIDE))
        gx0 = max(0, min(TOKENS_PER_SIDE - 1, gx0))
        gy0 = max(0, min(TOKENS_PER_SIDE - 1, gy0))
        gx1 = max(gx0 + 1, min(TOKENS_PER_SIDE, gx1))
        gy1 = max(gy0 + 1, min(TOKENS_PER_SIDE, gy1))
        mask[gy0:gy1, gx0:gx1] = True
    return mask


class InternVLGPProcessor:
    def __init__(self, tokenizer, max_dynamic_patch=12, min_dynamic_patch=1, use_thumbnail=True):
        self.tokenizer = tokenizer
        self.transform = build_transform()
        self.max_dynamic_patch = max_dynamic_patch
        self.min_dynamic_patch = min_dynamic_patch
        self.use_thumbnail = use_thumbnail
        self.num_image_token = 256

    @classmethod
    def from_pretrained(cls, model_name_or_path, max_dynamic_patch=12, min_dynamic_patch=1, use_thumbnail=True, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, use_fast=False)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return cls(
            tokenizer,
            max_dynamic_patch=max_dynamic_patch,
            min_dynamic_patch=min_dynamic_patch,
            use_thumbnail=use_thumbnail,
        )

    def save_pretrained(self, output_dir):
        self.tokenizer.save_pretrained(output_dir)

    def load_image(self, image_path, normed_bboxes=None):
        image = Image.open(image_path).convert("RGB")
        tiles = dynamic_preprocess_with_boxes(
            image,
            min_num=self.min_dynamic_patch,
            max_num=self.max_dynamic_patch,
            use_thumbnail=self.use_thumbnail,
        )
        pixel_values = []
        ref_masks = []
        for tile, tile_box in tiles:
            pixel_values.append(self.transform(tile))
            ref_masks.append(bbox_mask_for_tile(normed_bboxes, tile_box, image.size).reshape(-1))
        return torch.stack(pixel_values), torch.cat(ref_masks, dim=0)

    def build_prompt(self, question, num_patches, answer=None):
        image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
        question = question if "<image>" in question else "<image>\n" + question
        question = question.replace("<image>", image_tokens, 1)
        prefix = (
            f"<|im_start|>system\n{SYSTEM_MESSAGE}{SEP}"
            f"<|im_start|>user\n{question}{SEP}"
            f"<|im_start|>assistant\n"
        )
        if answer is None:
            return prefix
        return prefix + answer + SEP


class InternVLGPCollator:
    def __init__(self, processor, is_sft):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.is_sft = is_sft

    def __call__(self, features):
        pixel_values_list = []
        ref_token_masks = []
        num_patches_list = []
        prompts = []
        prefix_prompts = []
        querys = []
        answers = []
        score_funcs = []
        for feature in features:
            pixel_values, ref_mask = self.processor.load_image(
                feature[IMG_PATH_KEY],
                normed_bboxes=feature[NORMED_BBOXES_KEY],
            )
            num_patches = pixel_values.shape[0]
            pixel_values_list.append(pixel_values)
            ref_token_masks.append(ref_mask)
            num_patches_list.append(num_patches)
            query = feature[QUERY_KEY]
            answer = feature[ANSWER_KEY]
            prefix_prompts.append(self.processor.build_prompt(query, num_patches, answer=None))
            prompts.append(self.processor.build_prompt(query, num_patches, answer=answer if self.is_sft else None))
            querys.append(query)
            answers.append(answer)
            score_funcs.append(feature[SCORE_FUNCS_KEY])

        model_inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)
        inputs = {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "pixel_values": torch.cat(pixel_values_list, dim=0),
            "num_patches_list": num_patches_list,
            "ref_token_masks": ref_token_masks,
            QUERY_KEY: querys,
            ANSWER_KEY: answers,
            SCORE_FUNCS_KEY: score_funcs,
        }
        if self.is_sft:
            labels = inputs["input_ids"].clone()
            prefix_ids = [self.tokenizer(prefix, add_special_tokens=True)["input_ids"] for prefix in prefix_prompts]
            pad_id = self.tokenizer.pad_token_id
            for i, one_prefix_ids in enumerate(prefix_ids):
                nonpad = inputs["input_ids"][i].ne(pad_id).nonzero(as_tuple=False).flatten()
                st = int(nonpad[0].item()) if nonpad.numel() else 0
                labels[i, : st + len(one_prefix_ids)] = -100
                labels[i, inputs["attention_mask"][i] == 0] = -100
            inputs["labels"] = labels
        return inputs


@dataclass
class InternVLGPScriptArguments(GPScriptArguments):
    max_dynamic_patch: int = 12
    min_dynamic_patch: int = 1
    use_thumbnail: bool = True


def main():
    parser = TrlParser((InternVLGPScriptArguments, GPTrainingArguments, GPModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    processor = InternVLGPProcessor.from_pretrained(
        model_args.model_name_or_path,
        max_dynamic_patch=script_args.max_dynamic_patch,
        min_dynamic_patch=script_args.min_dynamic_patch,
        use_thumbnail=script_args.use_thumbnail,
    )
    train_dataset = GPDataset(script_args.train_dataset, processor, script_args) if script_args.train_dataset else None
    data_collator = InternVLGPCollator(processor, is_sft=training_args.le_weight > 0)

    model_args_dict = vars(model_args)
    model_init_kwargs = {}
    for key, value in model_args_dict.items():
        if "peft" in key or "lora" in key or "dora" in key:
            continue
        model_init_kwargs[key] = value
    model_name_or_path = model_init_kwargs.pop("model_name_or_path")
    gp_config = InternVL2_5_GP_ForConditionalGeneration.config_class(**model_init_kwargs)
    model = InternVL2_5_GP_ForConditionalGeneration.from_pretrained(
        model_name_or_path,
        config=gp_config,
        tokenizer=processor.tokenizer,
        torch_dtype=getattr(torch, model_args.torch_dtype) if model_args.torch_dtype not in (None, "auto") else torch.bfloat16,
        device_map={"": int(os.environ.get("LOCAL_RANK", 0))},
    )
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


if __name__ == "__main__":
    main()
