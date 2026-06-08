import math

import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer


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
        return cls(tokenizer, max_dynamic_patch=max_dynamic_patch, min_dynamic_patch=min_dynamic_patch, use_thumbnail=use_thumbnail)

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


__all__ = [
    "InternVLGPProcessor",
]

