import math
import os
from dataclasses import dataclass

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from .base import BaseInferModel


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = 448
DEFAULT_MAX_TILES = 12


@dataclass
class InternVLGeneratedResponse:
    text: str
    token_count: int

    def __len__(self):
        return self.token_count


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


def dynamic_preprocess(image, min_num=1, max_num=DEFAULT_MAX_TILES, image_size=IMAGE_SIZE, use_thumbnail=True):
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
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))

    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


class InternVL2_5(BaseInferModel):
    def _init_model(self, **kwargs):
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        use_flash_attn = self._attn_implementation == "flash_attention_2"
        model = AutoModel.from_pretrained(
            self._base_model,
            torch_dtype=self._torch_dtype,
            low_cpu_mem_usage=True,
            use_flash_attn=use_flash_attn,
            trust_remote_code=True,
        )
        self._model = model.to(device).eval()
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._base_model,
            trust_remote_code=True,
            use_fast=False,
        )

    def _init_processor(self, min_pixels=None, max_pixels=None, **kwargs):
        self._transform = build_transform(IMAGE_SIZE)
        self._min_tiles = self._pixels_to_tiles(min_pixels, default=1)
        self._max_tiles = self._pixels_to_tiles(max_pixels, default=DEFAULT_MAX_TILES)
        if self._min_tiles > self._max_tiles:
            raise ValueError(
                f"InternVL min tile count ({self._min_tiles}) cannot exceed max tile count ({self._max_tiles})."
            )
        print(f"InternVL2.5 dynamic tiles: min={self._min_tiles}, max={self._max_tiles}")

    @staticmethod
    def _pixels_to_tiles(num_pixels, default):
        if num_pixels is None:
            return default
        return max(1, int(math.ceil(num_pixels / (IMAGE_SIZE * IMAGE_SIZE))))

    @property
    def model_config(self):
        return self._model.config

    def _load_image(self, image_file):
        image = Image.open(image_file).convert("RGB")
        images = dynamic_preprocess(
            image,
            min_num=self._min_tiles,
            max_num=self._max_tiles,
            image_size=IMAGE_SIZE,
            use_thumbnail=True,
        )
        pixel_values = [self._transform(tile) for tile in images]
        return torch.stack(pixel_values)

    def _do_generate(self, inputs, generation_config, do_selection):
        generation_kwargs = {
            "max_new_tokens": generation_config.max_new_tokens,
            "do_sample": False,
        }

        responses = self._model.batch_chat(
            self._tokenizer,
            inputs["pixel_values"],
            num_patches_list=inputs["num_patches_list"],
            questions=inputs["questions"],
            generation_config=generation_kwargs,
        )
        return [
            InternVLGeneratedResponse(
                text=response,
                token_count=len(self._tokenizer.encode(response, add_special_tokens=False)),
            )
            for response in responses
        ]

    def _do_glimpse(self, inputs, generation_config):
        raise NotImplementedError("Glimpse is not supported for InternVL2.5 baseline inference.")

    def prepare_batch_inputs(self, batched_querys, batched_img_paths, batched_bboxes):
        pixel_values_list = []
        num_patches_list = []
        questions = []
        for query, image_path in zip(batched_querys, batched_img_paths):
            pixel_values = self._load_image(image_path)
            pixel_values_list.append(pixel_values)
            num_patches_list.append(pixel_values.shape[0])
            questions.append(f"<image>\n{query}")

        pixel_values = torch.cat(pixel_values_list, dim=0).to(
            device=self._device,
            dtype=self._torch_dtype,
        )
        return {
            "pixel_values": pixel_values,
            "num_patches_list": num_patches_list,
            "questions": questions,
        }

    def batch_decode(self, generate_ids, **kwargs):
        return [item.text if isinstance(item, InternVLGeneratedResponse) else str(item) for item in generate_ids]


__all__ = [
    "InternVL2_5",
]
