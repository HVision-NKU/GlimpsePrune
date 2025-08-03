from typing import List, Union, Optional
import math
from PIL import Image
import torch
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput, VideoInput
from transformers.processing_utils import Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.models.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor, Qwen2_5_VLProcessorKwargs


def get_ref_token_mask(normed_bboxes, grid_size):
    """
    Find the index of the bbox in the flatten grid.
    Args:
        bboxes: List[List[float]], N normed bbox, [0,1], xyxy
        grid: List[int], the grid size, [H, W]
    Return:
        BoolTensor [H, W]. The mask of the bbox in the grid.
    """
    H, W = grid_size
    mask = torch.zeros((H, W), dtype=torch.bool)
    for bbox in normed_bboxes:
        x1, y1, x2, y2 = bbox
        x1_grid = int(x1 * W)
        y1_grid = int(y1 * H)
        x2_grid = min(int(x2 * W), W - 1)
        y2_grid = min(int(y2 * H), H - 1)
        mask[y1_grid:y2_grid + 1, x1_grid:x2_grid + 1] = 1
    return mask



class Qwen2_5_VL_GP_Processor(Qwen2_5_VLProcessor):
    
    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        normed_bboxes: Optional[List[Optional[List[List[float]]]]] = None,
        ref_image_masks: Optional[List[Optional[Image.Image]]] = None,
        videos: VideoInput = None,
        **kwargs: Unpack[Qwen2_5_VLProcessorKwargs],
    ) -> BatchFeature:
        output_kwargs = self._merge_kwargs(
            Qwen2_5_VLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        ref_token_masks = None
        
        if images is not None:
            image_inputs = self.image_processor(images=images, videos=None, **output_kwargs["images_kwargs"])
            image_grid_thw = image_inputs["image_grid_thw"]
            
            if normed_bboxes is not None:
                assert len(normed_bboxes) == image_grid_thw.shape[0]
                ref_token_masks = []
                for i in range(len(normed_bboxes)):
                    if normed_bboxes[i] is None:
                        ref_token_masks.append(None)
                        continue
                    image_token_grid_hw = [image_grid_thw[i][1].item() // self.image_processor.merge_size, image_grid_thw[i][2].item() // self.image_processor.merge_size]
                    ref_token_masks.append(get_ref_token_mask(normed_bboxes[i], image_token_grid_hw))
            elif ref_image_masks is not None:
                assert len(ref_image_masks) == image_grid_thw.shape[0]
                ref_token_masks = []
                for i in range(len(ref_image_masks)):
                    if ref_image_masks[i] is None:
                        ref_token_masks.append(None)
                        continue
                    ref_grid_h = image_grid_thw[i][1].item() // self.image_processor.merge_size
                    ref_grid_w = image_grid_thw[i][2].item() // self.image_processor.merge_size
                    one_ref_token_mask = ref_image_masks[i].resize((ref_grid_w, ref_grid_h), Image.NEAREST)
                    one_ref_token_mask = torch.tensor(one_ref_token_mask > 127, dtype=torch.bool)
                    ref_token_masks.append(one_ref_token_mask)  
        else:
            image_inputs = {}
            image_grid_thw = None
            
        image_inputs.update({"ref_token_masks": ref_token_masks})
        
        if videos is not None:
            videos_inputs = self.image_processor(images=None, videos=videos, **output_kwargs["images_kwargs"])
            video_grid_thw = videos_inputs["video_grid_thw"]

            fps = output_kwargs["videos_kwargs"].pop("fps", 2.0)
            if isinstance(fps, (int, float)):
                second_per_grid_ts = [self.image_processor.temporal_patch_size / fps] * len(video_grid_thw)
            elif hasattr(fps, "__len__") and len(fps) == len(video_grid_thw):
                second_per_grid_ts = [self.image_processor.temporal_patch_size / tmp for tmp in fps]
            else:
                raise ValueError(
                    f"The length of fps ({len(fps) if hasattr(fps, '__len__') else fps}) must be equal to the length of video_grid_thw ({len(video_grid_thw)}) or fps should be a single number."
                )
            videos_inputs.update({"second_per_grid_ts": second_per_grid_ts})

        else:
            videos_inputs = {}
            video_grid_thw = None

        if not isinstance(text, list):
            text = [text]

        if image_grid_thw is not None:
            merge_length = self.image_processor.merge_size**2
            index = 0
            
            for i in range(len(text)):
                while self.image_token in text[i]:
                    text[i] = text[i].replace(
                        self.image_token,
                        "<|placeholder|>" * (image_grid_thw[index].prod() // merge_length),
                        1,
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)
                if normed_bboxes is not None:
                    assert index == i + 1, f"Assuming one image for one text, but got {index} images and {i + 1} texts"


        if video_grid_thw is not None:
            merge_length = self.image_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.video_token in text[i]:
                    text[i] = text[i].replace(
                        self.video_token,
                        "<|placeholder|>" * (video_grid_thw[index].prod() // merge_length),
                        1,
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.video_token)

        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        return BatchFeature(data={**text_inputs, **image_inputs, **videos_inputs})

__all__ = [
    "Qwen2_5_VL_GP_Processor",
]