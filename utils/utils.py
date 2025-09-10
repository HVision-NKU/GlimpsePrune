from typing import List
import re
import os
import numpy as np


def norm_bboxes(bboxes, height, width, bbox_type="xyxy"):
    assert bbox_type in ["xyxy", "xywh", "xyxy_norm1000"]
    normed_bboxes = []
    for bbox in bboxes:
        if bbox_type == "xyxy":
            x1, y1, x2, y2 = bbox
            normed_bboxes.append([x1 / width, y1 / height, x2 / width, y2 / height])
        elif bbox_type == "xyxy_norm1000":
            x1, y1, x2, y2 = bbox
            normed_bboxes.append([x1 / 1000.0, y1 / 1000.0, x2 / 1000.0, y2 / 1000.0])
        else:
            x1, y1, w, h = bbox
            normed_bboxes.append([x1 / width, y1 / height, (x1 + w) / width, (y1 + h) / height])
    return normed_bboxes


def extract_one_bbox_from_str(bbox_str: str) -> List[float]:
    bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
    match = re.search(bbox_pattern, bbox_str)
    if match:
        try:
            coords_str = match.groups()
            bbox_coords = [float(coord) for coord in coords_str]
            return bbox_coords
        except ValueError:
            return [0, 0, 0, 0] # Or raise an error
    else:
        return [0, 0, 0, 0]
    

def cal_paired_ious(bboxes_1: np.ndarray, bboxes_2: np.ndarray) -> np.ndarray:
    """
    Calculate IoU between a pair of bounding boxes.
    Args:
        bboxes_1 (np.ndarray): Array of shape (N, 4) for first set of boxes.
        bboxes_2 (np.ndarray): Array of shape (N, 4) for second set of boxes.
    Returns:
        np.ndarray: IoU of shape (N, ) where N is the number of boxes.
    """
    assert bboxes_1.shape == bboxes_2.shape, "Bounding boxes must have the same shape"
    
    x1 = np.maximum(bboxes_1[:, 0], bboxes_2[:, 0])
    y1 = np.maximum(bboxes_1[:, 1], bboxes_2[:, 1])
    x2 = np.minimum(bboxes_1[:, 2], bboxes_2[:, 2])
    y2 = np.minimum(bboxes_1[:, 3], bboxes_2[:, 3])
    
    intersection_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    
    area_1 = (bboxes_1[:, 2] - bboxes_1[:, 0]) * (bboxes_1[:, 3] - bboxes_1[:, 1])
    area_2 = (bboxes_2[:, 2] - bboxes_2[:, 0]) * (bboxes_2[:, 3] - bboxes_2[:, 1])
    
    union_area = area_1 + area_2 - intersection_area
    
    iou = intersection_area / (union_area + 1e-6) # Add small value to avoid division by zero
    return iou


def print_rank0(*args, **kwargs):
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    if local_rank == 0:
        print(*args, **kwargs)



    
    
