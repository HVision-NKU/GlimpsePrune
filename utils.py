from typing import List
import re
import os
import numpy as np
import torch
from openai import OpenAI
from huggingface_hub import snapshot_download


def download_model_from_hf(repo_id: str, local_dir: str = None, revision: str = "main", force_download: bool = False):
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    local_path = None
    if local_rank == 0:
        local_path = snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            revision=revision,
            force_download=force_download,
        )
    # sychronize the download across all processes
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    
    if local_path is None:
        local_path = snapshot_download(repo_id=repo_id, local_dir=local_dir, revision=revision)
    return local_path
    

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


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]
    
    
class LLMClient(metaclass=SingletonMeta):
    SYS_PROMPT = """
You are responsible for proofreading the answers, you need to give a score to the model's answer by referring to the standard answer, based on the given question. The full score is 1 point and the minimum score is 0 points. Please output the score in the form "score: <score>". The evaluation criteria require that the closer the model's answer is to the standard answer, the higher the score.
"""

    PROMPT = """
question: {}
standard answer: {}
model's answer: {}
"""
    def __init__(self, base_url, api_key, model_name, timeout=20.0):
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model_name = model_name
        self._check_init()
        
    
    def _check_init(self):
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": "Hello!"}
            ]
            )
        print(completion.choices[0].message)
        
    def _extract_score_from_str(self, score_str: str) -> float:
        lower_str = score_str.lower()
        if 'score' not in lower_str:
            return 0.0
        res = re.findall(r'score: ([\d\.]+)', lower_str)
        if len(res) != 1:
            return 0.0
        res = float(res[0])
        if res > 1.0:
            res = 1
        if res < 0.0:
            res = 0
        return res
    
    def score(self, query_texts, completion_texts, answer_texts):
        """
        Scores the completions based on the query and answer texts.
        """
        scores = []
        for query, answer, completion in zip(query_texts, answer_texts, completion_texts):
            messages = [
                {"role": "system", "content": self.SYS_PROMPT},
                {"role": "user", "content": self.PROMPT.format(query, answer, completion)},
            ]
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.0,
                )
                score_str = completion.choices[0].message.content
                score = self._extract_score_from_str(score_str)
            except Exception as e:
                print(f"Error during scoring: {e}")
                score = 0.0
            scores.append(score)
        return scores