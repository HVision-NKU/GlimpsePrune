import os
import torch
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
