# @file setup.py
# @brief Run setup
# @author Kohei Yamaguchi, Claude

from datetime import datetime
from pathlib  import Path

import numpy as np
import random
import torch

from utils.exceptions import Error



def setup_seed(seed: int) -> None:
    '''
    #* Set the seeds of Python, NumPy, and PyTorch.
    '''
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return None


def setup_device(device: str) -> torch.device:
    '''
    #* Choose a CPU or CUDA device from run.device.
    #! run.device is auto, cpu, or cuda. auto chooses CUDA when PyTorch can use it.
    '''
    
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device == "cuda":
        raise Error("CUDA is not available.", detail="run.device is 'cuda', but PyTorch cannot find a CUDA device.", hint="Check CUDA, or set run.device to 'auto' or 'cpu'.", code=3)
    raise Error("Invalid device.", detail=f"run.device: {device!r}", hint="Set run.device to 'auto', 'cuda', or 'cpu'.", code=4)


def setup_log_dir(log_dir: str | Path) -> Path:
    '''
    #* Make a new folder named with the date and time inside log_dir, and return it.
    #! When a folder with the same time exists, add a number such as _01.
    '''
    
    root  = Path(log_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        root.mkdir(parents=True, exist_ok=True)
        index = 0
        while True:
            candidate = root / (stamp if index == 0 else f"{stamp}_{index:02d}")
            try:
                candidate.mkdir()
                return candidate
            except FileExistsError:
                index += 1
    except OSError as error:
        raise Error("Could not make the log folder.", detail=f"{root}: {error}", hint="Check run.log_dir and the write permission.", code=7)
