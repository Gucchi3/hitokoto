# @file bin2dataset.py
# @brief Dataset that cuts token windows out of a tokenized .bin file

from pathlib          import Path
from torch.utils.data import Dataset

import numpy as np
import torch



class Bin2dataset(Dataset):
    '''
    #* Cut windows of max_length + 1 tokens out of a .bin file of uint16 token IDs, and give input_ids, attention_mask, and targets.
    #? Used by build_pretrain_bundle() in pretrain.py. The .bin files are made by python -m tool.data.
    #! repeat sets how many times the windows are given: 1 gives each window once, 2 twice, and 0.5 the first half of them. An empty file (such as test.bin without test data) has no windows.
    #! The file is opened at the first __getitem__() in each DataLoader worker. Do not call __getitem__() before making the DataLoader, or each worker on Windows gets a copy of the whole file.
    '''
    
    ids: np.memmap | None
    
    def __init__(self, path: Path, max_length: int, repeat: float) -> None:
        '''
        #* Keep the path and the window length, and count the windows from the file size.
        '''
        
        super().__init__()
        self.path       = path
        self.max_length = max_length
        self.windows    = max((path.stat().st_size // 2 - 1) // max_length, 0)
        self.length     = round(self.windows * repeat)
        self.ids        = None
        return None
    
    def __len__(self) -> int:
        '''
        #* Return the number of windows times repeat. The windows move by max_length tokens, so each token is a target only once in each repeat.
        '''
        
        return self.length
    
    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        '''
        #* Return input_ids, attention_mask, and targets of one window. targets are input_ids moved by one token.
        '''
        
        # Open the file in this worker at the first call
        if self.ids is None:
            self.ids = np.memmap(self.path, dtype=np.uint16, mode="r")
        
        # Cut max_length + 1 tokens, and move the targets by one token (an index past the last window starts again from the first window)
        start  = (index % self.windows) * self.max_length
        window = torch.from_numpy(self.ids[start:start + self.max_length + 1].astype(np.int64))
        return {"input_ids": window[:-1], "attention_mask": torch.ones(self.max_length, dtype=torch.int64), "targets": window[1:]}
