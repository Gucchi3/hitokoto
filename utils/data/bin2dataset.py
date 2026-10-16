# @file bin2dataset.py
# @brief Dataset that cuts token windows out of a tokenized .bin file (empty for now)
# @author Kohei Yamaguchi, Claude

from torch.utils.data import Dataset

import torch



class Bin2dataset(Dataset):
    '''
    #* Cut windows of max_length + 1 tokens out of a .bin file of token IDs, and give input_ids, attention_mask, and targets.
    #? Used by build_pretrain_bundle() in pretrain.py and build_sft_bundle() in sft.py. The .bin files are made by a script in tool/.
    #! Not written yet. targets are input_ids moved by one token. For SFT, targets outside the bot replies (from <bot> to <eos>) become -100, and the loss mask moves by one token together with targets.
    '''
    
    def __init__(self) -> None:
        '''
        #* Open the .bin file with np.memmap.
        '''
        
        super().__init__()
        return None
    
    def __len__(self) -> int:
        '''
        #* Return the number of windows that do not overlap.
        '''
        
        raise NotImplementedError("Bin2dataset.__len__() is not written yet (utils/data/bin2dataset.py).")
    
    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        '''
        #* Return input_ids, attention_mask, and targets of one window.
        '''
        
        raise NotImplementedError("Bin2dataset.__getitem__() is not written yet (utils/data/bin2dataset.py).")
