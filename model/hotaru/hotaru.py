# @file hotaru.py
# @brief Hotaru, a small GPT for edge devices (empty for now)
# @author Kohei Yamaguchi, Claude

from torch import nn

import torch



class Hotaru(nn.Module):
    '''
    #* Hold Hotaru, a small GPT (a decoder-only language model) for edge devices. It has no layers yet.
    #? Made by build_model() in model/build.py, with model.name = "hotaru". Its tokenizer goes in model/weight/hotaru/.
    #! Write the layers in __init__() and the computation in forward() before training.
    '''
    
    def __init__(self) -> None:
        '''
        #* Make the layers.
        '''
        
        super().__init__()
        return None
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        '''
        #* Return the logits for the inputs.
        #? Called from utils/trainer/epoch.py.
        #! Return logits of shape (*targets.shape, classes): (batch, tokens, vocab) for next-token prediction, or (batch, classes) for one label per text.
        #! For the CUDA graph, do not read tensor values on the CPU here (.item(), if on a tensor, boolean indexing).
        '''
        
        raise NotImplementedError("Hotaru.forward() is not written yet (model/hotaru/hotaru.py).")
