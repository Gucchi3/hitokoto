# @file cross_entropy_z_loss.py
# @brief Cross entropy with z-loss, which keeps the logits from growing large

from torch    import nn
from torch.nn import functional as F
from typing   import Literal

import torch



class CrossEntropyZLoss(nn.Module):
    '''
    #* Hold cross entropy plus z_weight * logsumexp(logits)², the z-loss of PaLM, which keeps the logits from drifting to large values.
    #? Registered as "cross_entropy_z_loss" in LOSS_REGISTRY in loss.py. train_model.py of Supra-Mini-v6 uses z_weight 1e-4.
    #! The loss of each target includes the z-loss, so the train and validation loss (and the perplexity from it) are a little larger than the cross entropy alone.
    '''
    
    def __init__(self, z_weight: float, reduction: Literal["none"]) -> None:
        '''
        #* Keep the weight of the z-loss and the reduction.
        #! reduction must be "none" (one loss for each target), which build_loss_fn() in loss.py always gives.
        '''
        
        super().__init__()
        self.z_weight  = z_weight
        self.reduction = reduction
        return None
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        '''
        #* Return the loss of each target: the cross entropy plus z_weight times the square of logsumexp of its logits.
        #! logits has the shape (targets, classes), and targets has the shape (targets,). A target of -100 gets only the z-loss, and loss_label() in loss.py masks it out.
        '''
        
        return F.cross_entropy(logits, targets, reduction=self.reduction) + self.z_weight * logits.logsumexp(dim=-1) ** 2
