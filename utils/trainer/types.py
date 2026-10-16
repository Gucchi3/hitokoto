# @file types.py
# @brief Training types
# @author Kohei Yamaguchi, Claude

from dataclasses              import dataclass
from torch                    import nn
from torch.optim              import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from utils.trainer.optimization import LossKind



@dataclass(frozen=True)
class EpochMetrics:
    '''
    #* Hold the loss and the accuracy of one epoch.
    #! accuracy counts only the targets that are not skipped (-100).
    '''
    
    loss    : float
    accuracy: float
    samples : int


@dataclass(frozen=True)
class TrainingComponents:
    '''
    #* Group the loss and its kind, the optimizer, and the scheduler.
    '''
    
    loss_fn  : nn.Module
    loss_kind: LossKind
    optimizer: Optimizer
    scheduler: LRScheduler
