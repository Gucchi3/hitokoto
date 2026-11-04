# @file types.py
# @brief Training types

from dataclasses              import dataclass
from torch                    import nn
from torch.optim              import Optimizer
from torch.optim.lr_scheduler import LRScheduler

import torch

from utils.trainer.optimization import LossKind



@dataclass(frozen=True)
class Metrics:
    '''
    #* Hold the loss and the accuracy of the batches between two validations, or of one pass over a loader.
    #! accuracy counts only the targets that are not skipped (-100).
    '''
    
    loss    : float
    accuracy: float
    samples : int
    
    @property
    def perplexity(self) -> float:
        '''
        #* Return the perplexity, exp(loss). It is inf, not an error, when the loss is too large (such as after the training breaks).
        '''
        
        return torch.tensor(self.loss, dtype=torch.float64).exp().item()


@dataclass(frozen=True)
class TrainingComponents:
    '''
    #* Group the loss and its kind, the optimizer, and the scheduler.
    '''
    
    loss_fn  : nn.Module
    loss_kind: LossKind
    optimizer: Optimizer
    scheduler: LRScheduler


@dataclass
class TrainState:
    '''
    #* Hold where the training is: the number of weight updates, the place in the train data, the best (lowest) validation perplexity, and the history.
    #? Made by pipeline_train() in pipeline.py, from the start or from train.resume. train_steps() and loop_train() move it, and loop_train() saves it to train_state.pt.
    #! batch is the number of micro-batches already used in the epoch. It is not frozen, because the training changes it.
    '''
    
    step     : int
    epoch    : int
    batch    : int
    best_ppl : float
    history  : list[dict[str, int | float]]
