# @file scheduler.py
# @brief Build a scheduler
# @author Kohei Yamaguchi, Claude

##########################################################
'''
Add a scheduler
1. Write build_xxx_scheduler(config, optimizer, total_steps) to return an LRScheduler. It moves one step at each weight update.
2. Add "name: build_xxx_scheduler" to SCHEDULER_REGISTRY.
3. Set config.train.scheduler.name to that name.
'''
##########################################################

from collections.abc          import Callable
from torch.optim              import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler
from types                    import SimpleNamespace

import math

from utils.exceptions import Error

SchedulerBuilder = Callable[[SimpleNamespace, Optimizer, int], LRScheduler]



def build_cosine_annealing_scheduler(config: SimpleNamespace, optimizer: Optimizer, total_steps: int) -> LRScheduler:
    '''
    #* Make a scheduler that raises the learning rate in a straight line for warmup_steps, then lowers it along a cosine to the minimum at the last step.
    #! The learning rate in the config is the top value after the warmup. One step is one weight update (not one batch, with gradient accumulation).
    '''
    
    # Read the settings
    warmup_steps  = config.train.scheduler.warmup_steps
    minimum_ratio = config.train.scheduler.minimum_learning_rate / config.train.optimizer.learning_rate
    decay_steps   = max(total_steps - warmup_steps, 1)
    
    # Give the ratio of the learning rate to the top value at each step
    return LambdaLR(optimizer, lambda step: (step + 1) / warmup_steps if step < warmup_steps else minimum_ratio + (1 - minimum_ratio) * 0.5 * (1 + math.cos(math.pi * min((step - warmup_steps) / decay_steps, 1.0))))


SCHEDULER_REGISTRY: dict[str, SchedulerBuilder] = {
    "cosine_annealing": build_cosine_annealing_scheduler,
}


def build_scheduler(config: SimpleNamespace, optimizer: Optimizer, total_steps: int) -> LRScheduler:
    '''
    #* Make the scheduler for config.train.scheduler.name.
    #? Called from utils/trainer/pipeline.py, with the number of weight updates in the whole training.
    '''
    
    name = config.train.scheduler.name
    if name not in SCHEDULER_REGISTRY:
        raise Error("Unknown scheduler.", detail=f"train.scheduler.name: {name!r}\nAvailable: {', '.join(SCHEDULER_REGISTRY)}", hint="Add the scheduler to SCHEDULER_REGISTRY in utils/trainer/optimization/scheduler.py.", code=12)
    return SCHEDULER_REGISTRY[name](config, optimizer, total_steps)
