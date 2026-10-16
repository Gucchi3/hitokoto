# @file optimizer.py
# @brief Build an optimizer
# @author Kohei Yamaguchi, Claude

##########################################################
'''
Add an optimizer
1. Write build_xxx_optimizer(config, model) to return an Optimizer.
2. Add "name: build_xxx_optimizer" to OPTIMIZER_REGISTRY.
3. Set config.train.optimizer.name to that name.
'''
##########################################################

from collections.abc import Callable
from torch           import nn
from torch.optim     import AdamW, Optimizer
from types           import SimpleNamespace

import torch

from utils.exceptions import Error

OptimizerBuilder = Callable[[SimpleNamespace, nn.Module], Optimizer]



def build_adamw_optimizer(config: SimpleNamespace, model: nn.Module) -> Optimizer:
    '''
    #* Make AdamW with weight decay only on the weight matrices (not on biases, norm weights, or embeddings).
    #! With the CUDA graph, the learning rate of each group is a tensor on the device, so the scheduler can change it between graph replays.
    '''
    
    # Split the parameters: weight decay on the weights of 2 or more dimensions, outside the embeddings
    embedding_ids = {id(parameter) for module in model.modules() if isinstance(module, nn.Embedding) for parameter in module.parameters()}
    parameters    = [parameter for parameter in model.parameters() if parameter.requires_grad]
    decay         = [parameter for parameter in parameters if parameter.ndim >= 2 and id(parameter) not in embedding_ids]
    no_decay      = [parameter for parameter in parameters if parameter.ndim < 2 or id(parameter) in embedding_ids]
    
    # Make AdamW, fused on CUDA, and capturable for the CUDA graph
    device        = parameters[0].device
    capturable    = config.train.cuda_graph and device.type == "cuda"
    learning_rate = config.train.optimizer.learning_rate
    decay_lr      = torch.tensor(learning_rate, device=device) if capturable else learning_rate
    no_decay_lr   = torch.tensor(learning_rate, device=device) if capturable else learning_rate
    groups        = [{"params": decay, "weight_decay": config.train.optimizer.weight_decay, "lr": decay_lr}, {"params": no_decay, "weight_decay": 0.0, "lr": no_decay_lr}]
    return AdamW(groups, fused=device.type == "cuda", capturable=capturable)


OPTIMIZER_REGISTRY: dict[str, OptimizerBuilder] = {
    "adamw": build_adamw_optimizer,
}


def build_optimizer(config: SimpleNamespace, model: nn.Module) -> Optimizer:
    '''
    #* Make the optimizer for config.train.optimizer.name.
    #? Called from utils/trainer/pipeline.py.
    '''
    
    name = config.train.optimizer.name
    if name not in OPTIMIZER_REGISTRY:
        raise Error("Unknown optimizer.", detail=f"train.optimizer.name: {name!r}\nAvailable: {', '.join(OPTIMIZER_REGISTRY)}", hint="Add the optimizer to OPTIMIZER_REGISTRY in utils/trainer/optimization/optimizer.py.", code=11)
    return OPTIMIZER_REGISTRY[name](config, model)
