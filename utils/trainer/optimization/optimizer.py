# @file optimizer.py
# @brief Build an optimizer

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
from typing          import Any

import torch

from utils.exceptions import Error
from .muon_adamw      import MuonAdamW

OptimizerBuilder = Callable[[SimpleNamespace, nn.Module], Optimizer]



def build_adamw_optimizer(config: SimpleNamespace, model: nn.Module) -> Optimizer:
    '''
    #* Make AdamW with weight decay only on the weight matrices (not on biases, norm weights, or embeddings), and with embedding_lr_scale times the learning rate on the embeddings.
    #! With the CUDA graph, the learning rate of each group is a tensor on the device, so the scheduler can change it between graph replays.
    #! betas are the decay rates of the two moving averages in AdamW, such as [0.9, 0.999] (the PyTorch default). An embedding_lr_scale of 1.0 gives the embeddings the same learning rate as the rest.
    '''
    
    # Split the parameters into the weight matrices (with weight decay), the embeddings, and the rest (biases and norm weights)
    embedding_ids = {id(parameter) for module in model.modules() if isinstance(module, nn.Embedding) for parameter in module.parameters()}
    parameters    = [parameter for parameter in model.parameters() if parameter.requires_grad]
    decay         = [parameter for parameter in parameters if parameter.ndim >= 2 and id(parameter) not in embedding_ids]
    embeddings    = [parameter for parameter in parameters if id(parameter) in embedding_ids]
    no_decay      = [parameter for parameter in parameters if parameter.ndim < 2 and id(parameter) not in embedding_ids]
    
    # Make AdamW, fused on CUDA, and capturable for the CUDA graph
    settings      = config.train.optimizer
    device        = parameters[0].device
    capturable    = config.train.cuda_graph and device.type == "cuda"
    learning_rate = settings.learning_rate
    decay_lr      = torch.tensor(learning_rate, device=device) if capturable else learning_rate
    embedding_lr  = torch.tensor(learning_rate * settings.embedding_lr_scale, device=device) if capturable else learning_rate * settings.embedding_lr_scale
    no_decay_lr   = torch.tensor(learning_rate, device=device) if capturable else learning_rate
    groups        = [{"params": decay, "weight_decay": settings.weight_decay, "lr": decay_lr}, {"params": embeddings, "weight_decay": 0.0, "lr": embedding_lr}, {"params": no_decay, "weight_decay": 0.0, "lr": no_decay_lr}]
    return AdamW(groups, betas=(settings.betas[0], settings.betas[1]), fused=device.type == "cuda", capturable=capturable)


def build_muon_optimizer(config: SimpleNamespace, model: nn.Module) -> Optimizer:
    '''
    #* Make the stable Muon: Newton-Schulz orthogonalization, Nesterov momentum of muon_momentum, and decoupled weight decay on the weight matrices, with AdamW for the embeddings and the rest.
    #? The same update as SingleDeviceMuonWithAuxAdam in Keller Jordan's Muon, which most Muon models on the Tiny-ML Leaderboard use. The groups come from _optimizer_muon_groups().
    #! learning_rate, betas, and embedding_lr_scale are for the AdamW groups. muon_learning_rate (such as 0.02) and weight_decay (such as 0.01) are for the weight matrices.
    '''
    
    muon = {"orthogonalization": "newton_schulz", "momentum_start": config.train.optimizer.muon_momentum, "momentum_warmup_steps": 0, "normuon": False, "beta2": 0.0, "cautious": False}
    return MuonAdamW(_optimizer_muon_groups(config, model, muon))


def build_normuon_optimizer(config: SimpleNamespace, model: nn.Module) -> Optimizer:
    '''
    #* Make the latest Muon: Polar Express orthogonalization, NorMuon, cautious weight decay, and momentum that warms up from muon_momentum_start to muon_momentum, with AdamW for the embeddings and the rest.
    #? The Muon of modded-nanogpt (the NanoGPT speedrun) from late 2025 to 2026, without its parts for many GPUs. Polar Express: arXiv 2505.16932. NorMuon: arXiv 2510.05491. Cautious weight decay: arXiv 2510.12402.
    #! The keys of build_muon_optimizer() are used in the same way. normuon_beta2 is the decay rate of the moving mean square of each row. muon_momentum_warmup_steps of 0 keeps the momentum at muon_momentum.
    '''
    
    settings = config.train.optimizer
    muon     = {"orthogonalization": "polar_express", "momentum_start": settings.muon_momentum_start, "momentum_warmup_steps": settings.muon_momentum_warmup_steps, "normuon": True, "beta2": settings.normuon_beta2, "cautious": True}
    return MuonAdamW(_optimizer_muon_groups(config, model, muon))


def _optimizer_muon_groups(config: SimpleNamespace, model: nn.Module, muon: dict[str, Any]) -> list[dict[str, Any]]:
    '''
    #* Return the groups of MuonAdamW: Muon for the weight matrices, AdamW for the embeddings (embedding_lr_scale times learning_rate), and AdamW for the rest (biases and norm weights).
    #? Used by build_muon_optimizer() and build_normuon_optimizer(). muon holds the settings that differ between the two.
    #! The weight matrices are the parameters of 2 or more dimensions outside the embeddings, so an output layer that is not tied to the embedding gets Muon too. The AdamW groups get no weight decay.
    #! With the CUDA graph, each learning rate is a tensor on the device, so the scheduler can change it between graph replays.
    '''
    
    # Split the parameters into the weight matrices, the embeddings, and the rest
    embedding_ids = {id(parameter) for module in model.modules() if isinstance(module, nn.Embedding) for parameter in module.parameters()}
    parameters    = [parameter for parameter in model.parameters() if parameter.requires_grad]
    matrices      = [parameter for parameter in parameters if parameter.ndim >= 2 and id(parameter) not in embedding_ids]
    embeddings    = [parameter for parameter in parameters if id(parameter) in embedding_ids]
    others        = [parameter for parameter in parameters if parameter.ndim < 2 and id(parameter) not in embedding_ids]
    
    # Make the learning rate of each group, as a tensor on the device for the CUDA graph
    settings   = config.train.optimizer
    device     = parameters[0].device
    capturable = config.train.cuda_graph and device.type == "cuda"
    rates      = [torch.tensor(rate, device=device) if capturable else rate for rate in (settings.muon_learning_rate, settings.learning_rate * settings.embedding_lr_scale, settings.learning_rate)]
    
    # Make the groups (the weight matrices first, as in build_adamw_optimizer(), so the history records the Muon learning rate)
    adamw = {"kind": "adamw", "betas": (settings.betas[0], settings.betas[1]), "eps": 1e-8, "weight_decay": 0.0}
    return [{"params": matrices, "kind": "muon", "lr": rates[0], "weight_decay": settings.weight_decay, "momentum": settings.muon_momentum, **muon}, {"params": embeddings, "lr": rates[1], **adamw}, {"params": others, "lr": rates[2], **adamw}]


OPTIMIZER_REGISTRY: dict[str, OptimizerBuilder] = {
    "adamw"  : build_adamw_optimizer,
    "muon"   : build_muon_optimizer,
    "normuon": build_normuon_optimizer,
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
