# @file loss.py
# @brief Build a loss function, and compute the label loss and the distillation loss
# @author Kohei Yamaguchi, Claude

##########################################################
'''
Add a loss function
1. Make a loss class that inherits from nn.Module, and returns one loss for each target with reduction="none".
2. Add "name: (class, kind)" to LOSS_REGISTRY. kind is "softmax" (one class for each target) or "sigmoid" (a yes or no for each target).
3. Set config.train.loss.name to that name.
'''
##########################################################

from torch.nn import functional as F
from types    import SimpleNamespace
from typing   import Literal

import torch
import torch.nn as nn

from utils.exceptions import Error

LossKind = Literal["softmax", "sigmoid"]

IGNORE_INDEX = -100

LOSS_REGISTRY: dict[str, tuple[type[nn.Module], LossKind]] = {
    "cross_entropy"  : (nn.CrossEntropyLoss , "softmax"),
    "bce_with_logits": (nn.BCEWithLogitsLoss, "sigmoid"),
}



def build_loss_fn(config: SimpleNamespace) -> tuple[nn.Module, LossKind]:
    '''
    #* Make the loss function for config.train.loss.name, and return it with its kind.
    #? Called from utils/trainer/pipeline.py.
    #! The other keys in config.train.loss go to the loss class (such as label_smoothing). reduction is always "none", because loss_label() skips the targets of -100.
    '''
    
    # Check the name
    options = vars(config.train.loss).copy()
    name    = options.pop("name")
    if name not in LOSS_REGISTRY:
        raise Error("Unknown loss.", detail=f"train.loss.name: {name!r}\nAvailable: {', '.join(LOSS_REGISTRY)}", hint="Add the loss to LOSS_REGISTRY in utils/trainer/optimization/loss.py.", code=10)
    
    # Make the loss that returns one value for each target
    loss_class, kind     = LOSS_REGISTRY[name]
    options["reduction"] = "none"
    return loss_class(**options), kind


def loss_label(logits: torch.Tensor, targets: torch.Tensor, loss_fn: nn.Module, kind: LossKind) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    #* Return the mean loss against the labels, the number of correct targets, and the number of targets, skipping the targets of -100.
    #? Called from utils/trainer/epoch.py for training and evaluation.
    #! softmax: targets are class numbers, and logits have one more dimension (classes) than targets. sigmoid: targets are 0 or 1, and logits have the same shape as targets.
    #! No value is read on the CPU, so this works inside a CUDA graph.
    '''
    
    # Flatten to one row for each target
    flat_logits  = logits.flatten(0, -2).float() if kind == "softmax" else logits.flatten().float()
    flat_targets = targets.flatten()
    valid        = flat_targets != IGNORE_INDEX
    count        = valid.sum()
    
    # Mean loss over the targets that are not skipped (the skipped targets become 0 for BCE, and are masked out)
    loss_targets = flat_targets if kind == "softmax" else flat_targets.float().clamp(min=0)
    loss         = (loss_fn(flat_logits, loss_targets) * valid).sum() / count.clamp(min=1)
    
    # Count the correct targets: the largest class for softmax, and the side of 0 for sigmoid
    matched = flat_logits.argmax(dim=-1) == flat_targets if kind == "softmax" else (flat_logits > 0) == (flat_targets > 0.5)
    correct = (matched & valid).sum()
    return loss, correct, count


def loss_distill(logits: torch.Tensor, teacher_logits: torch.Tensor, targets: torch.Tensor, kind: LossKind, temperature: float) -> torch.Tensor:
    '''
    #* Return the mean distillation loss between the student and the teacher, softened by the temperature, skipping the targets of -100.
    #? Called from utils/trainer/epoch.py.
    #! softmax: KL divergence of the class probabilities. sigmoid: BCE against the teacher's yes probability of each target.
    #! The teacher logits must have the same shape as the student logits. Multiply the result by temperature² to keep the gradient size.
    '''
    
    # Flatten to one row for each target
    flat_student = (logits.flatten(0, -2) if kind == "softmax" else logits.flatten()).float() / temperature
    flat_teacher = (teacher_logits.flatten(0, -2) if kind == "softmax" else teacher_logits.flatten()).float() / temperature
    valid        = targets.flatten() != IGNORE_INDEX
    
    # Loss of each target, then the mean over the targets that are not skipped
    if kind == "softmax":
        soft_by_target = F.kl_div(flat_student.log_softmax(dim=-1), flat_teacher.log_softmax(dim=-1), reduction="none", log_target=True).sum(dim=-1)
    else:
        soft_by_target = F.binary_cross_entropy_with_logits(flat_student, flat_teacher.sigmoid(), reduction="none")
    return (soft_by_target * valid).sum() / valid.sum().clamp(min=1)
