# @file muon_adamw.py
# @brief MuonAdamW: Muon for the weight matrices and AdamW for the other parameters, in one optimizer

from collections.abc import Callable
from typing          import Any, Literal

import torch

Orthogonalization = Literal["newton_schulz", "polar_express"]
Coefficients      = tuple[tuple[float, float, float], ...]

# The (a, b, c) of each step of X <- a X + b (X Xᵀ) X + c (X Xᵀ)² X
NEWTON_SCHULZ_COEFFICIENTS: Coefficients = ((3.4445, -4.7750, 2.0315),) * 5
POLAR_EXPRESS_COEFFICIENTS: Coefficients = (
    (8.156554524902461 , -22.48329292557795 , 15.878769915207462 ),
    (4.042929935166739 , -2.808917465908714 , 0.5000178451051316 ),
    (3.8916678022926607, -2.772484153217685 , 0.5060648178503393 ),
    (3.285753657755655 , -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
)

# The coefficients of each orthogonalization, and the safety margin of the first scaling (it keeps the largest singular value below 1)
ORTHOGONALIZATIONS: dict[Orthogonalization, tuple[Coefficients, float]] = {
    "newton_schulz": (NEWTON_SCHULZ_COEFFICIENTS, 0.0 ),
    "polar_express": (POLAR_EXPRESS_COEFFICIENTS, 0.02),
}



class MuonAdamW(torch.optim.Optimizer):
    '''
    #* Hold one optimizer that updates the groups of kind "muon" with Muon (momentum, then the nearest orthogonal matrix of the update) and the groups of kind "adamw" with AdamW.
    #? Made by build_muon_optimizer() (the stable Muon of Keller Jordan) and build_normuon_optimizer() (the latest Muon of modded-nanogpt) in optimizer.py.
    #! Muon is only for the weight matrices of hidden layers. Give the embeddings, an output layer, LoRA matrices, and the 1-D parameters (norm weights, biases) to the "adamw" groups.
    #! step() reads no value on the CPU and counts the steps in tensors on the device, so it works inside a CUDA graph. lr can be a float or a tensor on the device that the scheduler changes.
    '''
    
    def __init__(self, param_groups: list[dict[str, Any]]) -> None:
        '''
        #* Keep the parameter groups.
        #! A "muon" group needs lr, weight_decay, momentum, momentum_start, momentum_warmup_steps, orthogonalization, normuon, beta2, and cautious. An "adamw" group needs lr, betas, eps, and weight_decay.
        '''
        
        super().__init__(param_groups, {})
        return None
    
    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> Any:
        '''
        #* Update all the parameters once, and return the loss of closure when one is given (or else None).
        #! Muon: Nesterov momentum, orthogonalization, NorMuon (when normuon), scaling by sqrt(max(1, rows / columns)), and decoupled weight decay (only where the update and the weight have the same sign, when cautious).
        #! AdamW: the moving averages of the gradient and its square with bias correction, and decoupled weight decay.
        #! The return type is Any, so that it fits both overloads of Optimizer.step() (None without closure, float with it).
        '''
        
        # Run the closure for the loss, as the other PyTorch optimizers do
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        # Update each parameter that has a gradient
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.state[parameter]
                grad  = parameter.grad
                if group["kind"] == "muon":
                    # Make the state at the first step: the momentum, the step count, and for NorMuon the moving mean square of each row (or column, for a wide matrix)
                    rows, columns = grad.shape[0], grad[0].numel()
                    if not state:
                        state["momentum"] = torch.zeros_like(parameter)
                        state["step"]     = torch.zeros((), device=parameter.device)
                        if group["normuon"]:
                            state["square"] = torch.zeros((rows, 1) if rows >= columns else (1, columns), device=parameter.device)
                    
                    # Move the momentum (it warms up from momentum_start to momentum in momentum_warmup_steps), and take the Nesterov step
                    state["step"] += 1
                    momentum       = group["momentum"] if group["momentum_warmup_steps"] == 0 else group["momentum_start"] + (group["momentum"] - group["momentum_start"]) * (state["step"] / group["momentum_warmup_steps"]).clamp(max=1.0)
                    state["momentum"].lerp_(grad, 1 - momentum)
                    update         = grad.lerp(state["momentum"], momentum).reshape(rows, columns)
                    
                    # Replace the update with the nearest orthogonal matrix, in bfloat16 with the short side first
                    coefficients, safety = ORTHOGONALIZATIONS[group["orthogonalization"]]
                    x                    = update.bfloat16().mT if rows > columns else update.bfloat16()
                    x                    = x / (x.norm() * (1 + safety) + 1e-7)
                    for a, b, c in coefficients:
                        gram = x @ x.mT
                        x    = a * x + (b * gram + c * gram @ gram) @ x
                    update               = (x.mT if rows > columns else x).float()
                    
                    # NorMuon: divide each row (or column) by the root of its moving mean square, and keep the size of the whole update
                    if group["normuon"]:
                        size   = update.norm()
                        state["square"].lerp_(update.square().mean(dim=1 if rows >= columns else 0, keepdim=True), 1 - group["beta2"])
                        update = update * state["square"].clamp_min(1e-10).rsqrt()
                        update = update * (size / update.norm().clamp_min(1e-10))
                    
                    # Scale the update for the shape, and make the weight decay (only where the update and the weight have the same sign, when cautious)
                    update = update.reshape(parameter.shape) * max(1.0, rows / columns) ** 0.5
                    decay  = group["weight_decay"] * parameter * ((update * parameter) >= 0) if group["cautious"] else group["weight_decay"] * parameter
                else:
                    # Make the state at the first step: the moving averages of the gradient and its square, and the step count
                    if not state:
                        state["average"] = torch.zeros_like(parameter)
                        state["square"]  = torch.zeros_like(parameter)
                        state["step"]    = torch.zeros((), device=parameter.device)
                    
                    # Move the moving averages, divide them with bias correction, and make the weight decay
                    beta1, beta2   = group["betas"]
                    state["step"] += 1
                    state["average"].lerp_(grad, 1 - beta1)
                    state["square"].lerp_(grad.square(), 1 - beta2)
                    update         = (state["average"] / (1 - beta1 ** state["step"])) / ((state["square"] / (1 - beta2 ** state["step"])).sqrt() + group["eps"])
                    decay          = group["weight_decay"] * parameter
                
                # Change the parameter by the learning rate
                parameter.sub_(group["lr"] * (update + decay))
        return loss
