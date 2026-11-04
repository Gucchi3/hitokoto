# @file epoch.py
# @brief Evaluate the model on all the batches of a loader

from torch.utils.data import DataLoader
from types            import SimpleNamespace

import torch
import torch.nn as nn

from utils.runtime              import print_progress
from utils.trainer.optimization import loss_label
from .types                     import Metrics, TrainingComponents



def evaluate_epoch(config: SimpleNamespace, device: torch.device, model: nn.Module, loader: DataLoader, components: TrainingComponents, description: str, low_precision: bool) -> Metrics:
    '''
    #* Find the loss and the accuracy of the model on the loader, without training.
    #? Called from loop.py for validation (low_precision=True, fast BF16), and from pipeline.py for the test (low_precision=False, exact float32).
    #! The batches have the same keys as in train_steps() in train.py. The loss is the mean over the targets that are not skipped.
    '''
    
    # Set evaluation mode, and reset the totals on the device
    model.eval()
    amp_enabled = low_precision and device.type == "cuda"
    loss_total  = torch.zeros((), device=device, dtype=torch.float64)
    correct     = torch.zeros((), device=device, dtype=torch.int64  )
    counted     = torch.zeros((), device=device, dtype=torch.int64  )
    samples     = 0
    
    # Evaluate each batch without gradients
    with torch.no_grad(), print_progress() as progress:
        task = progress.add_task(description, total=len(loader), loss="-")
        for step, batch in enumerate(loader, start=1):
            # Run the model, with BF16 AMP on CUDA when low_precision is on
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                logits = model(input_ids, batch["attention_mask"].to(device, non_blocking=True))
            
            # Compute the loss against the labels
            loss, batch_correct, count = loss_label(logits, batch["targets"].to(device, non_blocking=True), components.loss_fn, components.loss_kind)
            
            # Add the batch results to the totals (the loss weighted by the number of targets)
            loss_total.add_(loss.double() * count)
            correct.add_(batch_correct)
            counted.add_(count)
            samples += input_ids.shape[0]
            
            # Show the running loss every log_interval steps (this waits for the GPU)
            progress.update(task, advance=1)
            if step % config.train.log_interval == 0:
                progress.update(task, loss=f"{(loss_total / counted.clamp(min=1)).item():.4f}")
    
    # Read the totals once at the end
    total = max(int(counted.item()), 1)
    return Metrics(loss_total.item() / total, correct.item() / total, samples)
