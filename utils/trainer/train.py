# @file train.py
# @brief Train the model for a number of weight updates

from torch.nn.utils   import clip_grad_norm_
from torch.utils.data import DataLoader
from types            import SimpleNamespace

import torch
import torch.nn as nn

from utils.data                 import EpochSampler
from utils.runtime              import print_progress
from utils.trainer.optimization import loss_distill, loss_label
from .types                     import Metrics, TrainState, TrainingComponents

GRAPH_WARMUP_UPDATES = 3



def _train_forward_backward(config: SimpleNamespace, model: nn.Module, teacher: nn.Module | None, components: TrainingComponents, inputs: dict[str, torch.Tensor], totals: dict[str, torch.Tensor]) -> None:
    '''
    #* Run the forward and backward pass of one micro-batch, and add its label loss and accuracy to the totals.
    #? Called from train_steps(), both eagerly and inside the CUDA graph capture.
    #! Do not read tensor values on the CPU here. A CUDA graph cannot hold that.
    '''
    
    # Run the student, with BF16 AMP on CUDA (no cast cache, for the CUDA graph)
    device_type = inputs["input_ids"].device.type
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=device_type == "cuda", cache_enabled=False):
        logits = model(inputs["input_ids"], inputs["attention_mask"])
    
    # Compute the loss against the labels
    label_loss, correct, count = loss_label(logits, inputs["targets"], components.loss_fn, components.loss_kind)
    loss                       = label_loss
    
    # Mix in the loss against the teacher's softened outputs
    if teacher is not None:
        with torch.no_grad(), torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=device_type == "cuda", cache_enabled=False):
            teacher_logits = teacher(inputs.get("teacher_input_ids", inputs["input_ids"]), inputs.get("teacher_attention_mask", inputs["attention_mask"]))
        temperature = config.train.distillation.temperature
        alpha       = config.train.distillation.alpha
        soft_loss   = loss_distill(logits, teacher_logits, inputs["targets"], components.loss_kind, temperature)
        loss        = (1 - alpha) * label_loss + alpha * temperature**2 * soft_loss
    
    # Add the gradients, scaled so that one cycle of micro-batches gives their mean
    (loss / config.train.accumulation_steps).backward()
    
    # Add the label loss (weighted by the number of targets) and the accuracy to the totals
    totals["loss"].add_(label_loss.detach().double() * count)
    totals["correct"].add_(correct)
    totals["counted"].add_(count)
    return None


def _train_update(config: SimpleNamespace, model: nn.Module, components: TrainingComponents) -> None:
    '''
    #* Clip the gradients, update the weights, and set the gradients to zero.
    #? Called from train_steps(), both eagerly and inside the CUDA graph capture.
    #! Keep the gradient tensors (set_to_none=False). The CUDA graphs add to the same memory. Do not move the scheduler here, because a CUDA graph would replay the same learning rate.
    '''
    
    clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
    components.optimizer.step()
    components.optimizer.zero_grad(set_to_none=False)
    return None


def train_steps(config: SimpleNamespace, device: torch.device, model: nn.Module, teacher: nn.Module | None, train_loader: DataLoader, train_sampler: EpochSampler, components: TrainingComponents, state: TrainState, updates: int) -> Metrics:
    '''
    #* Train the model for the given number of weight updates, with gradient accumulation and clipping, going on to the next epoch when the train data ends. With a teacher, also learn the teacher's softened outputs (distillation).
    #? Called from loop.py between validations. Shows the progress of the whole training and the running loss in one line, which goes away at the end.
    #! The train data starts at micro-batch state.batch of epoch state.epoch (through train_sampler), and state.step, state.epoch, and state.batch move with the training.
    #! Each batch is a dict of input_ids, attention_mask, and targets (-100 to skip). Add teacher_input_ids and teacher_attention_mask when the teacher uses another tokenizer.
    #! The shapes of the logits and the targets follow the loss kind (see loss_label() in optimization/loss.py). The teacher must give the same shape as the student.
    #! With train.cuda_graph on CUDA, the first updates of each call run eagerly. After that, one micro-batch and one update each run as a CUDA graph. Batches of another shape run eagerly.
    '''
    
    # Set training mode, and reset the totals on the device
    model.train()
    accumulation_steps = config.train.accumulation_steps
    use_graph          = config.train.cuda_graph and device.type == "cuda"
    side_stream        = torch.cuda.Stream() if use_graph else None
    totals             = {"loss": torch.zeros((), device=device, dtype=torch.float64), "correct": torch.zeros((), device=device, dtype=torch.int64), "counted": torch.zeros((), device=device, dtype=torch.int64)}
    static             = dict[str, torch.Tensor]()
    forward_graph      = None
    update_graph       = None
    samples            = 0
    micro_batches      = 0
    done               = 0
    
    # Run on a side stream for the CUDA graph, as PyTorch asks for the warmup and the capture
    if side_stream is not None:
        side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream), print_progress("train") as progress:
        task = progress.add_task("Train", total=config.train.steps, completed=state.step, loss="-")
        while done < updates:
            # Start the train data at the place in the epoch where the training is
            train_sampler.set_position(state.epoch, state.batch * config.data.batch_size)
            for batch in train_loader:
                micro_batches += 1
                state.batch   += 1
                
                # Copy the batch into the fixed buffers of the CUDA graph when the shape matches, or else move it to the device
                if use_graph and not static:
                    static = {key: value.to(device) for key, value in batch.items()}
                same_shape = use_graph and all(static[key].shape == value.shape for key, value in batch.items())
                if same_shape:
                    for key, value in batch.items():
                        static[key].copy_(value, non_blocking=True)
                inputs     = static if same_shape else {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                
                # Capture the CUDA graphs at the start of a cycle, after the warmup updates
                if same_shape and forward_graph is None and micro_batches > GRAPH_WARMUP_UPDATES * accumulation_steps and (micro_batches - 1) % accumulation_steps == 0:
                    forward_graph = torch.cuda.CUDAGraph()
                    update_graph  = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(forward_graph):
                        _train_forward_backward(config, model, teacher, components, static, totals)
                    with torch.cuda.graph(update_graph, pool=forward_graph.pool()):
                        _train_update(config, model, components)
                
                # Run the micro-batch, by the CUDA graph or eagerly
                if forward_graph is not None and same_shape:
                    forward_graph.replay()
                else:
                    _train_forward_backward(config, model, teacher, components, inputs, totals)
                samples += inputs["input_ids"].shape[0]
                
                # Update the weights at the end of each cycle, then move the learning rate one step
                if micro_batches % accumulation_steps == 0 and update_graph is not None:
                    update_graph.replay()
                elif micro_batches % accumulation_steps == 0:
                    _train_update(config, model, components)
                if micro_batches % accumulation_steps == 0:
                    components.scheduler.step()
                    done       += 1
                    state.step += 1
                    
                    # Show the running loss every log_interval updates (this waits for the GPU)
                    progress.update(task, advance=1)
                    if state.step % config.train.log_interval == 0:
                        progress.update(task, loss=f"{(totals['loss'] / totals['counted'].clamp(min=1)).item():.4f}")
                if done == updates:
                    break
            else:
                # Go on to the next epoch when the train data ends
                state.epoch += 1
                state.batch  = 0
    if side_stream is not None:
        torch.cuda.current_stream().wait_stream(side_stream)
    
    # Read the totals once at the end
    counted = max(int(totals["counted"].item()), 1)
    return Metrics(totals["loss"].item() / counted, totals["correct"].item() / counted, samples)
