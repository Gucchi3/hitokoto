# @file loop.py
# @brief Training loop: train, validate, and save in turns until the last step

from dataclasses import asdict
from pathlib     import Path
from types       import SimpleNamespace

import torch
import torch.nn as nn

from utils.data    import DataBundle
from utils.runtime import print_validation, save_graph, save_history
from .evaluate     import evaluate_epoch
from .train        import train_steps
from .types        import TrainState, TrainingComponents



def loop_train(config: SimpleNamespace, device: torch.device, log_dir: Path, bundle: DataBundle, model: nn.Module, teacher: nn.Module | None, components: TrainingComponents, state: TrainState) -> None:
    '''
    #* Train until train.steps weight updates. Every train.eval_interval updates and at the last one, validate, and save the weights, the history, the graph, and the training state.
    #? Called from pipeline.py. Uses train_steps() in train.py and evaluate_epoch() in evaluate.py.
    #! state says where to start: the first step, or the place saved in train_state.pt for train.resume. A resumed run starts again from the last validation.
    #! best.pth keeps the weights with the lowest validation perplexity. Pass a teacher only for distillation.
    '''
    
    # Train and validate in turns until the last step
    while state.step < config.train.steps:
        # Train until the next multiple of eval_interval or the last step, then validate
        updates       = min(config.train.eval_interval - state.step % config.train.eval_interval, config.train.steps - state.step)
        train_metrics = train_steps(config, device, model, teacher, bundle.train_loader, bundle.train_sampler, components, state, updates)
        val_metrics   = evaluate_epoch(config, device, model, bundle.validation_loader, components, "Validation", low_precision=True)
        
        # Save the latest weights, and the best weights by validation perplexity
        torch.save(model.state_dict(), log_dir / "latest.pth")
        if val_metrics.perplexity < state.best_ppl:
            state.best_ppl = val_metrics.perplexity
            torch.save(model.state_dict(), log_dir / "best.pth")
        
        # Record, save, and show the results (tokens counts the targets of all the weight updates so far)
        record = {
            "step"         : state.step,
            "tokens"       : state.step * config.data.batch_size * config.train.accumulation_steps * config.data.max_length,
            "learning_rate": float(components.optimizer.param_groups[0]["lr"]),
            "train_loss"   : train_metrics.loss,
            "val_loss"     : val_metrics.loss,
            "val_ppl"      : val_metrics.perplexity,
            "val_acc"      : val_metrics.accuracy,
            "best_ppl"     : state.best_ppl,
        }
        state.history.append(record)
        save_history(log_dir, state.history)
        save_graph(state.history, log_dir)
        print_validation(state.history, config.train.steps)
        
        # Save the training state, so that train.resume can start again from here
        torch.save({"model": model.state_dict(), "optimizer": components.optimizer.state_dict(), "scheduler": components.scheduler.state_dict(), "state": asdict(state)}, log_dir / "train_state.pt")
    
    return None
