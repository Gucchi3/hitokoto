# @file loop.py
# @brief Training loop
# @author Kohei Yamaguchi, Claude

from pathlib import Path
from types   import SimpleNamespace

import torch
import torch.nn as nn

from utils.data    import DataBundle
from utils.runtime import print_epoch, save_graph, save_history
from .epoch        import evaluate_epoch, train_epoch
from .types        import TrainingComponents



def train_loop(config: SimpleNamespace, device: torch.device, log_dir: Path, bundle: DataBundle, model: nn.Module, teacher: nn.Module | None, components: TrainingComponents) -> None:
    '''
    #* Train and validate for each epoch, and save the weights, the history, and the graph.
    #? Called from pipeline.py. Uses train_epoch() and evaluate_epoch() in epoch.py.
    #! Pass a teacher only for distillation.
    '''
    
    # Set the starting values
    best_loss = float("inf")
    history   = list[dict[str, int | float]]()
    
    # Run each epoch
    for epoch in range(1, config.train.epochs + 1):
        # Train and validate (the learning rate moves at each weight update inside train_epoch())
        learning_rate = float(components.optimizer.param_groups[0]["lr"])
        train_metrics = train_epoch(config, device, model, teacher, bundle.train_loader, components)
        val_metrics   = evaluate_epoch(config, device, model, bundle.validation_loader, components, "Validation", low_precision=True)
        
        # Save the latest weights, and the best weights by validation loss
        torch.save(model.state_dict(), log_dir / "latest.pth")
        if val_metrics.loss < best_loss:
            best_loss = val_metrics.loss
            torch.save(model.state_dict(), log_dir / "best.pth")
        
        # Record, save, and show the results
        history.append({"epoch": epoch, "learning_rate": learning_rate, "train_loss": train_metrics.loss, "val_loss": val_metrics.loss, "val_acc": val_metrics.accuracy, "best_loss": best_loss})
        save_history(log_dir, history)
        save_graph(history, log_dir)
        print_epoch(history, config.train.epochs)
    
    return None
