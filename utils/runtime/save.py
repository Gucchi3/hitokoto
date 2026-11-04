# @file save.py
# @brief Save training files

from dataclasses       import asdict
from matplotlib.figure import Figure
from pathlib           import Path
from types             import SimpleNamespace

import json

from utils.data import DatasetMetadata



def save_metadata(metadata: DatasetMetadata, config: SimpleNamespace, log_dir: Path) -> None:
    '''
    #* Save the dataset details and the whole config to metadata.json.
    #? Called from utils/trainer/pipeline.py.
    '''
    
    metadata_data           = asdict(metadata)
    metadata_data["config"] = config
    (log_dir / "metadata.json").write_text(json.dumps(metadata_data, default=vars, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return None


def save_history(log_dir: Path, history: list[dict[str, int | float]]) -> None:
    '''
    #* Save the training history to result.json.
    #? Called from utils/trainer/loop.py after each validation.
    '''
    
    (log_dir / "result.json").write_text(json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return None


def save_graph(history: list[dict[str, int | float]], log_dir: Path) -> None:
    '''
    #* Save the loss and accuracy curves to training_curves.png, with the number of weight updates on the x-axis.
    #? Called from utils/trainer/loop.py after each validation.
    #! The figure is made with Figure, not pyplot, so no screen backend (Tk on Windows) is used. A Tk figure that is cleaned up in another thread (such as the pin_memory thread of a DataLoader) stops the whole run with Tcl_AsyncDelete.
    '''
    
    # Take each value out of the history
    steps        = [record["step"]       for record in history]
    train_losses = [record["train_loss"] for record in history]
    val_losses   = [record["val_loss"]   for record in history]
    val_accs     = [record["val_acc"]    for record in history]
    best_ppl     = history[-1]["best_ppl"]
    latest_acc   = history[-1]["val_acc"]
    
    # Make the figure with two graphs
    fig     = Figure(figsize=(12, 4), layout="constrained")
    ax_loss = fig.add_subplot(1, 2, 1)
    ax_acc  = fig.add_subplot(1, 2, 2)
    
    # Left: train and validation loss, with the best validation perplexity
    ax_loss.plot(steps, train_losses, color="blue"  , marker=None, label="Train loss"     )
    ax_loss.plot(steps, val_losses  , color="yellow", marker=None, label="Validation loss")
    ax_loss.set_xlabel("Step")
    ax_loss.set_ylabel("Loss")
    ax_loss.text(0.03, 0.97, f"Best validation perplexity: {best_ppl:.2f}", transform=ax_loss.transAxes, va="top")
    ax_loss.legend(loc="upper right")
    
    # Right: validation accuracy, with the latest value
    ax_acc.plot(steps, val_accs, color="red", marker=None, label="Validation accuracy")
    ax_acc.set_xlabel("Step")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.text(0.03, 0.97, f"Latest acc: {latest_acc:.3f}", transform=ax_acc.transAxes, va="top")
    ax_acc.legend(loc="upper right")
    
    # Save the figure
    fig.savefig(log_dir / "training_curves.png", dpi=150)
    return None
