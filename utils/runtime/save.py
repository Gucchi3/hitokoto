# @file save.py
# @brief Save training files
# @author Kohei Yamaguchi, Claude

from dataclasses import asdict
from pathlib     import Path
from types       import SimpleNamespace

import json
import matplotlib.pyplot as plt

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
    #? Called from utils/trainer/loop.py after each epoch.
    '''
    
    (log_dir / "result.json").write_text(json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return None


def save_graph(history: list[dict[str, int | float]], log_dir: Path) -> None:
    '''
    #* Save the loss and accuracy curves to training_curves.png.
    #? Called from utils/trainer/loop.py after each epoch.
    '''
    
    # Take each value out of the history
    epochs       = [record["epoch"]      for record in history]
    train_losses = [record["train_loss"] for record in history]
    val_losses   = [record["val_loss"]   for record in history]
    val_accs     = [record["val_acc"]    for record in history]
    best_loss    = history[-1]["best_loss"]
    latest_acc   = history[-1]["val_acc"]
    
    # Make the figure with two graphs
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    
    # Left: train and validation loss, with the best validation loss
    ax_loss.plot(epochs, train_losses, color="blue"  , marker=None, label="Train loss"     )
    ax_loss.plot(epochs, val_losses  , color="yellow", marker=None, label="Validation loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.text(0.03, 0.97, f"Best validation loss: {best_loss:.4f}", transform=ax_loss.transAxes, va="top")
    ax_loss.legend(loc="upper right")
    
    # Right: validation accuracy, with the latest value
    ax_acc.plot(epochs, val_accs, color="red", marker=None, label="Validation accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.text(0.03, 0.97, f"Latest acc: {latest_acc:.3f}", transform=ax_acc.transAxes, va="top")
    ax_acc.legend(loc="upper right")
    
    # Save the figure
    fig.savefig(log_dir / "training_curves.png", dpi=150)
    plt.close(fig)
    return None
