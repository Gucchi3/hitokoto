# @file pipeline.py
# @brief Training pipeline
# @author Kohei Yamaguchi, Claude

from rich.console import Console
from rich.panel   import Panel
from types        import SimpleNamespace

import math
import torch

from model                      import build_model
from tool.helpers               import remove_pycache
from utils.data                 import build_data_bundle
from utils.exceptions           import Error
from utils.runtime              import print_metadata, save_metadata, setup_device, setup_log_dir, setup_seed
from utils.trainer.optimization import build_loss_fn, build_optimizer, build_scheduler
from .epoch                     import evaluate_epoch
from .loop                      import train_loop
from .types                     import TrainingComponents

TRAIN_METHODS = ("normal", "distill")



def train_pipeline(config: SimpleNamespace) -> None:
    '''
    #* Prepare the data, the models, and the training parts, then train and test.
    #? Called from main.py. Calls train_loop() in loop.py, which calls epoch.py.
    #! Set train.method to normal or distill. distill also needs model.teacher_name, model.teacher_checkpoint, and train.distillation.
    '''
    
    # Check the training method
    if config.train.method not in TRAIN_METHODS:
        raise Error("Unknown training method.", detail=f"train.method: {config.train.method!r}\nAvailable: {', '.join(TRAIN_METHODS)}", hint="Set train.method to 'normal' or 'distill'.", code=9)
    
    # Set the seed and the device
    setup_seed(config.run.seed)
    device = setup_device(config.run.device)
    
    # Prepare the data
    bundle = build_data_bundle(config.data, config.run.seed)
    
    # Prepare the student, and the frozen teacher for distillation
    model   = build_model(config.model.name, config.model.checkpoint).to(device=device, dtype=torch.float32)
    teacher = None
    if config.train.method == "distill":
        teacher = build_model(config.model.teacher_name, config.model.teacher_checkpoint).to(device)
        teacher.eval()
        teacher.requires_grad_(False)
    
    # Build the loss, optimizer, and scheduler (the scheduler moves one step at each weight update)
    loss_fn, loss_kind = build_loss_fn(config)
    optimizer          = build_optimizer(config, model)
    total_steps        = config.train.epochs * math.ceil(len(bundle.train_loader) / config.train.accumulation_steps)
    scheduler          = build_scheduler(config, optimizer, total_steps)
    components         = TrainingComponents(loss_fn, loss_kind, optimizer, scheduler)
    
    # Make a log folder, then save and show the settings
    log_dir = setup_log_dir(config.run.log_dir)
    save_metadata(bundle.metadata, config, log_dir)
    print_metadata(config)
    remove_pycache()
    
    # Train, then test the weights of the last epoch in float32
    train_loop(config, device, log_dir, bundle, model, teacher, components)
    test_metrics = evaluate_epoch(config, device, model, bundle.test_loader, components, "Test", low_precision=False)
    Console().print(Panel(f"[bold green]Test loss: {test_metrics.loss:.4f} | Test accuracy: {test_metrics.accuracy:.2%}[/bold green]", border_style="green", expand=False))
    return None
