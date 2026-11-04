# @file pipeline.py
# @brief Training pipeline

from pathlib      import Path
from rich.console import Console
from rich.panel   import Panel
from tokenizers   import Tokenizer
from types        import SimpleNamespace

import torch

from model                      import build_model
from utils.data                 import build_data_bundle
from utils.exceptions           import Error
from utils.runtime              import print_metadata, remove_pycache, save_metadata, setup_device, setup_log_dir, setup_seed
from utils.trainer.optimization import build_loss_fn, build_optimizer, build_scheduler
from utils.validator            import benchmark_check, benchmark_tasks
from .evaluate                  import evaluate_epoch
from .loop                      import loop_train
from .types                     import TrainState, TrainingComponents

TRAIN_METHODS = ("normal", "distill")



def pipeline_train(config: SimpleNamespace) -> None:
    '''
    #* Prepare the data, the models, and the training parts, then train and test.
    #? Called from main.py. Calls loop_train() in loop.py, which calls train.py and evaluate.py.
    #! Set train.method to normal or distill. distill also needs model.teacher_name, model.teacher_checkpoint, and train.distillation.
    #! With train.resume set to a train_state.pt, the training goes on from that state in the same log folder. The model and the optimizer must be the same as in the saved run.
    #! After the training, the test uses best.pth, the weights with the lowest validation perplexity.
    #! Without test data (a test of 0 in --split of tool/data), the benchmarks in validator.benchmark run on best.pth instead of the test, and benchmark.json goes to the log folder.
    '''
    
    # Check the training method and the length of the training
    if config.train.method not in TRAIN_METHODS:
        raise Error("Unknown training method.", detail=f"train.method: {config.train.method!r}\nAvailable: {', '.join(TRAIN_METHODS)}", hint="Set train.method to 'normal' or 'distill'.", code=9)
    if config.train.steps < 1 or config.train.eval_interval < 1:
        raise Error("Invalid length of the training.", detail=f"train.steps: {config.train.steps}, train.eval_interval: {config.train.eval_interval}", hint="Use 1 or more for both.", code=30)
    
    # Set the seed and the device
    setup_seed(config.run.seed)
    device = setup_device(config.run.device)
    
    # Prepare the data, and check that the train data fills one batch. Without test data, check the benchmark names now, because the benchmarks run after the training
    bundle = build_data_bundle(config.data, config.run.seed)
    if len(bundle.train_loader) == 0:
        raise Error("The train data is smaller than one batch.", detail=f"data.batch_size: {config.data.batch_size}", hint="Use a smaller data.batch_size, or more train data.", code=31)
    if len(bundle.test_loader) == 0:
        benchmark_check(config)
    
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
    scheduler          = build_scheduler(config, optimizer, config.train.steps)
    components         = TrainingComponents(loss_fn, loss_kind, optimizer, scheduler)
    
    # Read the training state for train.resume, or start from the first step in a new log folder
    state   = TrainState(step=0, epoch=0, batch=0, best_ppl=float("inf"), history=[])
    log_dir = setup_log_dir(config.run.log_dir) if config.train.resume is None else Path(config.train.resume).parent
    if config.train.resume is not None:
        try:
            saved = torch.load(config.train.resume, map_location=device, weights_only=True)
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
        except (OSError, KeyError, RuntimeError, ValueError) as error:
            raise Error("Could not load the training state.", detail=f"{config.train.resume}: {error}", hint="Set train.resume to a train_state.pt of a run with the same model and optimizer, or to null.", code=29)
        state = TrainState(**saved["state"])
    
    # Save and show the settings
    save_metadata(bundle.metadata, config, log_dir)
    print_metadata(config)
    remove_pycache()
    
    # Train, then load the best weights (the lowest validation perplexity)
    loop_train(config, device, log_dir, bundle, model, teacher, components, state)
    best_path = log_dir / "best.pth"
    if not best_path.exists():
        raise Error("There are no best weights.", detail=f"Missing: {best_path}", hint="The validation perplexity was never a finite number. Check the learning rate and the data, then train again.", code=37)
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    
    # Test the best weights in float32, or run the benchmarks on them when there is no test data
    if len(bundle.test_loader) > 0:
        test_metrics = evaluate_epoch(config, device, model, bundle.test_loader, components, "Test", low_precision=False)
        Console().print(Panel(f"[bold green]Test loss: {test_metrics.loss:.4f} | Test perplexity: {test_metrics.perplexity:.2f} | Test accuracy: {test_metrics.accuracy:.2%}[/bold green]", border_style="green", expand=False))
    else:
        benchmark_tasks(model.eval(), Tokenizer.from_file(str(Path(config.data.tokenizer_dir) / "tokenizer.json")), device, config, log_dir, str(best_path))
    return None
