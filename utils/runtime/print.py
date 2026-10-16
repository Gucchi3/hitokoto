# @file print.py
# @brief Training output
# @author Kohei Yamaguchi, Claude

from rich.console  import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table    import Table
from rich.text     import Text
from types         import SimpleNamespace



def print_metadata(config: SimpleNamespace) -> None:
    '''
    #* Show the settings before training.
    #? Called from utils/trainer/pipeline.py.
    '''
    
    # Make the table
    table = Table(title="Training Overview")
    table.add_column("Item" , style="bold cyan"   , no_wrap=True)
    table.add_column("Value", style="bright_white", no_wrap=True)
    
    # Run settings
    table.add_row("Run Mode"     , config.run.mode     )
    table.add_row("Seed"         , str(config.run.seed))
    table.add_row("Device"       , config.run.device   )
    table.add_row("Log Directory", config.run.log_dir  )
    table.add_row(""             , ""                  )
    
    # Data settings
    table.add_row("Dataset"   , config.data.name           )
    table.add_row("Batch Size", str(config.data.batch_size))
    table.add_row(""          , ""                         )
    
    # Model settings, with the teacher and its settings for distillation
    table.add_row("Model", config.model.name)
    if config.train.method == "distill":
        table.add_row("Teacher Model", config.model.teacher_name                 )
        table.add_row("Temperature"  , str(config.train.distillation.temperature))
        table.add_row("Alpha"        , str(config.train.distillation.alpha)      )
    table.add_row("", "")
    
    # Training settings
    table.add_row("Training Method"      , config.train.method                              )
    table.add_row("Epochs"               , str(config.train.epochs)                         )
    table.add_row("Accumulation Steps"   , str(config.train.accumulation_steps)             )
    table.add_row("Max Gradient Norm"    , str(config.train.max_grad_norm)                  )
    table.add_row("CUDA Graph"           , str(config.train.cuda_graph)                     )
    table.add_row("Loss"                 , config.train.loss.name                           )
    table.add_row("Optimizer"            , config.train.optimizer.name                      )
    table.add_row("Learning Rate"        , str(config.train.optimizer.learning_rate)        )
    table.add_row("Weight Decay"         , str(config.train.optimizer.weight_decay)         )
    table.add_row("Scheduler"            , config.train.scheduler.name                      )
    table.add_row("Warmup Steps"         , str(config.train.scheduler.warmup_steps)         )
    table.add_row("Minimum Learning Rate", str(config.train.scheduler.minimum_learning_rate))
    
    # Show the table
    Console().print(table)
    return None


def print_epoch(history: list[dict[str, int | float]], epochs: int) -> None:
    '''
    #* Show the results of the last epoch in one line.
    #? Called from utils/trainer/loop.py.
    '''
    
    # Make the line from the last record
    record  = history[-1]
    message = Text()
    message.append("Epoch"                          , style="bold cyan"   )
    message.append(f" {record['epoch']}/{epochs}"   , style="cyan"        )
    message.append(" | "                            )
    message.append("lr"                             , style="bold magenta")
    message.append(f"={record['learning_rate']:.3e}", style="magenta"     )
    message.append(" | "                            )
    message.append("train loss"                     , style="bold yellow" )
    message.append(f"={record['train_loss']:.3f}"   , style="yellow"      )
    message.append(" | "                            )
    message.append("validation loss"                , style="bold blue"   )
    message.append(f"={record['val_loss']:.3f} "    , style="blue"        )
    message.append("accuracy"                       , style="bold red"    )
    message.append(f"={record['val_acc']:.2%}"      , style="red"         )
    if record["val_loss"] == record["best_loss"]:
        message.append(" (best)", style="bold green")
    
    # Show the line
    Console().print(message)
    return None


def print_progress() -> Progress:
    '''
    #* Make a progress line with the step count and the running loss. The line goes away when it ends.
    #? Used in utils/trainer/epoch.py with a with statement. Add a task with a loss field, and update the loss with progress.update().
    '''
    
    return Progress(TextColumn("[bold cyan]{task.description}"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(), TimeRemainingColumn(), TextColumn("loss: {task.fields[loss]}"), transient=True)
