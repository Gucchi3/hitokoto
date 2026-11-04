# @file load.py
# @brief Load a trained model, its weights, and its tokenizer for the validator

from pathlib      import Path
from rich.console import Console
from rich.text    import Text
from tokenizers   import Tokenizer
from torch        import nn
from types        import SimpleNamespace

import torch

from model            import build_model
from utils.exceptions import Error
from utils.runtime    import setup_device, setup_seed



def load_model(config: SimpleNamespace) -> tuple[nn.Module, Tokenizer, torch.device]:
    '''
    #* Set the seed and the device, and return the model with the weights in model.checkpoint, the tokenizer in data.tokenizer_dir, and the device.
    #? Used by chat_model() in chat.py and benchmark_model() in benchmark.py.
    #! The model is in evaluation mode and in float32. With a null model.checkpoint, the model keeps random weights, and a warning is shown.
    '''
    
    # Check that the tokenizer is there
    tokenizer_path = Path(config.data.tokenizer_dir) / "tokenizer.json"
    if not tokenizer_path.exists():
        raise Error("Tokenizer is missing.", detail=f"Missing: {tokenizer_path}", hint="Set data.tokenizer_dir to the folder of the tokenizer of the model.", code=33)
    
    # Load the model with its weights on the device
    setup_seed(config.run.seed)
    device = setup_device(config.run.device)
    model  = build_model(config.model.name, config.model.checkpoint).to(device=device, dtype=torch.float32).eval()
    if config.model.checkpoint is None:
        Console().print(Text("[Validator] model.checkpoint is null, so the model has random weights.", style="yellow"))
    return model, Tokenizer.from_file(str(tokenizer_path)), device
