# @file byte_window.py
# @brief Test a byte model on fixed windows of a token file, in the same way as lab/tinylm_subbit (next-byte accuracy and bits per byte)
# @author Kohei Yamaguchi, Claude

from json         import JSONDecodeError
from pathlib      import Path
from rich.console import Console
from types        import SimpleNamespace

import argparse
import json
import math
import numpy as np
import torch
import torch.nn as nn

from model            import build_model
from utils.exceptions import Error
from utils.runtime    import remove_pycache, setup_device

# The windows that go through the model at once (as TEST_CHUNK in tinylm_subbit)
WINDOW_CHUNK = 128



def byte_window_evaluate(model: nn.Module, ids: torch.Tensor, count: int, context: int, low_precision: bool) -> tuple[float, float]:
    '''
    #* Return the bits per byte (cross entropy / ln 2) and the next-token accuracy of the model over count fixed windows of context tokens spread evenly over ids.
    #? Called from main(). The windows are made as text_data_windows() in lab/tinylm_subbit/utils/text_data.py: starts = linspace(0, len - context - 2, count) in float32 on the device of ids, cut to integers.
    #! Put ids on the same kind of device as tinylm_subbit (CUDA) to get the same windows: the float32 linspace can round in another way on the CPU.
    #! low_precision runs the model in BF16 autocast on CUDA, as lm_train_evaluate() in tinylm_subbit. Without it the model runs in float32.
    '''
    
    # Cut the fixed windows and the tokens that follow each position
    starts  = torch.linspace(0, ids.numel() - context - 2, count, device=ids.device).long()
    window  = ids[starts[:, None] + torch.arange(context + 1, device=ids.device)].long()
    tokens  = window[:, :-1]
    targets = window[:, 1:]
    
    # Add the cross entropy and the correct tokens over the windows, a chunk at a time
    model.eval()
    total   = 0.0
    correct = 0
    with torch.no_grad(), torch.autocast(device_type=ids.device.type, dtype=torch.bfloat16, enabled=low_precision and ids.device.type == "cuda"):
        for begin in range(0, count, WINDOW_CHUNK):
            chunk    = tokens[begin:begin + WINDOW_CHUNK]
            logits   = model(chunk, torch.ones_like(chunk)).float()
            total   += float(nn.functional.cross_entropy(logits.flatten(0, 1), targets[begin:begin + WINDOW_CHUNK].flatten(), reduction="sum"))
            correct += int((logits.argmax(dim=-1) == targets[begin:begin + WINDOW_CHUNK]).sum())
    return total / targets.numel() / math.log(2), correct / targets.numel()


def main() -> None:
    '''
    #* Read the command line and the config, load the model with the checkpoint, and show its bits per byte and accuracy on the fixed windows, in BF16 and in float32.
    #! Run from the project folder, such as python -m tool.evaluate.byte_window --config config/tiny_gpt.json --checkpoint log/<run>/best.pth.
    #! The token file is a .bin of uint16 token IDs, such as data/tinystories_byte/validation.bin (made by python -m tool.data.byte_text).
    '''
    
    # Read the command line
    parser = argparse.ArgumentParser(description="Test a byte model on fixed windows of a token file, in the same way as lab/tinylm_subbit.")
    parser.add_argument("--config"    , type=Path, required=True                                     , help="config of the model (model.name and data.max_length are used)")
    parser.add_argument("--checkpoint", type=Path, required=True                                     , help="weights of the model, such as log/<run>/best.pth")
    parser.add_argument("--tokens"    , type=Path, default=Path("data/tinystories_byte/validation.bin"), help="token file (.bin of uint16)")
    parser.add_argument("--count"     , type=int , default=1024                                      , help="number of windows")
    
    # Check the files and the config, and stop quietly on Ctrl+C, and always remove the caches
    try:
        args = parser.parse_args()
        if not (args.checkpoint.is_file() and args.tokens.is_file()) or args.count < 1:
            raise Error("Invalid settings of the window test.", detail=f"--checkpoint {args.checkpoint}, --tokens {args.tokens}, --count {args.count}", hint="Give existing files and a count of 1 or more.", code=40)
        try:
            with args.config.open(encoding="utf-8") as file:
                config = json.load(file, object_hook=lambda data: SimpleNamespace(**data))
        except (OSError, JSONDecodeError) as error:
            raise Error("Failed to load the config file.", detail=f"{args.config}: {error}", hint="Check the path and the JSON format.", code=1)
        
        # Load the model and the tokens on the device
        device = setup_device(config.run.device)
        model  = build_model(config.model.name, str(args.checkpoint)).to(device=device, dtype=torch.float32)
        ids    = torch.from_numpy(np.fromfile(args.tokens, dtype=np.uint16).astype(np.int64)).to(device)
        
        # Test in BF16 (as tinylm_subbit) and in float32
        console = Console()
        for name, low_precision in (("BF16", True), ("float32", False)):
            bits, accuracy = byte_window_evaluate(model, ids, args.count, config.data.max_length, low_precision)
            console.print(f"[Window test] {name:7s} {args.count} windows of {config.data.max_length} tokens: accuracy {accuracy:.2%}, {bits:.4f} bits/byte")
    except KeyboardInterrupt:
        Error.show_interrupted()
    finally:
        remove_pycache()
    return None


if __name__ == "__main__":
    main()
