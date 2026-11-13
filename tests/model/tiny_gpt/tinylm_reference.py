# @file tinylm_reference.py
# @brief Save the first weights, random weights, inputs, and outputs of TinyGpt of lab/tinylm_subbit (ccca, float), for test_tiny_gpt.py
# @author Kohei Yamaguchi, Claude

from pathlib import Path

import importlib
import os
import sys
import torch

# The size of the model (the same as model/tiny_gpt/tiny_gpt.py), and the token mixers of the blocks
MODEL_SIZE = (256, 4, 4, 256)
MIXERS     = ("conv", "conv", "conv", "attention")



def main() -> None:
    '''
    #* Make TinyGpt with seed 0 and keep its first weights, then move every weight by random noise, and save the weights and the outputs for two inputs to the file in sys.argv[1].
    #? Run by test_tiny_gpt.py with the Python of lab/tinylm_subbit, in that folder (so its model package is found, not the one of hitokoto).
    #! The noise makes the LayerNorm weights differ from 1 and the biases from 0, so that a wrong norm or a missing term changes the outputs.
    '''
    
    # TinyGpt of the tinylm_subbit folder (the working folder)
    sys.path.insert(0, os.getcwd())
    TinyGpt = importlib.import_module("model.tiny_gpt").TinyGpt
    
    # The first weights with seed 0
    torch.manual_seed(0)
    model = TinyGpt(None, *MODEL_SIZE, MIXERS)
    first = {name: value.clone() for name, value in model.state_dict().items()}
    
    # Random weights, and the outputs for a full window and a short one
    generator = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for value in model.parameters():
            value.add_(torch.randn(value.shape, generator=generator) * 0.05)
        inputs  = [torch.randint(0, 256, (3, 256), generator=generator), torch.randint(0, 256, (2, 17), generator=generator)]
        outputs = [model(tokens) for tokens in inputs]
    torch.save({"first": first, "weights": model.state_dict(), "inputs": inputs, "outputs": outputs}, Path(sys.argv[1]))
    return None


if __name__ == "__main__":
    main()
