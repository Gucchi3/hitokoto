# @file test_byte_window.py
# @brief Tests of tool/evaluate/byte_window.py against a float64 reference over the same fixed windows
# @author Kohei Yamaguchi, Claude

import math
import pytest
import torch

from tests.fake_causal_model  import FakeCausalModel
from tool.evaluate.byte_window import WINDOW_CHUNK, byte_window_evaluate



@pytest.mark.parametrize("count", [WINDOW_CHUNK + 20, WINDOW_CHUNK + 1])
def test_byte_window_matches_reference(count: int) -> None:
    '''
    #* The bits per byte and the accuracy are those of a float64 reference over windows that start at k x step (k = 0 to count - 1), with the targets moved by one token.
    #! The text length makes the step an integer above 1, so the float32 linspace has no rounding and the reference starts are exact. count is above WINDOW_CHUNK, so the windows go through in more than one chunk (the last chunk has 1 window for WINDOW_CHUNK + 1).
    '''
    
    # A text whose window starts are 0, step, 2 x step, ... (len - context - 2 = (count - 1) x step)
    context, step        = 6, 3
    length               = (count - 1) * step + context + 2
    ids                  = torch.randint(0, 20, (length,), generator=torch.Generator().manual_seed(5))
    model                = FakeCausalModel(vocab=20, positions=context, seed=4)
    assert step > 1 and count > WINDOW_CHUNK
    
    # The float64 reference: the log likelihood and the most likely token at each position of each window
    embed, position, out = model.floats
    total, correct       = 0.0, 0
    for start in range(0, count * step, step):
        window = ids[start:start + context + 1].tolist()
        total -= model.log_likelihood(window, context)
        for index in range(1, context + 1):
            hidden   = [math.tanh(math.fsum(embed[token][k] for token in window[:index]) / index + position[index - 1][k]) for k in range(4)]
            logits   = [math.fsum(hidden[k] * out[k][token] for k in range(4)) for token in range(20)]
            correct += int(max(range(20), key=lambda token: logits[token]) == window[index])
    expected_bits     = total / (count * context) / math.log(2)
    expected_accuracy = correct / (count * context)
    assert 0.0 < expected_accuracy < 1.0
    
    # The window test in float32 on the CPU
    bits, accuracy = byte_window_evaluate(model, ids, count, context, low_precision=False)
    assert abs(bits - expected_bits) < 1e-5, (bits, expected_bits)
    assert accuracy == expected_accuracy
    return None
