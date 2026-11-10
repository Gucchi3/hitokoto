# @file fake_causal_model.py
# @brief A small causal language model with a float64 reference, for the tests of utils/validator

from torch import nn

import math
import torch



class FakeCausalModel(nn.Module):
    '''
    #* A small causal language model for the tests. The logits at each position depend on all the tokens up to it and on the position, so a wrong padding, cut, or order changes the scores.
    #? Used by tests/utils/validator/test_score.py and test_benchmark.py. log_likelihood() is the reference, written with Python floats (float64) and loops.
    #! The weights are fixed by the seed. The mask argument of forward() is not used, as in lm-evaluation-harness.
    '''
    
    embed   : torch.Tensor
    position: torch.Tensor
    out     : torch.Tensor
    
    def __init__(self, vocab: int, positions: int, seed: int) -> None:
        '''
        #* Make random weights for vocab tokens and positions positions, and keep a copy of them as Python floats for the reference.
        '''
        
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        embed     = torch.randn(vocab, 4, generator=generator)
        position  = torch.randn(positions, 4, generator=generator)
        out       = torch.randn(4, vocab, generator=generator) * 3
        self.register_buffer("embed"   , embed   )
        self.register_buffer("position", position)
        self.register_buffer("out"     , out     )
        self.floats = (embed.double().tolist(), position.double().tolist(), out.double().tolist())
        self.widths = list[int]()
        return None
    
    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        '''
        #* Return the logits (batch, length, vocab): tanh(the mean of the token vectors up to each position + the position vector) @ out.
        #! The width of each batch is kept in self.widths.
        '''
        
        self.widths.append(inputs.shape[1])
        counts = torch.arange(1, inputs.shape[1] + 1, device=inputs.device)[None, :, None]
        hidden = torch.cumsum(self.embed[inputs], dim=1) / counts + self.position[:inputs.shape[1]]
        return torch.tanh(hidden) @ self.out
    
    def log_likelihood(self, ids: list[int], count: int) -> float:
        '''
        #* Return the sum of log P(token | the tokens before it) over the last count tokens of ids, when the model reads ids without its last token from position 0.
        #! The reference for score_continuations(). Only Python floats are used, so the result does not depend on the batch.
        '''
        
        embed, position, out = self.floats
        total                = 0.0
        for index in range(len(ids) - count, len(ids)):
            # The hidden vector at position index - 1 reads ids[:index]
            hidden = [math.tanh(math.fsum(embed[token][k] for token in ids[:index]) / index + position[index - 1][k]) for k in range(4)]
            logits = [hidden[0] * out[0][token] + hidden[1] * out[1][token] + hidden[2] * out[2][token] + hidden[3] * out[3][token] for token in range(len(out[0]))]
            top    = max(logits)
            total += logits[ids[index]] - top - math.log(math.fsum(math.exp(logit - top) for logit in logits))
        return total
