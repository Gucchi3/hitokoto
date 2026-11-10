# @file glint2.py
# @brief Glint-2 with the coda block of the official checkpoint, a model of 1.71M parameters

from torch    import nn
from torch.nn import functional as F

import math
import torch

from .glint2_block import Glint2Block



VOCAB_SIZE       = 4096
DIM              = 96
NUM_HEADS        = 8
FFN_HIDDEN       = 2112
LORA_RANK        = 4
MAX_LOOPS        = 16
LOOPS            = 8
ATTENTION_WINDOW = 256
MAX_POSITIONS    = 1024
ROPE_BASE        = 10000.0
INIT_STD         = 0.02



class Glint2(nn.Module):
    '''
    #* Hold Glint-2 with one more block (the coda) after the loops and before the last norm. The coda has its own weights and runs once, with no loop embedding and no LoRA.
    #? Made by build_model() in model/build.py, with model.name = "glint2".
    #! The official glint-2.pt has this block (coda_layers = 1, 1,710,049 parameters), though its model card says there are no layers outside the loop, and generate.py drops its keys as stale. With these weights, this model gives a WikiText-2 byte perplexity of 3.126 by evaluate.py of Glint-2.3-Mini, and the published value is 3.127. Without the coda it is 3.501.
    #! The place and the inputs of the coda are the same as in forward() of infer.py in Glint-Research/Glint-2.3: after the loops, with no loop embedding and no LoRA, and before final_norm.
    '''
    
    cos   : torch.Tensor
    sin   : torch.Tensor
    window: torch.Tensor
    
    def __init__(self) -> None:
        '''
        #* Make the layers of Glint2 and the coda block, and set the first weights of the coda.
        #! The first weights of the coda are not published. They are the same as those of the shared block: normal with std 0.02, and std 0.02 / sqrt(2 * LOOPS) for out and down (which add to the residual).
        '''
        
        super().__init__()
        self.embed      = nn.Embedding(VOCAB_SIZE, DIM)
        self.shared     = Glint2Block(DIM, NUM_HEADS, FFN_HIDDEN)
        self.lora_down  = nn.ModuleList(nn.Linear(DIM, LORA_RANK, bias=False) for _ in range(MAX_LOOPS))
        self.lora_up    = nn.ModuleList(nn.Linear(LORA_RANK, 3 * DIM, bias=False) for _ in range(MAX_LOOPS))
        self.loop_embed = nn.Embedding(MAX_LOOPS, DIM)
        self.final_norm = nn.RMSNorm(DIM)
        
        head_dim  = DIM // NUM_HEADS
        inv_freq  = 1.0 / ROPE_BASE ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        angles    = torch.outer(torch.arange(MAX_POSITIONS, dtype=torch.float32), inv_freq)
        positions = torch.arange(MAX_POSITIONS)
        distance  = positions[:, None] - positions[None, :]
        self.register_buffer("cos"   , angles.cos()                                  , persistent=False)
        self.register_buffer("sin"   , angles.sin()                                  , persistent=False)
        self.register_buffer("window", (distance >= 0) & (distance < ATTENTION_WINDOW), persistent=False)
        
        for name, parameter in self.named_parameters():
            if name.startswith("lora_up."):
                nn.init.zeros_(parameter)
            elif parameter.ndim == 2:
                residual = name in ("shared.out.weight", "shared.down.weight")
                nn.init.normal_(parameter, std=INIT_STD / math.sqrt(2 * LOOPS) if residual else INIT_STD)
        self.coda = Glint2Block(DIM, NUM_HEADS, FFN_HIDDEN)
        
        # Set the first weights of the coda (the norm weights start at 1)
        for name, parameter in self.coda.named_parameters():
            if parameter.ndim == 2:
                nn.init.normal_(parameter, std=INIT_STD / math.sqrt(2 * LOOPS) if name in ("out.weight", "down.weight") else INIT_STD)
        return None
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        '''
        #* Return the logits of the next token at each position, of shape (batch, tokens, VOCAB_SIZE).
        #? Called from train.py and evaluate.py in utils/trainer/, and from generate.py and score.py in utils/validator/.
        #! attention_mask is not used, as in Glint2. For the CUDA graph, no tensor value is read on the CPU here.
        '''
        
        # Take the RoPE tables of the first positions, and the window mask only when a token can be farther than the window
        tokens = input_ids.shape[1]
        cos    = self.cos[:tokens]
        sin    = self.sin[:tokens]
        window = self.window[:tokens, :tokens] if tokens > ATTENTION_WINDOW else None
        
        # Run the shared block LOOPS times
        x = self.embed(input_ids)
        for loop in range(LOOPS):
            x     = x + self.loop_embed.weight[loop]
            delta = self.lora_up[loop](self.lora_down[loop](x))
            x     = self.shared(x, cos, sin, delta, window)
        
        # Run the coda once with no LoRA change, and turn the last hidden states into logits with the embedding weights
        x = self.coda(x, cos, sin, x.new_zeros(()), window)
        return F.linear(self.final_norm(x), self.embed.weight)
