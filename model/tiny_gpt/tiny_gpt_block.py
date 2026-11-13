# @file tiny_gpt_block.py
# @brief One block of TinyGpt: a token mixer (gated short convolution or causal attention) and a GELU MLP, each after a LayerNorm
# @author Kohei Yamaguchi, Claude

from torch    import nn
from torch.nn import functional as F
from typing   import Literal

import torch

# The token mixer of a block: a gated short convolution (as in LFM2) or causal softmax attention
type Mixer = Literal["conv", "attention"]

# The kernel size of the gated short convolution (the current byte and the 2 before it), and the std of the first weights of the linear layers
CONV_KERNEL = 3
INIT_STD    = 0.02



class TinyGptBlock(nn.Module):
    '''
    #* Hold one block of TinyGpt: x + proj(mix(qkv(LN(x)))), then x + out(GELU(fc(LN(x)))).
    #? Made by TinyGpt in tiny_gpt.py. The same as GptBlock of lab/tinylm_subbit with float layers (setting None), so its weights load into this block by the same names.
    #? The convolution mixer uses the three parts of qkv as B, C, and h: (q x v) goes through a causal depthwise convolution over time, and is then multiplied by k. So both mixers have the same linear layers.
    #! The linear layers have no biases. Their first weights are normal with std INIT_STD, made in the same order as in tinylm_subbit, so the same seed gives the same first weights.
    '''
    
    qkv : nn.Linear
    proj: nn.Linear
    fc  : nn.Linear
    out : nn.Linear
    
    def __init__(self, dim: int, num_heads: int, mixer: Mixer) -> None:
        '''
        #* Make the norms, the linear layers, and the convolution (only for the convolution mixer).
        #! The linear layers are made on the meta device first, so their default initialization uses no random numbers. Then the random numbers are used as in tinylm_subbit.
        '''
        
        # The norm and the layers of the token mixer
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.qkv   = nn.Linear(dim, 3 * dim, bias=False, device="meta").to_empty(device="cpu")
        with torch.no_grad():
            self.qkv.weight.copy_(torch.randn(3 * dim, dim) * INIT_STD)
        self.proj  = nn.Linear(dim, dim, bias=False, device="meta").to_empty(device="cpu")
        with torch.no_grad():
            self.proj.weight.copy_(torch.randn(dim, dim) * INIT_STD)
        
        # The norm and the layers of the MLP (4 times as wide as the model)
        self.norm2 = nn.LayerNorm(dim)
        self.fc    = nn.Linear(dim, 4 * dim, bias=False, device="meta").to_empty(device="cpu")
        with torch.no_grad():
            self.fc.weight.copy_(torch.randn(4 * dim, dim) * INIT_STD)
        self.out   = nn.Linear(4 * dim, dim, bias=False, device="meta").to_empty(device="cpu")
        with torch.no_grad():
            self.out.weight.copy_(torch.randn(dim, 4 * dim) * INIT_STD)
        
        # The convolution of the convolution mixer (one filter for each channel)
        self.num_heads = num_heads
        self.mixer     = mixer
        self.conv      = nn.Conv1d(dim, dim, CONV_KERNEL, groups=dim, bias=False) if mixer == "conv" else None
        return None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        #* Return the block output for x (batch x tokens x dim).
        #! Each token sees only itself and the tokens before it.
        '''
        
        # The three parts of qkv
        batch, tokens, dim = x.shape
        q, k, v            = self.qkv(self.norm1(x)).split(dim, dim=2)
        
        # Gated short convolution: (q x v) through a causal depthwise convolution over time (padded on the left), then times k
        if self.conv is not None:
            gated = F.pad((q * v).transpose(1, 2), (CONV_KERNEL - 1, 0))
            x     = x + self.proj(self.conv(gated).transpose(1, 2) * k)
            return x + self.out(F.gelu(self.fc(self.norm2(x))))
        
        # Causal softmax attention, with the heads split from the model width
        q, k, v  = (part.view(batch, tokens, self.num_heads, dim // self.num_heads).transpose(1, 2) for part in (q, k, v))
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x        = x + self.proj(attended.transpose(1, 2).reshape(batch, tokens, dim))
        return x + self.out(F.gelu(self.fc(self.norm2(x))))
