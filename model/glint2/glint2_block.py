# @file glint2_block.py
# @brief The shared block of Glint-2: attention with RoPE and a window, then a SwiGLU MLP

from torch    import nn
from torch.nn import functional as F

import torch



class Glint2Block(nn.Module):
    '''
    #* Hold the one block that Glint-2 runs again and again: RMSNorm and attention, then RMSNorm and a SwiGLU MLP, each added to the input (pre-norm residual).
    #? Made by Glint2 in glint2.py, once for the whole model, and once more for the coda. The layers follow generate.py of Glint-Research/Glint-2, with shorter names (attn.qkv -> qkv, ffn.gate_up -> gate_up, and so on).
    #! The queries, keys, and values come from one matrix (qkv), and the loop's LoRA output (qkv_delta) is added to them. All heads have their own keys and values (no GQA).
    '''
    
    def __init__(self, dim: int, num_heads: int, ffn_hidden: int) -> None:
        '''
        #* Make the norms and the linear layers (no biases).
        #! The norms use the default eps of nn.RMSNorm, as in generate.py of Glint-2.
        '''
        
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        
        # Layers of the attention
        self.attn_norm = nn.RMSNorm(dim)
        self.qkv       = nn.Linear(dim, 3 * dim, bias=False)
        self.out       = nn.Linear(dim, dim    , bias=False)
        
        # Layers of the MLP (the gate and the up layer are one matrix)
        self.ffn_norm = nn.RMSNorm(dim)
        self.gate_up  = nn.Linear(dim       , 2 * ffn_hidden, bias=False)
        self.down     = nn.Linear(ffn_hidden, dim           , bias=False)
        return None
    
    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, qkv_delta: torch.Tensor, window: torch.Tensor | None) -> torch.Tensor:
        '''
        #* Return x of shape (batch, tokens, dim) after the attention and the MLP.
        #! cos and sin are the RoPE tables of shape (tokens, head_dim / 2). qkv_delta has the shape of qkv(x), (batch, tokens, 3 * dim).
        #! window is the mask of the tokens each token may see (itself and the tokens before it, inside the window), or None when every earlier token is inside the window.
        '''
        
        # Make the queries, keys, and values of shape (batch, heads, tokens, head_dim), with the loop's LoRA output added
        batch, tokens, dim = x.shape
        q, k, v            = (self.qkv(self.attn_norm(x)) + qkv_delta).split(dim, dim=-1)
        q                  = q.view(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k                  = k.view(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v                  = v.view(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Turn the queries and the keys by the position of each token (RoPE, which pairs each even element with the odd element after it)
        q = torch.stack((q[..., 0::2] * cos - q[..., 1::2] * sin, q[..., 0::2] * sin + q[..., 1::2] * cos), dim=-1).flatten(-2)
        k = torch.stack((k[..., 0::2] * cos - k[..., 1::2] * sin, k[..., 0::2] * sin + k[..., 1::2] * cos), dim=-1).flatten(-2)
        
        # Run the attention (causal, and inside the window when one is given), and add it to x
        attention = F.scaled_dot_product_attention(q, k, v, is_causal=True) if window is None else F.scaled_dot_product_attention(q, k, v, attn_mask=window)
        x         = x + self.out(attention.transpose(1, 2).reshape(batch, tokens, dim))
        
        # Run the SwiGLU MLP, and add it to x
        gate, up = self.gate_up(self.ffn_norm(x)).chunk(2, dim=-1)
        return x + self.down(F.silu(gate) * up)
