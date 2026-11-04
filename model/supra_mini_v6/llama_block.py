# @file llama_block.py
# @brief One decoder layer of Llama: attention with RoPE and GQA, then a SwiGLU MLP

from torch    import nn
from torch.nn import functional as F

import torch



class LlamaBlock(nn.Module):
    '''
    #* Hold one decoder layer of Llama: RMSNorm and attention, then RMSNorm and a SwiGLU MLP, each added to the input (pre-norm residual).
    #? Made by SupraMiniV6 in supra_mini_v6.py, one for each layer. The layer names are the same as in LlamaDecoderLayer of transformers, without "self_attn." and "mlp.".
    #! hidden_size must be a multiple of num_heads, and num_heads a multiple of num_kv_heads. Each group of num_heads / num_kv_heads query heads shares one key head and one value head (GQA).
    '''
    
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, intermediate_size: int, eps: float) -> None:
        '''
        #* Make the norms and the linear layers (no biases).
        '''
        
        super().__init__()
        self.num_heads    = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim     = hidden_size // num_heads
        
        # Layers of the attention (the keys and the values have fewer heads than the queries)
        kv_size              = num_kv_heads * self.head_dim
        self.input_layernorm = nn.RMSNorm(hidden_size, eps=eps)
        self.q_proj          = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj          = nn.Linear(hidden_size, kv_size    , bias=False)
        self.v_proj          = nn.Linear(hidden_size, kv_size    , bias=False)
        self.o_proj          = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Layers of the MLP
        self.post_attention_layernorm = nn.RMSNorm(hidden_size, eps=eps)
        self.gate_proj                = nn.Linear(hidden_size      , intermediate_size, bias=False)
        self.up_proj                  = nn.Linear(hidden_size      , intermediate_size, bias=False)
        self.down_proj                = nn.Linear(intermediate_size, hidden_size      , bias=False)
        return None
    
    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        '''
        #* Return x of shape (batch, tokens, hidden_size) after the attention and the MLP.
        #! cos and sin are the RoPE tables for the positions of the tokens, of shape (tokens, head_dim).
        #! Each token sees only itself and the tokens before it (causal attention), so padding at the end of a row does not change the tokens before it.
        '''
        
        # Make the queries, keys, and values of shape (batch, heads, tokens, head_dim)
        batch, tokens, _ = x.shape
        h                = self.input_layernorm(x)
        q                = self.q_proj(h).view(batch, tokens, self.num_heads   , self.head_dim).transpose(1, 2)
        k                = self.k_proj(h).view(batch, tokens, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v                = self.v_proj(h).view(batch, tokens, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Turn the queries and the keys by the position of each token (RoPE, which pairs the first and the second half of each head as in Llama)
        half = self.head_dim // 2
        q    = q * cos + torch.cat((-q[..., half:], q[..., :half]), dim=-1) * sin
        k    = k * cos + torch.cat((-k[..., half:], k[..., :half]), dim=-1) * sin
        
        # Run the attention (each group of query heads uses one key head and one value head), and add it to x
        attention = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        x         = x + self.o_proj(attention.transpose(1, 2).reshape(batch, tokens, -1))
        
        # Run the SwiGLU MLP, and add it to x
        h = self.post_attention_layernorm(x)
        return x + self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))
