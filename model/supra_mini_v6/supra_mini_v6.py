# @file supra_mini_v6.py
# @brief Supra-Mini-v6, a Llama model of 1.41M parameters from the Tiny-ML Leaderboard

from torch    import nn
from torch.nn import functional as F

import math
import torch

from .llama_block import LlamaBlock

VOCAB_SIZE        = 4096
HIDDEN_SIZE       = 128
INTERMEDIATE_SIZE = 256
NUM_LAYERS        = 6
NUM_HEADS         = 4
NUM_KV_HEADS      = 2
MAX_POSITIONS     = 1024
ROPE_THETA        = 10000.0
RMS_NORM_EPS      = 1e-5
INIT_STD          = 0.02



class SupraMiniV6(nn.Module):
    '''
    #* Hold Supra-Mini-v6, a Llama model of 1,410,688 parameters: 6 layers, hidden size 128, 4 query heads and 2 key-value heads, and one embedding for both the input and the output.
    #? Made by build_model() in model/build.py, with model.name = "supra_mini_v6". The sizes follow config.json of SupraLabs/Supra-Mini-v6-1M on the Hugging Face Hub. Its tokenizer is made by python -m tool.data (model/weight/hotaru_edu/).
    #! The tokenizer must have VOCAB_SIZE (4096) tokens, the default --vocab_size of the dataset scripts in tool/data/. An input can have up to MAX_POSITIONS (1024) tokens.
    '''
    
    cos: torch.Tensor
    sin: torch.Tensor
    
    def __init__(self) -> None:
        '''
        #* Make the layers and the RoPE tables, and set the first weights.
        #! The first weights follow train_model.py of Supra-Mini-v6: normal with std 0.02, and std 0.02 / sqrt(2 * layers) for o_proj and down_proj, which add to the residual.
        '''
        
        super().__init__()
        
        # Make the embedding, the layers, and the last norm. The output uses the embedding weights again, so there is no separate output layer
        self.embed_tokens = nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
        self.layers       = nn.ModuleList(LlamaBlock(HIDDEN_SIZE, NUM_HEADS, NUM_KV_HEADS, INTERMEDIATE_SIZE, RMS_NORM_EPS) for _ in range(NUM_LAYERS))
        self.norm         = nn.RMSNorm(HIDDEN_SIZE, eps=RMS_NORM_EPS)
        
        # Make the cos and sin tables of RoPE for each position. They are not saved with the weights, because they are made again here
        head_dim = HIDDEN_SIZE // NUM_HEADS
        inv_freq = 1.0 / ROPE_THETA ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        angles   = torch.outer(torch.arange(MAX_POSITIONS, dtype=torch.float32), inv_freq).repeat(1, 2)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)
        
        # Set the first weights of the matrices (the norm weights start at 1)
        for name, parameter in self.named_parameters():
            if parameter.ndim == 2:
                residual = name.endswith(("o_proj.weight", "down_proj.weight"))
                nn.init.normal_(parameter, std=INIT_STD / math.sqrt(2 * NUM_LAYERS) if residual else INIT_STD)
        return None
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        '''
        #* Return the logits of the next token at each position, of shape (batch, tokens, VOCAB_SIZE).
        #? Called from train.py and evaluate.py in utils/trainer/, and from generate.py and score.py in utils/validator/.
        #! attention_mask is not used. The attention is causal, so padding at the end of a row does not change the tokens before it (and its targets are -100).
        #! For the CUDA graph, no tensor value is read on the CPU here.
        '''
        
        # Run the layers with the RoPE tables of the first positions
        tokens = input_ids.shape[1]
        x      = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, self.cos[:tokens], self.sin[:tokens])
        
        # Turn the last hidden states into logits with the embedding weights
        return F.linear(self.norm(x), self.embed_tokens.weight)
