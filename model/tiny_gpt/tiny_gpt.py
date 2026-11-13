# @file tiny_gpt.py
# @brief TinyGpt, the small byte-level language model of lab/tinylm_subbit (the ccca model, in float)
# @author Kohei Yamaguchi, Claude

from torch import nn

import torch

from .tiny_gpt_block import Mixer, TinyGptBlock

# The size of the model: 256 byte values, width 256, 4 blocks, 4 heads, and 256 bytes of context
VOCAB_SIZE = 256
DIM        = 256
NUM_LAYERS = 4
NUM_HEADS  = 4
MAX_LENGTH = 256

# The token mixer of each block: 3 gated short convolutions, then 1 attention (ccca)
MIXERS: tuple[Mixer, ...] = ("conv", "conv", "conv", "attention")

# The std of the first position embeddings
POSITION_STD = 0.02



class TinyGpt(nn.Module):
    '''
    #* Hold TinyGpt: byte embedding + learned positions, NUM_LAYERS TinyGptBlocks, a final LayerNorm, and an output layer over the 256 bytes.
    #? Made by build_model() in model/build.py, with model.name = "tiny_gpt". The same computation and weight names as TinyGpt of lab/tinylm_subbit (FP32, mixers conv,conv,conv,attention).
    #? Its tokenizer is the byte tokenizer in model/weight/tiny_gpt_byte/ (made by python -m tool.data.byte_text): token i is the byte i.
    #! The output layer is not tied to the embedding. The layers are made in the same order as in TinyGpt, so the same seed gives the same first weights.
    '''
    
    def __init__(self) -> None:
        '''
        #* Make the embedding, the positions, the blocks, the final norm, and the output layer.
        '''
        
        super().__init__()
        self.embed    = nn.Embedding(VOCAB_SIZE, DIM)
        self.position = nn.Parameter(torch.randn(MAX_LENGTH, DIM) * POSITION_STD)
        self.blocks   = nn.ModuleList(TinyGptBlock(DIM, NUM_HEADS, mixer) for mixer in MIXERS)
        self.norm     = nn.LayerNorm(DIM)
        self.head     = nn.Linear(DIM, VOCAB_SIZE, bias=False)
        return None
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        '''
        #* Return the logits of the next byte, of shape (batch, tokens, VOCAB_SIZE).
        #? Called from utils/trainer/train.py and evaluate.py, and from the validators.
        #! attention_mask is not used: every token sees the tokens before it, so padding is not supported. The pretraining windows have no padding.
        #! The inputs must have MAX_LENGTH tokens or fewer.
        '''
        
        x = self.embed(input_ids) + self.position[: input_ids.shape[1]]
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))
