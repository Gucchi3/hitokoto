# @file yokoyama.py
# @brief Yokoyama, a causal language model based on the EmbBERT architecture

from torch import nn

import math
import torch



VOCAB_SIZE       = 4096
HIDDEN_SIZE      = 128
REDUCED_EMBEDDING = 16
NUM_LAYERS        = 4
FORWARD_EXPANSION = 1.0
KERNEL_SIZE       = 32
MAX_LENGTH        = 256
EPSILON           = 1e-5



class NanoCausalEmbedder(nn.Module):
    '''
    #* Map tokens to hidden vectors with reduced token and position embeddings.
    #? This follows the Nano_Embedder used by the referenced EmbBERT implementation, without bidirectional sentence segments.
    #! Token and position embeddings use the reduced dimension, then expand to the model hidden dimension.
    '''
    
    def __init__(self) -> None:
        '''
        #* Make the reduced token and position embeddings and their expansion layers.
        '''
        
        super().__init__()
        self.token        = nn.Embedding(VOCAB_SIZE, REDUCED_EMBEDDING, padding_idx=0)
        self.tok_expander = nn.Linear(REDUCED_EMBEDDING, HIDDEN_SIZE, bias=False)
        self.position     = nn.Embedding(MAX_LENGTH, REDUCED_EMBEDDING)
        self.pos_expander = nn.Linear(REDUCED_EMBEDDING, HIDDEN_SIZE, bias=False)
        return None
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        '''
        #* Return one hidden vector for every input token.
        '''
        
        tokens    = input_ids.shape[1]
        positions = torch.arange(tokens, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        return self.tok_expander(self.token(input_ids)) + self.pos_expander(self.position(positions))



class EfficientDifferentialSkipAttention(nn.Module):
    '''
    #* Combine efficient causal attention with a depthwise causal convolution branch.
    #? The attention and convolution equations follow the EfficientDifferentialSkipAttention in the referenced repository.
    #! The convolution is left-padded by the framework and cut back to the input length, so it cannot read future tokens.
    '''
    
    def __init__(self) -> None:
        '''
        #* Make the differential attention, convolution, activation, and contraction layers.
        '''
        
        super().__init__()
        d_inner = int(HIDDEN_SIZE * FORWARD_EXPANSION)
        self.dropout    = nn.Dropout(0.1)
        self.act        = nn.SiLU()
        self.lambda_q1  = nn.Parameter(torch.zeros(HIDDEN_SIZE).normal_(mean=0, std=0.1))
        self.lambda_k1  = nn.Parameter(torch.zeros(HIDDEN_SIZE).normal_(mean=0, std=0.1))
        self.lambda_q2  = nn.Parameter(torch.zeros(HIDDEN_SIZE).normal_(mean=0, std=0.1))
        self.lambda_k2  = nn.Parameter(torch.zeros(HIDDEN_SIZE).normal_(mean=0, std=0.1))
        self.query      = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.att_output = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.conv1d     = nn.Conv1d(HIDDEN_SIZE, d_inner, KERNEL_SIZE, groups=HIDDEN_SIZE, padding=KERNEL_SIZE - 1)
        self.contract   = nn.Linear(d_inner, HIDDEN_SIZE)
        return None
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        '''
        #* Return the differential combination of the attention and convolution branches.
        '''
        
        lambda_1 = torch.exp(torch.dot(self.lambda_q1, self.lambda_k1))
        lambda_2 = torch.exp(torch.dot(self.lambda_q2, self.lambda_k2))
        
        scores  = torch.matmul(self.query(x), x.permute(0, 2, 1)) / math.sqrt(x.size(-1))
        scores  = scores.masked_fill(mask == 0, float("-inf"))
        weights = torch.nan_to_num(nn.functional.softmax(scores, dim=-1))
        context = torch.matmul(self.dropout(weights), x)
        out1    = self.att_output(context)
        
        tokens = x.shape[1]
        branch = self.conv1d(x.transpose(1, 2))[:, :, :tokens].transpose(1, 2)
        out2   = self.contract(self.act(branch))
        return lambda_1 * out1 - lambda_2 * out2



def make_score_mask(input_mask: torch.Tensor) -> torch.Tensor:
    '''
    #* Expand a padding mask into a score mask for every query position.
    '''
    
    batch_size, tokens = input_mask.shape
    score_mask = torch.zeros((batch_size, tokens, tokens), dtype=input_mask.dtype, device=input_mask.device)
    for batch in range(batch_size):
        length = (input_mask[batch] != 0).sum()
        score_mask[batch, :length, :] = input_mask[batch]
    return score_mask



class Yokoyama(nn.Module):
    '''
    #* Hold the causal EmbGPT-style language model with a reduced embedder and differential skip attention.
    #? Made by build_model() in model/build.py, with model.name = "yokoyama".
    #! The model uses the EmbBERT base dimensions from the referenced project: 4096 tokens, hidden size 128, four layers, reduced embedding 16, expansion 1, kernel size 32, and context length 256.
    '''
    
    def __init__(self) -> None:
        '''
        #* Make the embedder, pre-norm attention blocks, and tied-size language-model head.
        '''
        
        super().__init__()
        self.embedder  = NanoCausalEmbedder()
        self.norms     = nn.ModuleList(nn.RMSNorm(HIDDEN_SIZE, eps=EPSILON) for _ in range(NUM_LAYERS))
        self.attentions = nn.ModuleList(EfficientDifferentialSkipAttention() for _ in range(NUM_LAYERS))
        self.lm_head   = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE)
        return None
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        '''
        #* Return causal next-token logits of shape (batch, tokens, vocabulary).
        #? Called from the trainer and validators through model/build.py.
        #! Each token attends only to non-padding tokens at positions up to and including itself.
        '''
        
        tokens = input_ids.shape[1]
        padding_mask = make_score_mask(attention_mask)
        causal = torch.tril(torch.ones(tokens, tokens, dtype=padding_mask.dtype, device=input_ids.device))
        score_mask = padding_mask * causal
        
        x = self.embedder(input_ids)
        for norm, attention in zip(self.norms, self.attentions):
            normalized = norm(x)
            update = attention(normalized, score_mask)
            x = x + update
        return self.lm_head(x)
