# @file score.py
# @brief Log probabilities of token lists, by the rules of the Tiny-ML Leaderboard

from torch import nn

import math
import torch



def score_texts(model: nn.Module, ids_list: list[list[int]], device: torch.device, batch_size: int) -> list[float]:
    '''
    #* Return, for each token list, the sum of log P(token | the tokens before it) from the second token to the last.
    #? Used by benchmark.py for BLiMP, ARC-Easy, and WikiText-2.
    #! The first token is not scored, because no token comes before it (no BOS token is added). A list of fewer than 2 tokens gets -inf.
    #! The lists of a batch are padded on the right. The model must be causal, so that the padding does not change the scores of the tokens before it.
    '''
    
    # Score the lists of 2 or more tokens, batch_size lists at a time
    scores = [-math.inf] * len(ids_list)
    usable = [index for index, ids in enumerate(ids_list) if len(ids) >= 2]
    with torch.no_grad():
        for start in range(0, len(usable), batch_size):
            # Make the inputs, the targets (the inputs moved by one token), and the mask of the tokens that are not padding
            batch   = usable[start:start + batch_size]
            width   = max(len(ids_list[index]) for index in batch) - 1
            inputs  = torch.zeros((len(batch), width), dtype=torch.int64)
            targets = torch.zeros((len(batch), width), dtype=torch.int64)
            mask    = torch.zeros((len(batch), width), dtype=torch.bool)
            for row, index in enumerate(batch):
                ids                         = torch.tensor(ids_list[index])
                inputs[row, :len(ids) - 1]  = ids[:-1]
                targets[row, :len(ids) - 1] = ids[1:]
                mask[row, :len(ids) - 1]    = True
            
            # Add the log probabilities of the targets that are not padding
            logits    = model(inputs.to(device), torch.ones_like(inputs).to(device)).float()
            log_probs = torch.log_softmax(logits, dim=-1).gather(-1, targets.to(device).unsqueeze(-1)).squeeze(-1)
            for index, total in zip(batch, (log_probs * mask.to(device)).sum(dim=-1).tolist()):
                scores[index] = total
    return scores
