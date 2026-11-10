# @file score.py
# @brief Log probabilities of token lists, for the rules of the Tiny-ML Leaderboard and the Open SLM Leaderboard

from rich.progress import track
from torch         import nn

import math
import torch



def score_texts(model: nn.Module, ids_list: list[list[int]], device: torch.device, batch_size: int) -> list[float]:
    '''
    #* Return, for each token list, the sum of log P(token | the tokens before it) from the second token to the last.
    #? Used by benchmark.py for BLiMP, ARC-Easy, and WikiText-2 (the Tiny-ML Leaderboard).
    #! The first token is not scored, because no token comes before it (no BOS token is added). A list of fewer than 2 tokens gets -inf.
    #! The lists of a batch are padded on the right. The model must be causal, so that the padding does not change the scores of the tokens before it.
    #! A progress bar is shown while the batches run, and removed at the end.
    '''
    
    # Score the lists of 2 or more tokens, batch_size lists at a time
    scores = [-math.inf] * len(ids_list)
    usable = [index for index, ids in enumerate(ids_list) if len(ids) >= 2]
    with torch.no_grad():
        for start in track(range(0, len(usable), batch_size), description="Scoring", transient=True):
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


def score_continuations(model: nn.Module, requests: list[tuple[list[int], int]], device: torch.device, batch_size: int) -> list[float]:
    '''
    #* Return, for each pair of a token list and a count n, the sum of log P(token | the tokens before it) over the last n tokens of the list.
    #? Used by benchmark.py for HellaSwag, ARC, PIQA, and ArithMark-3 (the Open SLM Leaderboard), where the last n tokens are the choice.
    #! The model reads the list without its last token, so n must be 1 or more and less than the length of the list.
    #! The longest lists run first, so that the lists of a batch have similar lengths (little padding), and a lack of memory shows at the first batch. The scores come back in the order of requests.
    #! The lists of a batch are padded on the right, so the model must be causal. A progress bar is shown while the batches run, and removed at the end.
    '''
    
    # Sort the requests from the longest list to the shortest (requests of the same length keep their order)
    scores = [0.0] * len(requests)
    order  = sorted(range(len(requests)), key=lambda index: len(requests[index][0]), reverse=True)
    with torch.no_grad():
        for start in track(range(0, len(order), batch_size), description="Scoring", transient=True):
            # Make the inputs, the targets (the inputs moved by one token), and the mask of the last n targets of each list
            batch   = order[start:start + batch_size]
            width   = len(requests[batch[0]][0]) - 1
            inputs  = torch.zeros((len(batch), width), dtype=torch.int64)
            targets = torch.zeros((len(batch), width), dtype=torch.int64)
            mask    = torch.zeros((len(batch), width), dtype=torch.bool)
            for row, index in enumerate(batch):
                ids, count                 = torch.tensor(requests[index][0]), requests[index][1]
                end                        = len(ids) - 1
                inputs[row, :end]          = ids[:-1]
                targets[row, :end]         = ids[1:]
                mask[row, end - count:end] = True
            
            # Add the log probabilities of the masked targets
            logits    = model(inputs.to(device), torch.ones_like(inputs).to(device)).float()
            log_probs = torch.log_softmax(logits, dim=-1).gather(-1, targets.to(device).unsqueeze(-1)).squeeze(-1)
            for index, total in zip(batch, torch.where(mask.to(device), log_probs, 0.0).sum(dim=-1).tolist()):
                scores[index] = total
    return scores
