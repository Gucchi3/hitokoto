# @file generate.py
# @brief Make new tokens one at a time with a language model

from torch import nn

import torch



def generate_tokens(model: nn.Module, ids: list[int], device: torch.device, max_length: int, max_new_tokens: int, temperature: float, top_k: int, stop_id: int | None) -> list[int]:
    '''
    #* Make up to max_new_tokens new token IDs that follow ids, one at a time, and stop before stop_id.
    #? Used by chat_model() in chat.py.
    #! The model sees only the last max_length tokens. temperature 0 always picks the most likely token. top_k keeps only the k most likely tokens before the pick (0 keeps all).
    #! The random pick uses the global random state of PyTorch, so setup_seed() makes the answers the same each time.
    '''
    
    new_ids = list[int]()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Find the scores of the next token from the last max_length tokens
            context = torch.tensor([(ids + new_ids)[-max_length:]], device=device)
            logits  = model(context, torch.ones_like(context))[0, -1].float()
            
            # Pick the most likely token, or a random token from the top_k by the probabilities softened by the temperature
            if temperature == 0:
                next_id = int(logits.argmax())
            else:
                if top_k > 0:
                    logits = logits.masked_fill(logits < logits.topk(min(top_k, logits.numel())).values[-1], -torch.inf)
                next_id = int(torch.multinomial(torch.softmax(logits / temperature, dim=-1), 1))
            
            # Stop before the stop token
            if next_id == stop_id:
                break
            new_ids.append(next_id)
    return new_ids
