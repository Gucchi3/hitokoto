# @file hotaru.py
# @brief Build the DataBundle for Hotaru (empty for now)
# @author Kohei Yamaguchi, Claude

from types import SimpleNamespace

from .types import DataBundle



def build_hotaru_bundle(config: SimpleNamespace, seed: int) -> DataBundle:
    '''
    #* Return the train, validation, and test loaders for Hotaru.
    #? Chosen in bundle.py when data.name is "hotaru". The tokenizers are in data.tokenizer_dir and data.teacher_tokenizer_dir.
    #! Not written yet. Each batch must be a dict of input_ids, attention_mask, and targets (see utils/trainer/epoch.py).
    #! For distillation with a teacher that uses another tokenizer, also add teacher_input_ids and teacher_attention_mask, made from the same texts.
    #! For the CUDA graph, pad every batch to the same shape (max_length), and use pin_memory=True.
    '''
    
    raise NotImplementedError("build_hotaru_bundle() is not written yet (utils/data/hotaru.py).")
