# @file tokenizer.py
# @brief Make an untrained tokenizer and its trainer from a tokenizer type

##########################################################
'''
Add a tokenizer type
1. Write build_xxx_tokenizer(vocab_size, special_tokens) to return an untrained Tokenizer and its trainer.
2. Add "name: build_xxx_tokenizer" to TOKENIZER_REGISTRY, and add the name to TokenizerType.
3. Pass --tokenizer_type name to the tokenizer stage of a dataset script.
'''
##########################################################

from collections.abc import Callable
from tokenizers      import Tokenizer, decoders, models, pre_tokenizers, trainers
from typing          import Literal

TokenizerType    = Literal["bytelevel_bpe"]
TokenizerBuilder = Callable[[int, list[str]], tuple[Tokenizer, trainers.Trainer]]



def build_bytelevel_bpe_tokenizer(vocab_size: int, special_tokens: list[str]) -> tuple[Tokenizer, trainers.Trainer]:
    '''
    #* Make a byte-level BPE tokenizer and its trainer, for languages with spaces between words, such as English.
    #! Any text can be encoded, because the base vocabulary has all 256 bytes.
    #! For Japanese, add another type. This one makes a long run of text without spaces into one piece before BPE, which works badly.
    '''
    
    tokenizer               = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder       = decoders.ByteLevel()
    trainer                 = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens, initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    return tokenizer, trainer


TOKENIZER_REGISTRY: dict[str, TokenizerBuilder] = {
    "bytelevel_bpe": build_bytelevel_bpe_tokenizer,
}


def build_tokenizer(tokenizer_type: TokenizerType, vocab_size: int, special_tokens: list[str]) -> tuple[Tokenizer, trainers.Trainer]:
    '''
    #* Return the untrained tokenizer and the trainer for tokenizer_type.
    #? Called from train_corpus_tokenizer() in corpus.py. run_corpus() there lets only the names in TOKENIZER_REGISTRY through.
    '''
    
    return TOKENIZER_REGISTRY[tokenizer_type](vocab_size, special_tokens)
