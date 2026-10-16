# @file sft.py
# @brief Build the DataBundle for SFT on daily talk (empty for now)
# @author Kohei Yamaguchi, Claude

from types import SimpleNamespace

from .types import DataBundle



def build_sft_bundle(config: SimpleNamespace, seed: int) -> DataBundle:
    '''
    #* Return the train, validation, and test loaders for SFT (next-token prediction on the bot replies of the talks).
    #? Chosen in bundle.py when data.name is "sft". Uses Bin2dataset in bin2dataset.py.
    #! Not written yet. Only the bot replies are targets; the user turns and the padding are -100.
    '''
    
    raise NotImplementedError("build_sft_bundle() is not written yet (utils/data/sft.py).")
