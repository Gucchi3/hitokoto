# @file pretrain.py
# @brief Build the DataBundle for pretraining (empty for now)
# @author Kohei Yamaguchi, Claude

from types import SimpleNamespace

from .types import DataBundle



def build_pretrain_bundle(config: SimpleNamespace, seed: int) -> DataBundle:
    '''
    #* Return the train, validation, and test loaders for pretraining (next-token prediction on plain text).
    #? Chosen in bundle.py when data.name is "pretrain". Uses Bin2dataset in bin2dataset.py.
    #! Not written yet. Every token is a target, so no loss mask is needed.
    '''
    
    raise NotImplementedError("build_pretrain_bundle() is not written yet (utils/data/pretrain.py).")
