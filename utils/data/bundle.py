# @file bundle.py
# @brief Choose a DataBundle builder from the dataset name
# @author Kohei Yamaguchi, Claude

##########################################################
'''
Add a dataset
1. Write build_xxx_bundle(config, seed) in utils/data/xxx.py to return a DataBundle.
2. Add "name: build_xxx_bundle" to DATASET_REGISTRY.
3. Set config.data.name to that name.
'''
##########################################################

from collections.abc import Callable
from types           import SimpleNamespace

from utils.exceptions import Error
from .pretrain        import build_pretrain_bundle
from .sft             import build_sft_bundle
from .types           import DataBundle

DatasetBuilder = Callable[[SimpleNamespace, int], DataBundle]

DATASET_REGISTRY: dict[str, DatasetBuilder] = {
    "pretrain": build_pretrain_bundle,
    "sft"     : build_sft_bundle,
}



def build_data_bundle(config: SimpleNamespace, seed: int) -> DataBundle:
    '''
    #* Return the DataBundle for config.name.
    #? Called from utils/trainer/pipeline.py with config.data.
    '''
    
    if config.name not in DATASET_REGISTRY:
        raise Error("Unknown dataset.", detail=f"data.name: {config.name!r}\nAvailable: {', '.join(DATASET_REGISTRY)}", hint="Add a builder to DATASET_REGISTRY in utils/data/bundle.py.", code=5)
    return DATASET_REGISTRY[config.name](config, seed)
