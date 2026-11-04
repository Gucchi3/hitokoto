# @file types.py
# @brief Data types

from dataclasses      import dataclass
from math             import prod
from torch.utils.data import DataLoader
from typing           import Any

from .epoch_sampler import EpochSampler

'''
DataBundle
├── train_loader
├── train_sampler: EpochSampler
├── validation_loader
├── test_loader
└── metadata: DatasetMetadata
    ├── name
    ├── input_shape / input_size
    ├── num_classes / class_names
    ├── train_size / validation_size / test_size
    ├── split_seed
    └── preprocessing: PreprocessingMetadata
        ├── name
        ├── parameters
        └── output_dtype
'''



@dataclass(frozen=True)
class PreprocessingMetadata:
    '''
    #* Hold the details of the data preprocessing, such as the tokenizer.
    '''
    
    name        : str
    parameters  : dict[str, str | int | float | bool]
    output_dtype: str


@dataclass(frozen=True)
class DatasetMetadata:
    '''
    #* Hold the details of the dataset and its split.
    #? Saved to metadata.json by save_metadata() in utils/runtime/save.py.
    '''
    
    name           : str
    input_shape    : tuple[int, ...] | None
    num_classes    : int | None
    class_names    : tuple[str, ...] | None
    train_size     : int
    validation_size: int
    test_size      : int
    split_seed     : int
    preprocessing  : PreprocessingMetadata
    
    @property
    def input_size(self) -> int | None:
        '''
        #* Return the number of elements in one input of fixed shape.
        #! Return None for inputs of different lengths.
        '''
        
        return prod(self.input_shape) if self.input_shape is not None else None


@dataclass(frozen=True)
class DataBundle:
    '''
    #* Group the train, validation, and test loaders with the sampler of the train loader and the metadata.
    #! The trainer moves train_sampler to the place where the training is, such as the place of a resumed run.
    '''
    
    train_loader     : DataLoader[Any]
    train_sampler    : EpochSampler
    validation_loader: DataLoader[Any]
    test_loader      : DataLoader[Any]
    metadata         : DatasetMetadata
