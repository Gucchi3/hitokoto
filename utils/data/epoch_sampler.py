# @file epoch_sampler.py
# @brief Sampler that gives the train data in a new random order in each epoch, and can start in the middle of an epoch

from collections.abc  import Iterator
from torch.utils.data import Sampler

import numpy as np



class EpochSampler(Sampler[int]):
    '''
    #* Give the indices 0 to length - 1 in a random order that depends only on the seed and the epoch, from a chosen place in that order.
    #? Used for the train loader by build_pretrain_bundle() in pretrain.py. train_steps() in utils/trainer/train.py sets the place with set_position().
    #! The order of an epoch is made again from the seed and the epoch number, so a run that restarts from train_state.pt gets the same batches as a run without a stop.
    '''
    
    def __init__(self, length: int, seed: int) -> None:
        '''
        #* Keep the number of indices and the seed, and start at the beginning of epoch 0.
        '''
        
        super().__init__()
        self.length = length
        self.seed   = seed
        self.epoch  = 0
        self.start  = 0
        return None
    
    def set_position(self, epoch: int, start: int) -> None:
        '''
        #* Make the next pass give the order of the epoch, from its start-th index.
        '''
        
        self.epoch = epoch
        self.start = start
        return None
    
    def __len__(self) -> int:
        '''
        #* Return the number of indices that the next pass gives.
        '''
        
        return self.length - self.start
    
    def __iter__(self) -> Iterator[int]:
        '''
        #* Give the indices of the epoch in its random order, from the start-th one.
        #! The indices come from a NumPy array and not from a Python list, because a list of tens of millions of indices takes over 1 GB.
        '''
        
        order = np.random.default_rng([self.seed, self.epoch]).permutation(self.length)
        return iter(order[self.start:])
