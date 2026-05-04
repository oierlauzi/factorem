from typing import Tuple
import scipy.sparse
import numpy as np


class BsrArrayBuilder:
    def __init__(self, shape: Tuple[int, int]):
        self.data = []
        self.indices = []
        self.indptr = [0]
        self.shape = shape
        
    def add_block(self, index: int, block: np.ndarray):
        if len(self.data) > 0 and self.data[0].shape != block.shape:
            raise ValueError('Blocks must be homogeneously sized')
            
        self.data.append(block)
        self.indices.append(index)
    
    def next_block_row(self):
        self.indptr.append(len(self.indices))
        
    def build(self) -> scipy.sparse.bsr_array:
        return scipy.sparse.bsr_array(
            (np.stack(self.data), np.array(self.indices), np.array(self.indptr)),
            shape=self.shape
        )
