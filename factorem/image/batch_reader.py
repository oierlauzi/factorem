from typing import Iterable, Optional, Tuple
from collections import OrderedDict
import numpy as np
import mrcfile
import os

from .image_location import ImageLocation

def _index_or_none(position_in_stack: Optional[int]) -> Optional[int]:
    return None if position_in_stack is None else position_in_stack - 1

def _batch_files(paths: Iterable[ImageLocation]) -> Tuple[str, Optional[slice]]:
    it = iter(paths)
    
    # Initialize with the first loop iteration
    path = next(it)
    current_filename = path.filename
    current_end = path.position_in_stack
    current_start = _index_or_none(current_end)
    
    for path in it:
        filename = path.filename
        index = _index_or_none(path.position_in_stack)
        
        if filename == current_filename and index == current_end and current_end is not None:
            current_end += 1
                
        else:
            if current_start is not None:
                assert (current_end is not None)
                yield current_filename, slice(current_start, current_end)
            else:
                yield current_filename, None
            
            current_filename = path.filename
            current_end = path.position_in_stack
            current_start = _index_or_none(current_end)
     
    if current_start is not None:
        assert (current_end is not None)
        yield current_filename, slice(current_start, current_end)
    else:
        yield current_filename, None

class BatchReader:
    def __init__(self, prefix: Optional[str] = None, max_open: int = 64):
        self._open_files = OrderedDict()
        self._prefix = prefix
        self._max_open = max_open
        
    def read_batch(self, locations: Iterable[ImageLocation]) -> np.ndarray:
        output_stacks = []
        
        for filename, index_slice in _batch_files(locations):
            mrc = self._read_file(filename)
            data = mrc.data
            
            if mrc.is_image_stack() or mrc.is_volume_stack():
                if index_slice is None:
                    raise RuntimeError(
                        'Image index should be provided for image stacks'
                    )
                output_stacks.append(data[index_slice])
            else:
                output_stacks.append(data[None])

        if len(output_stacks) == 1:
            result = output_stacks[0]
        else:
            result = np.concatenate(output_stacks, axis=0)

        return result
    
    def _read_file(self, filename: str) -> mrcfile.MrcFile:      
        mrc = self._open_files.get(filename, None)
        
        if mrc is None:            
            mrc = mrcfile.open(self._make_abs_filename(filename), 'r')
            self._open_files[filename] = mrc
            if(len(self._open_files) >= self._max_open):
                self._open_files.popitem(last=False)
                
        else:
            self._open_files.move_to_end(filename, last=True)

        assert(mrc is not None)
        return mrc
    
    def _make_abs_filename(self, filename: str) -> str:
        if self._prefix is not None:
            filename = os.path.join(self._prefix, filename)
            
        return filename
    