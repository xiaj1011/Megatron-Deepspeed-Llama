# Copyright (c) 2023, XIAJUN (xiaj1011@gmail.com)
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2021, EleutherAI.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import struct
from functools import lru_cache
from itertools import accumulate

import torch
import numpy as np

from megatron import print_rank_0

def __best_fitting_dtype(vocab_size=None):
    if vocab_size is not None and vocab_size < 65500:
        return np.uint16
    
    return np.uint32

def make_builder(out_file, impl, vocab_size=None):
    if impl == 'mmap':
        return MMapIndexedDatasetBuilder(
            out_file, dtype=__best_fitting_dtype(vocab_size)
        )
    
    raise NotImplementedError(f"{impl} not implemented.")
    

def index_file_path(prefix_path):
    return prefix_path + ".idx"

def data_file_path(prefix_path):
    return prefix_path + ".bin"

_code_to_dtype = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float64,
    7: np.double,
    8: np.uint16,
    9: np.uint32,
    10: np.uint64,
}

def code(dtype) -> int:
    for k, _dtype in _code_to_dtype.items():
        if _dtype == dtype:
            return k
    
    raise ValueError(dtype)

def _warmup_mmap_file(path):
    # mmap是延迟加载，这里预加载到缓存中，提高后续mmap访问操作的性能
    with open(path, 'rb') as stream:
        while stream.read(100 * 1024 * 1024):
            pass
    
class MMapIndexedDataset(torch.utils.data.Dataset):
    class Index(object):
        _HDR_MAGIC = b"MMIDIDX\x00\x00"    
        
        @classmethod
        def writer(cls, path, dtype):
            class _Writer(object):
                def __enter__(self):
                    self._file = open(path, 'wb')
                    
                    # Write Magic string so we can check the file format then opening it again.
                    self._file.write(cls._HDR_MAGIC)
                    # Write version number
                    # Little endian unsigned 64 Bit integer
                    self._file.write(struct.pack("<Q", 1))
                    # Little endian unsigned 8 Bit integer
                    self._file.write(struct.pack("<B", code(dtype)))

                    return self
                
                @staticmethod
                def _get_pointers(sizes):
                    # https://github.com/EleutherAI/gpt-neox/pull/771
                    pointers = np.zeros(len(sizes), dtype=np.uint64)
                    sizes = np.array(sizes, dtype=np.uint64)
                    
                    np.cumsum(sizes[:-1], out=pointers[1:]) # why [:-1]?
                    pointers = pointers * dtype().itemsize
                    return pointers
                
                def write(self, sizes, doc_idx):
                    pointers = self._get_pointers(sizes)
                    
                    # Little endian unsigned 64 Bit integer
                    self._file.write(struct.pack("<Q", len(sizes))) # 多少个句子
                    self._file.write(struct.pack("<Q", len(doc_idx))) # 多少个文档，每个文档可能有多个句子
                    
                    sizes = np.array(sizes, dtype=np.uint32) # change int32
                    self._file.write(sizes.tobytes(order="C"))
                    del sizes
                    
                    pointers = np.array(pointers, dtype=np.uint64)
                    self._file.write(pointers.tobytes(order="C"))
                    del pointers
                    
                    doc_idx = np.array(doc_idx, dtype=np.uint64)
                    self._file.write(doc_idx.tobytes(order="C"))
                    
                def __exit__(self, exc_type, exc_val, exc_tb):
                    self._file.close()
                          
            return _Writer()   
                        
        def __init__(self, path, skip_warmup=False):
            with open(path, 'rb') as stream:
                magic_test = stream.read(9)
                assert magic_test == self._HDR_MAGIC, (
                    "Index file doesn't match expected format."
                    "Make sure that --dataset-impl is configured properly."
                )
                
                # Little endian unsigned 64 Bit integer
                version = struct.unpack("<Q", stream(8)) # 返回为tuple
                assert (1,) == version
                
                # Little endian unsigned 8 Bit integer
                (dtype_code, ) = struct.unpack("<B", stream.read(1))
                self._dtype = _code_to_dtype(dtype_code)
                self._dtype_size = self.dtype().itemsize
                
                self._len = struct.unpack("<Q", stream.read(8))[0]      # 多少个句子
                self._doc_count = struct.unpack("<Q", stream.read(8))[0] # 多少个文档
                offset = stream.tell()
                
            if not skip_warmup:
                print_rank_0("  warming up index map file...")
                _warmup_mmap_file(path)
            
            self._bin_buffer_mmap = np.memmap(path, mode="r", order="C")
            self._bin_buffer = memoryview(self._bin_buffer_mmap)
            
            print_rank_0("  reading sizes...")
            self._sizes = np.frombuffer(
                self._bin_buffer, dtype=np.uint32, count=self._len, offset=offset
            )
            
            print_rank_0("  reading pointers...")
            self._pointers = np.frombuffer(
                self._bin_buffer, dtype=np.uint64,
                count=self._len, offset=offset + self._sizes.nbytes
            )
            
            print_rank_0("  reading document index...")
            self._doc_idx = np.frombuffer(
                self._bin_buffer, 
                dtype=np.uint64,
                count=self._doc_count,
                offset=offset + self._sizes.nbytes + self._pointers.nbytes
            )
            
        def __del__(self):
            self._bin_buffer_mmap.__mmap.close()
            del self._bin_buffer_mmap
            
        @property
        def dtype(self):
            return self._dtype
                
        @property
        def sizes(self):
            return self._sizes
        
        @property
        def doc_idx(self):
            return self._doc_idx
        
        @lru_cache(maxsize=8)
        def __getitem__(self, i):
            return self._pointers[i], self._sizes[i]
        
        def __len__(self):
            return self._len
                 

    def __init__(self, path, skip_warmup=False):
        super().__init__()

        self._path = None
        self._index = None
        self._bin_buffer = None

        self._do_init(path, skip_warmup)

    def __getstate__(self):
        return self._path

    def __setstate__(self, state):
        self._do_init(state)

    def _do_init(self, path, skip_warmup):
        self._path = path
        self._index = self.Index(index_file_path(self._path), skip_warmup)

        if not skip_warmup:
            print_rank_0("    warming up data mmap file...")
            _warmup_mmap_file(data_file_path(self._path))
            
        print_rank_0("    creating numpy buffer of mmap...")
        self._bin_buffer_mmap = np.memmap(
            data_file_path(self._path), mode="r", order="C"
        )
        
        print_rank_0("    creating memory view of numpy buffer...")
        self._bin_buffer = memoryview(self._bin_buffer_mmap)

    def __del__(self):
        self._bin_buffer_mmap._mmap.close()
        del self._bin_buffer_mmap
        del self._index

    def __len__(self):
        return len(self._index)

    # @lru_cache(maxsize=8)
    def __getitem__(self, idx):
        if isinstance(idx, int):
            ptr, size = self._index[idx]
            np_array = np.frombuffer(
                self._bin_buffer, dtype=self._index.dtype, count=size, offset=ptr
            )
            return np_array
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            if step != 1:
                raise ValueError("Slices into indexed_dataset must be contiguous")
            ptr = self._index._pointers[start]
            sizes = self._index._sizes[idx]
            offsets = list(accumulate(sizes))
            total_size = sum(sizes)
            np_array = np.frombuffer(
                self._bin_buffer, dtype=self._index.dtype, count=total_size, offset=ptr
            )
            sents = np.split(np_array, offsets[:-1])
            return sents

    def get(self, idx, offset=0, length=None):
        """Retrieves a single item from the dataset with the option to only
        return a portion of the item.

        get(idx) is the same as [idx] but get() does not support slicing.
        """
        ptr, size = self._index[idx]
        if length is None:
            length = size - offset
        ptr += offset * np.dtype(self._index.dtype).itemsize
        np_array = np.frombuffer(
            self._bin_buffer, dtype=self._index.dtype, count=length, offset=ptr
        )
        return np_array

    @property
    def sizes(self):
        return self._index.sizes

    @property
    def doc_idx(self):
        return self._index.doc_idx

    def get_doc_idx(self):
        return self._index._doc_idx

    def set_doc_idx(self, doc_idx_):
        self._index._doc_idx = doc_idx_

    @property
    def supports_prefetch(self):
        return False

    @staticmethod
    def exists(path):
        return os.path.exists(index_file_path(path)) and os.path.exists(
            data_file_path(path)
        )
    

class MMapIndexedDatasetBuilder(object):
    def __init__(self, out_file, dtype=np.uint32):
        self._data_file = open(out_file, 'wb')
        self._dtype = dtype
        self._sizes = []
        self._doc_idx = [0]
        
    @property
    def dtype(self):
        return self._dtype
    
    def add_item(self, np_array: np.ndarray):
        
        assert np_array.dtype == self.dtype
        self._data_file.write(np_array.tobytes(order="C"))
        self._sizes.append(np_array.size)

    def end_document(self):
        self._doc_idx.append(len(self._sizes)) # 每篇文档结束的位置
    
    def merge_file_(self, another_file):
        # concatenate file
        pass
    
    def finalize(self, index_file):
        self._data_file.close()
        
        with MMapIndexedDataset.Index.writer(index_file, self.dtype) as index:
            index.write(self._sizes, self._doc_idx)