import h5py
import torch
import numpy as np
from typing import Dict, List, Sequence, Tuple, Union
from torch.utils.data.dataloader import DataLoader, default_collate
from glob import glob
from re import L

class AMIDEDataset:
    def __init__(self, data: Union[str, Dict[str, np.ndarray]] = None):
        self._data = data if data is not None else dict()
        if len(self._data):
            self.load_dict(self._data)

    def __getitem__(self, key):
        if isinstance(key, int):
            key = self.keys()[key]
        # print(key)
        data = self._data[key]
        
        # If in loader mode, filter data based on x and y parameters
        if hasattr(self, 'loader_mode') and self.loader_mode:
            filtered_data = {}
            
            # Add input features (x)
            if hasattr(self, 'x') and self.x:
                for feature in self.x:
                    if feature in data:
                        filtered_data[feature] = data[feature]
            
            # Add target features (y)
            if hasattr(self, 'y') and self.y:
                for feature in self.y:
                    if feature in data:
                        filtered_data[feature] = data[feature]
            
            return filtered_data
        
        return data

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise ValueError(
                f'Failed to set key of type {type(key)}, expected str.')
        self._data[key] = value

    def __delitem__(self, key):
        del self._data[key]

    def __contains__(self, key):
        return key in self._data

    def __len__(self):
        return len(self._data)

    def load_dict(self, data):
        for k,v in data.items():
            self[k] = v
        return

    def items(self):
        return [(k, self[k]) for k in self.keys()]

    def values(self):
        return [self[k] for k in self.keys()]

    def keys(self):
        return sorted(self._data.keys())

    def pop(self, key):
        return self._data.pop(key)

    def merge(self, merge_ds):
        for k,v in merge_ds.items():
            self[k] = v
        return 

    def save_h5(self, filename):
        with h5py.File(filename, 'w') as f:
            for n, g in self.items():
                h5g = f.create_group(n)
                for k, v in g.items():
                    h5g.create_dataset(k, data=v)
                    
    def load_h5(self, data, keys=None, shard: Tuple[int, int] = None):
        with h5py.File(data, 'r') as f:
            for k, g in f.items():
                s = slice(shard[0], None, shard[1]) if shard is not None else slice(None)
                keys = g.keys()
                data = dict((k, v[s]) for k, v in g.items() if k in keys)
                self[k] = data
                
    def get_loader(self, sampler, x, y=None, **loader_kwargs):
        self.loader_mode = True
        self.x = x
        self.y = y or {}

        loader = DataLoader(self, batch_sampler=sampler, **loader_kwargs)
        '''
        def _squeeze(t):
            for d in t:
                for v in d.values():
                    v.squeeze_(0)
            return t
        loader.collate_fn = lambda x: _squeeze(default_collate(x))
        '''
        return loader

    #### for run_ann_model.py
    # def get_loader(self, x, y=None, batch_size=1, shuffle=False, **loader_kwargs):
    #     self.loader_mode = True
    #     self.x = x
    #     self.y = y or {}

    #     # The collate_fn is important for custom dictionary-based datasets
    #     def collate_fn(batch):
    #         # batch is a list of dictionaries
    #         keys = batch[0].keys()
    #         collated = {key: default_collate([d[key] for d in batch]) for key in keys}
    #         return collated

    #     loader = DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, **loader_kwargs)
    #     '''
    #     def _squeeze(t):
    #         for d in t:
    #             for v in d.values():
    #                 v.squeeze_(0)
    #         return t
    #     loader.collate_fn = lambda x: _squeeze(default_collate(x))
    #     '''
    #     return loader


class NamedSampler:
    def __init__(self, ds: AMIDEDataset,
                 shuffle=True, 
                 batches_per_epoch=-1):
        self.ds = ds
        self.batch_size = 1
        self.shuffle = shuffle
        self.batches_per_epoch = batches_per_epoch

    def __len__(self):
        if self.batches_per_epoch > 0:
            return self.batches_per_epoch
        else:
            return sum(self._get_num_batches_for_group(g) for g in self.ds.groups)

    def __iter__(self):
        return iter(self._samples_list())

    def _get_num_batches_for_group(self, g):
        return int(np.ceil(len(g) / self.batch_size))

    def _samples_list(self):
        samples = list()
        for group_key, g in self.ds.items():
            n = len(g)
            if n == 0:
                continue
            idx = np.arange(n)
            if self.shuffle:
                np.random.shuffle(idx)
            n_batches = self._get_num_batches_for_group(g)
            samples.extend(([group_key]) for idx_batch in np.array_split(idx, n_batches))
            #samples.extend(((group_key, idx_batch),) for idx_batch in np.array_split(idx, n_batches))
        if self.shuffle:
            np.random.shuffle(samples)
            
        return samples
        '''
        
        if self.batches_per_epoch:
            if len(samples) > self.batches_per_epoch:
                samples = samples[:self.batches_per_epoch]
            else:
                idx = np.arange(len(samples))
                np.random.shuffle(idx)
                n = self.batches_per_epoch - len(samples)
                samples.extend([samples[i] for i in np.random.choice(idx, n, replace=True)])
        print("2", samples, len(samples))
        return samples
        '''
