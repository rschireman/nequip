"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import pickle

import lmdb
from torch.utils.data import Dataset
from nequip.data import AtomicData, AtomicDataDict

class SinglePointLmdbDataset(Dataset):
    r"""Dataset class to load from LMDB files containing single point computations.
    Useful for Initial Structure to Relaxed Energy (IS2RE) task.

    Args:
        config (dict): Dataset configuration
        transform (callable, optional): Data transform function.
            (default: :obj:`None`)
    """

    def __init__(self, config, transform=None):
        super(SinglePointLmdbDataset, self).__init__()

        self.config = config

        self.db_path = self.config["src"]
        assert os.path.isfile(self.db_path), "{} not found".format(
            self.db_path
        )

        self.env = self.connect_db(self.db_path)

        self._keys = [
            f"{j}".encode("ascii") for j in range(self.env.stat()["entries"])
        ]
        self.transform = transform
        # this is a bit of hack for nequip
        self.fixed_fields = {"cell": []}

    def __len__(self):
        return len(self._keys)

    def __getitem__(self, idx):
        # Return features.
        datapoint_pickled = self.env.begin().get(self._keys[idx])
        data_object = pickle.loads(datapoint_pickled)
        data_object = (
            data_object
            if self.transform is None
            else self.transform(data_object)
        )
        
        # convert to nequip atomic data object
        atomic_data_obj = convert_ocp(data_object)

        return atomic_data_obj

    def connect_db(self, lmdb_path=None):
        env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,
        )
        return env

    def close_db(self):
        self.env.close()

def convert_ocp(data):
    data = AtomicData(
        pos=data.pos, 
        forces=data.force,
        total_energy=data.y_relaxed,
        edge_index=data.edge_index,
        edge_cell_shift=data.cell_offsets.float(),
        cell=data.cell,
        atomic_numbers=data.atomic_numbers.long(),
     )
    return data