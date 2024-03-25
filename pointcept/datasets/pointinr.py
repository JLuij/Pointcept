"""
ModelNet40 Dataset

get sampled point clouds of ModelNet40 (XYZ and normal from mesh, 10k points per shape)
at "https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip"

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import glob
import os
import numpy as np
from torch.utils.data import Dataset
import torch
from pathlib import Path

from pointcept.utils.logger import get_root_logger
from .builder import DATASETS
from .transform import Compose


@DATASETS.register_module()
class PointINRDataSet(Dataset):
    def __init__(
        self,
        split="train",
        data_root="data/pointINR",
        class_names=None,
        transform=None,
        test_mode=False,
        test_cfg=None,
        cache_data=False,
        loop=1,
    ):
        super(PointINRDataSet, self).__init__()
        self.data_root = data_root
        self.class_names = dict(zip(class_names, range(len(class_names))))
        self.split = split
        self.transform = Compose(transform)
        self.loop = (
            loop if not test_mode else 1
        )  # force make loop = 1 while in test mode
        self.cache_data = cache_data
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None
        self.cache = {}
        if test_mode:
            # TODO: Optimize
            pass

        self.data_list = self.get_data_list()
        logger = get_root_logger()
        logger.info(
            "Totally {} x {} samples in {} set.".format(
                len(self.data_list), self.loop, split
            )
        )

    def get_data_list(self):
        assert isinstance(self.split, str)
        
        # train = 'test' if self.split == 'val' else 'train'
        # print(f'!!! path {os.path.join(self.data_root, f"{train}_*.pth")}')
        # data_list = list(glob.glob(os.path.join(self.data_root, train, "_*.pth")))

        print(f'!!! {Path(self.data_root) / f"{self.split}_*.pt"}')
        data_list = list(Path(self.data_root).glob(f'{self.split}_*.pt'))
        
        return data_list

    def get_data(self, idx):
        data_idx = idx % len(self.data_list)
        if self.cache_data:
            return self.cache[data_idx]
        else:
            # data_shape = "_".join(self.data_list[data_idx].split("_")[0:-1])
            # data_path = os.path.join(
            #     self.data_root, data_shape, self.data_list[data_idx] + ".txt"
            # )
            # data = np.loadtxt(data_path, delimiter=",").astype(np.float32)
            # coord, normal = data[:, 0:3], data[:, 3:6]
            # category = np.array([self.class_names[data_shape]])
            path = Path(self.data_list[data_idx])
            state_dict = torch.load(path, map_location='cpu')
            
            num_points = state_dict['encoder']['coords.weight'].shape[0]
            coords_3d = torch.cat([state_dict['encoder']['coords.weight'], torch.zeros(num_points, 1)], dim=1)
            
            label = int(path.stem.split('_')[1])
            
            pointINR = {
                'category': label,
                'coord': coords_3d,
                'feats': state_dict['encoder']['feature_vecs.weight'],
            }
            # A shared decoder was not provided so use this
            #   pointINR's decoder
            # if self.config['shared_decoder_path'] is None:
            #     pointINR['decoder'] = state_dict['decoder']
            
            if self.cache_data:
                self.cache[data_idx] = pointINR
                
            return pointINR

    def prepare_train_data(self, idx):
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        return data_dict

    def prepare_test_data(self, idx):
        assert idx < len(self.data_list)
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        return data_dict

    def get_data_name(self, idx):
        data_idx = idx % len(self.data_list)
        return self.data_list[data_idx]

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        else:
            return self.prepare_train_data(idx)

    def __len__(self):
        return len(self.data_list) * self.loop
