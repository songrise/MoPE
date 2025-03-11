# -*- coding : utf-8 -*-
# @FileName  : dataset_downsample.py
# @Author    : Ruixiang JIANG (Songrise)
# @Time      : Nov 04, 2023
# @Github    : https://github.com/songrise
# @Description: downsample the dataset to a smaller size
import torch.nn as nn
import torch
from torch.utils.data import Dataset
import numpy as np
import os

class LowShotDataset(Dataset):
    def __init__(self, dataset, n_shot):
        self.dataset = dataset
        self.n_shot = n_shot
        self.original_length = len(dataset)
        self.select_idx = np.random.choice(self.original_length, self.n_shot, replace=False)

    def __getitem__(self, index):
        return self.dataset[self.select_idx[index]]
    def __len__(self):
        return self.n_shot
    
