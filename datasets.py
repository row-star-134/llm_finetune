import torch
import pickle
import os
from torch.utils.data import IterableDataset

class CustomDataset(IterableDataset):
    def __init__(self,root_path, mode='train'):
        self.root_path = os.path.join(root_path, mode)

    def __iter__(self):
        for path in os.listdir(self.root_path):
            with open(os.path.join(self.root_path,path),'rb') as f:
                chunk = pickle.load(f)
                for sample in chunk:
                    yield sample

