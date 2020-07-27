#%%
import h5py
import numpy as np
from utilities import importDataOccupancyType
import pandas as pd
import os
from torch.utils.data import Dataset
import torch
from utilities import loadmat
from torchvision import transforms


#%%
def firstBatchdataPrep():
    x, y, occupiedSeat, occupantType, path = importDataOccupancyType("/home/vayyar/FirstBatch")
    with h5py.File('training_dataset.hdf5', 'w') as f:
        f.create_dataset('x', data=x)
        f.create_dataset('y', data=y)
        f.create_dataset('occupiedSeat', data=occupiedSeat)
        f.create_dataset('occupantType', data=occupantType)
        f.create_dataset('path', data=path)

class vCabDataSet(Dataset):
    def __init__(self, rootDir, transform = None):
        self.rootDir = rootDir
        self.path_label = pd.read_pickle(os.path.join(rootDir, "path_label.pickle"))
        self.transform = transform

    def __len__(self):
        return len(self.path_label)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()
        rfImagePath = os.path.join(self.rootDir, str(self.path_label.iloc[idx, 2]) + '.npy')
        imagePower = np.load(rfImagePath)
        imagePower = np.squeeze(np.sum(imagePower, axis=2)) #TEMPORARY SOLUTION: Squeeze the 3D dataset into 2D 
        label = self.path_label.iloc[idx, 1]
        if self.transform:
            imagePower = np.squeeze(self.transform(imagePower))
        sample = {'imagePower':imagePower, 'label':label}
        return sample
# %%
