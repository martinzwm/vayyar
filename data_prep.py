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
    x, y, occupiedSeat, occupantType, path = importDataOccupancyType("/home/vayyar_data/FirstBatch")
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
        image_power = np.load(rfImagePath)
        if self.transform:
            image_power = self.transform(image_power)
        label = self.path_label.iloc[idx, 1]
        sample = {'image_power':image_power, 'label':label, 'path':rfImagePath}
        return sample

class cropR(object):
    """
    Crop the third dimension to be 24
    e.g. a rf image of shape 29x29x28 would be cropped to 29x29x24
    """
    def __init__(self, r_dimension):
        self.r_dimension = r_dimension

    def __call__(self, image_power):
        assert len(image_power.shape) == 3
        assert image_power.shape[-1] >= self.r_dimension, "Cannot crop image to a dimension that is greater than the original dimension, original dimension = {}".format(image_power.shape[-1])
        image_power = image_power[:,:,:self.r_dimension]
        return image_power
# %%
