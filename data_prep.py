#%%
import h5py
import numpy as np
from utilities import importDataOccupancyType
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
import torch
from utilities import loadmat
from torchvision import transforms


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

def calculateMeanStd(dataset):
    dataset = vCabDataSet('/home/vayyar_data/processed_vCab_Recordings')
    batch_size = 256
    dataset_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=8,
        shuffle=True
    )
    mean = 0.
    std = 0.
    count = 0
    for samples in dataset_loader:
        samples = samples['imagePower']
        mean += samples.mean()
        std += samples.std()
        count += 1

    mean /= len(dataset_loader.dataset)/count
    std /= len(dataset_loader.dataset)/count
    print(f'mean: {mean}, standard deviation: {std}')
    return mean, std

def verifyNormalization(dataset, image_mean, image_std):
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[image_mean],
                                 std=[image_std])
        ])
    dataset = vCabDataSet('/home/vayyar_data/processed_vCab_Recordings', transform=transform)
    subset_percent = 0.01
    subset_num = subset_percent * len(dataset)
    subset_indices = random.sample(range(0, len(dataset)), subset_num)
    subset = Subset(dataset, subset_indices)
    batch_size = 256
    dataset_loader = DataLoader(
        subset,
        batch_size=batch_size,
        num_workers=8,
        shuffle=True
    )
    mean = 0.
    std = 0.
    count = 0
    for samples in dataset_loader:
        samples = samples['imagePower']
        mean += samples.mean()
        std += samples.std()
        count += 1

    mean /= len(dataset_loader.dataset)/count
    std /= len(dataset_loader.dataset)/count
    print(f'mean: {mean}, standard deviation: {std}')
    return mean, std

