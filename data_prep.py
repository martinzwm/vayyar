#%%
import h5py
import numpy as np
from utilities import importDataOccupancyType
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import torch
from utilities import loadmat
from torchvision import transforms
import random


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
        sample = {'imagePower':image_power, 'label':label, 'path':self.path_label.iloc[idx, 0]}
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
def calculateMeanStd(rootDir):
    torch.manual_seed(0)
    transform = transforms.Compose([
                cropR(24),
            ])
    dataset = vCabDataSet(rootDir, transform)
    train_percent = 0.9
    validation_percent = 0.05
    testing_percent = 0.05
    total_num = len(dataset)
    training_num = int(train_percent * total_num)
    validation_num = int(validation_percent * total_num)
    testing_num = int(total_num - training_num - validation_num)
    train_set, val_set, test_set = random_split(dataset, [training_num, validation_num, testing_num])

    
    batch_size = 512
    dataset_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=8,
        shuffle=False
    )
    mean = 0.
    count_mean = 0
    for samples in dataset_loader:
        samples = samples['imagePower']
        mean += samples.mean()
        count_mean += 1
    mean /= count_mean

    var = 0
    count_var = 0
    for samples in dataset_loader:
        samples = torch.flatten(samples['imagePower'])
        var = torch.sum(torch.pow(torch.sub(samples, mean), 2)) / samples.size(0)
        count_var += 1
    var /= count_var
    std = torch.sqrt(var)
    print(f'mean: {mean}, standard deviation: {std}')
    return mean.item(), std.item()

def verifyNormalization(rootDir, image_mean, image_std):
    torch.manual_seed(0)
    transform = transforms.Compose([
            cropR(24),
            transforms.ToTensor(),
            transforms.Normalize(mean=[image_mean],
                                 std=[image_std])
        ])
    dataset = vCabDataSet(rootDir, transform=transform)
    subset_percent = 0.8
    subset_num = int(subset_percent * len(dataset))
    print(f'Tested on {subset_num} samples.')
    random.seed(1)
    subset_indices = random.sample(range(0, len(dataset)), subset_num)
    subset = Subset(dataset, subset_indices)
    
    train_percent = 0.9
    validation_percent = 0.05
    testing_percent = 0.05
    total_num = len(dataset)
    training_num = int(train_percent * total_num)
    validation_num = int(validation_percent * total_num)
    testing_num = int(total_num - training_num - validation_num)
    train_set, val_set, test_set = random_split(dataset, [training_num, validation_num, testing_num])

    
    batch_size = 512
    dataset_loader = DataLoader(
        subset,
        batch_size=batch_size,
        num_workers=8,
        shuffle=False
    )

    mean = 0.
    count_mean = 0
    for samples in dataset_loader:
        samples = samples['imagePower']
        mean += samples.mean()
        count_mean += 1
    mean /= count_mean
    
    var = 0
    count_var = 0
    for samples in dataset_loader:
        samples = torch.flatten(samples['imagePower'])
        var = torch.sum(torch.pow(torch.sub(samples, mean), 2)) / samples.size(0)
        count_var += 1
    var /= count_var
    std = torch.sqrt(var)
    print(f'mean: {mean}, standard deviation: {std}')



# %%
