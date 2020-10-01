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
import sys
import matplotlib.pyplot as plt


def firstBatchdataPrep():
    '''
    One-time calling function
    This function is to store firstBatch dataset into a h5py file. 
    '''
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

def calculateMeanStd(rootDir):
    torch.manual_seed(0)
    transform = transforms.Compose([
                cropR(24),
            ])
    dataset = vCabDataSet(rootDir, transform)
    batch_size = 512
    dataset_loader = DataLoader(
        dataset,
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

def findMinAndMax(rootDir, normalize=False, image_mean=None, image_std=None):
    torch.manual_seed(0)
    if normalize == False:
        transform = transforms.Compose([
                cropR(24),
                transforms.ToTensor(),
            ])
    else:
        transform = transforms.Compose([
                cropR(24),
                transforms.ToTensor(),
                transforms.Normalize(mean=[image_mean],
                                 std=[image_std])
            ])

    dataset = vCabDataSet(rootDir, transform=transform)
    batch_size = 512
    dataset_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=8,
        shuffle=False
    )

    min_val = sys.maxsize
    max_val = -sys.maxsize
    for samples in dataset_loader:
        imagePower = samples['imagePower']
        sample_min = torch.min(imagePower)
        sample_max = torch.max(imagePower)
        if sample_max >8000:
            print(samples['path'])
            break
        if sample_min < min_val: min_val = sample_min
        if sample_max > max_val: max_val = sample_max
    print(f'min value: {min_val}, max value: {max_val}')

#FirstBatch min: 0, max: 0.04205722827464342
#Vcab_recordings min: 0, max: 8178.7294921875
def plotHistogram(data_min, data_max, num_bins, rootDir, graph_title, normalize=False, image_mean=None, image_std=None):
    torch.manual_seed(0)
    if normalize == False:
        transform = transforms.Compose([
                cropR(24),
                transforms.ToTensor(),
            ])
    else:
        transform = transforms.Compose([
                cropR(24),
                transforms.ToTensor(),
                transforms.Normalize(mean=[image_mean],
                                 std=[image_std])
            ])
    dataset = vCabDataSet(rootDir, transform=transform)
    batch_size = 512
    dataset_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=8,
        shuffle=False
    )

    mybins = np.linspace(data_min, data_max, num_bins)
    myhist = np.zeros(num_bins-1)
    for samples in dataset_loader:
        imagePower = torch.flatten(samples['imagePower']).detach().cpu().numpy()
        htemp, jnk = np.histogram(imagePower, mybins)
        myhist += htemp
    print(myhist)
    plt.hist(mybins[:-1], bins=mybins, weights=myhist)
    plt.yscale('log')
    plt.xlabel('power')
    plt.ylabel('count')
    plt.title(graph_title) 
    plt.show()
# plotHistogram(0, 0.05, 100, '/home/vayyar_data/processed_FirstBatch_nonthreshold', "FirstBatch Data Distribution No Threshold - 100 Bins")
# plotHistogram(0, 8200, 100, '/home/vayyar_data/processed_vCab_Recordings_nonthreshold', "vCab_Recordings Data Distribution No Threshold - 100 Bins")
# plotHistogram(-1, 3025, 100, '/home/vayyar_data/processed_FirstBatch_nonthreshold', "FirstBatch Data Distribution No Threshold (normalized) - 100 Bins", normalize=True,image_mean=1.655461726353112e-06,image_std=1.3920989854294221e-05)
# plotHistogram(-1, 1, 100, '/home/vayyar_data/processed_FirstBatch_nonthreshold', "FirstBatch Data Distribution No Threshold (normalized)- 100 Bins", normalize=True,image_mean=1.655461726353112e-06,image_std=1.3920989854294221e-05)
# plotHistogram(-1.5, 1340, 100, '/home/vayyar_data/processed_vCab_Recordings_nonthreshold/', "vCab_Recordings Data Distribution No Threshold (normalized)- 100 Bins", normalize=True,image_mean=7.608346462249756,image_std=6.12775993347168)
# plotHistogram(-1, 1, 100, '/home/vayyar_data/processed_vCab_Recordings_nonthreshold/', "vCab_Recordings Data Distribution No Threshold (normalized)- 100 Bins", normalize=True,image_mean=7.608346462249756,image_std=6.12775993347168)



# %%
