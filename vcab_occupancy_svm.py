from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from utilities import loadmat, getPreprocessedRFImage, load_npy
import os
from models import vCabDataSet

#%% Import vCab_Recordings dataset
dataset = vCabDataSet('/home/vayyar_data/vCab_Recordings')

#%% Split training and testing dataset
train_percent = 0.9
validation_percent = 0.05
testing_percent = 0.05
total_num = len(dataset)
training_num = int(train_percent * total_num)
validation_num = int(validation_percent * total_num)
testing_num = int(total_num - training_num - validation_num)

train_set, val_set, test_set = random_split(dataset, [training_num, validation_num, testing_num])

#%% TEMPORARY SOLUTION: Squeeze the 3D dataset into 2D 

#%% Normalize and transform

#%% Training the SVM

#%% Making prediction and write misclassified samples to another file

#%% Error analysis