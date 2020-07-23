#%%
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from utilities import loadmat, getPreprocessedRFImage

class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv_layer1 = self._conv_layer_set(1, 32)
        self.conv_layer2 = self._conv_layer_set(32, 64)
        self.fc1 = nn.Linear(22400, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.LeakyReLU()
        self.batch=nn.BatchNorm1d(128)
        self.drop=nn.Dropout(p=0.15) 
        self.sigmoid = nn.Sigmoid()               
        
    def _conv_layer_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=0),
        nn.LeakyReLU(),
        nn.MaxPool3d((2, 2, 2)),
        )
        return conv_layer
    

    def forward(self, x):
        # Set 1
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = out.view(out.size(0), -1) #Flatten it out
        out = self.fc1(out)
        out = self.relu(out)
        out = self.batch(out)
        out = self.drop(out)
        out = self.fc2(out)
        #Apply sigmoid activation for the output layer
        out = self.sigmoid(out)

        return out

class vCabDataSet(Dataset):
    def __init__(self, rootDir):
        self.rootDir = rootDir
        self.path_label = pd.read_pickle(os.path.join(rootDir, "path_label.pickle"))
        print(self.path_label.head())
    def __len__(self):
        return len(self.path_label)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()
        #print(self.path_label)
        rfImagePath = os.path.join(self.rootDir, self.path_label.iloc[idx, 0])
        rfImageStruct = loadmat(rfImagePath)['rfImageStruct']
        imagePower = getPreprocessedRFImage(rfImageStruct)
        label = self.path_label.iloc[idx, 1]
        sample = {'imagePower':imagePower, 'label':label}
        return sample

dataset = vCabDataSet('/home/vayyar_data/vCab_Recordings')
#%%
for i in range(len(dataset)):
    sample = dataset[i]
    print(sample['label'])
    print(type(sample['label']))
    print(dataset.path_label.iloc[i,0])
    print(sample['label'].dtype)
    print(sample['imagePower'].dtype)
    break

# %%
