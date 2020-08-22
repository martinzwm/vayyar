import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from utilities import loadmat, getPreprocessedRFImage


class SVM(nn.Module):
    """
    Linear Support Vector Machine
    -----------------------------
    This SVM is a subclass of the PyTorch nn module that
    implements the Linear  function. The  size  of  each 
    input sample is 2 and output sample  is 1.
    """
    def __init__(self):
        super(SVM, self).__init__()  # Call the init function of nn.Module
        self.fully_connected = nn.Linear(20184, 15)  # Implement the Linear function

    def forward(self, x):
        x = x.view(x.size(0), -1)
        fwd = self.fully_connected(x)  # Forward pass
        return fwd

class CNNModelRC(nn.Module):
    def __init__(self, num_classes):
        super(CNNModelRC, self).__init__()
        self.conv_layer1 = self._conv_layer_set(1, 32)
        self.conv_layer2 = self._conv_layer_set(32, 64)
        self.fc1 = nn.Linear(6400, 128)
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

class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv_layer1 = self._conv_layer_set(1, 32)
        self.conv_layer2 = self._conv_layer_set(32, 16)
        self.conv_layer3 = self._conv_layer_set(16, 16)
        self.fc1 = nn.Linear(1344, 2048)
        self.fc2 = nn.Linear(2048, num_classes)
        self.relu = nn.LeakyReLU()
        self.batch=nn.BatchNorm1d(2048)
        # self.drop=nn.Dropout(p=0.15) 
        self.sigmoid = nn.Sigmoid()               
        
    def _conv_layer_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(2, 2, 2), padding=0),
        nn.LeakyReLU(),
        nn.MaxPool3d((2, 2, 1)),
        )
        return conv_layer
    

    def forward(self, x):
        # Set 1
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.conv_layer3(out)
        out = out.view(out.size(0), -1) #Flatten it out
        out = self.fc1(out)
        out = self.relu(out)
        out = self.batch(out)
        # out = self.drop(out)
        out = self.fc2(out)
        #Apply sigmoid activation for the output layer
        out = self.sigmoid(out)

        return out