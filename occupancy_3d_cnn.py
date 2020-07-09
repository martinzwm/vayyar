# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
import numpy as np
import json
import scipy.io as sio
from sklearn.preprocessing import MultiLabelBinarizer
import pickle


# %%
import pandas as pd
from tqdm import tqdm

# for reading and displaying images
import matplotlib.pyplot as plt

# for creating validation set
from sklearn.model_selection import train_test_split
# for evaluating the model
from sklearn.metrics import accuracy_score


# PyTorch libraries and modules
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import *
import h5py

# %% [markdown]
# Import data

# %%
def importDataFromMatFiles(rootDir):
    """
    Param: rootDir: The parent directory to the directories that start with Copy ....
                    For example, in this case, rootDir = "/Users/jameshe/Documents/radar_ura/FirstBatch"
    """
    xList = list()
    yList = list()
    mlb = MultiLabelBinarizer()
    for f in os.scandir(rootDir):
        if f.is_dir():
            with open(os.path.join(f.path, "test_data.json")) as labelFile:
                labels = json.load(labelFile)
                occupancyLabel = labels["Occupied_Seats"]
            for file in os.scandir(os.path.join(f.path, "SavedVars_RF_Data")):
                if '.mat' in file.name:
                    frame = sio.loadmat(file)
                    imagePower = np.absolute(np.power(frame["Image"],2))
                    imageMaxPower = np.max(imagePower)
                    maskG = frame["Mask"].astype(bool)
                    allowedDropRelPeak = 5
                    maskT = (imagePower >= imageMaxPower/allowedDropRelPeak)
                    imagePower[~ (maskG & maskT)] = 0
                    xList.append(imagePower)
                    yList.append(occupancyLabel)
    yList = mlb.fit_transform(yList)
    xList = np.array(xList)
    return (xList, yList)


# %%
def saveData(data, path, fileName):
    """
    open file mode w: write, if the file exist, erase it
    open file mode b: open the file as a binary file
    """
    filePath = os.path.join(path, fileName)
    with open(filePath,'wb') as pickleFileHandle:
        pickle.dump(data, pickleFileHandle)
        pickleFileHandle.close()


# %%
def loadData(path, fileName):
    """
    open file mode b: open the file as a binary file
    open file mode r: read file
    """
    filePath = os.path.join(path, fileName)
    with open(filePath, 'rb') as pickleFileHandle:
        data = pickle.load(pickleFileHandle)
        return data


# %%
x, y = importDataFromMatFiles("/Users/jameshe/Documents/radar_ura/vayyar/FirstBatch")


# %%
x.shape


# %%
saveData(x, './', 'x.pickle')
saveData(y, './', 'y.pickle')


# %%
#https://towardsdatascience.com/pytorch-step-by-step-implementation-3d-convolution-neural-network-8bf38c70e8b3
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

x_train = torch.from_numpy(x_train).float()
y_train = torch.from_numpy(y_train).float()
x_test = torch.from_numpy(x_test).float()
y_test = torch.from_numpy(y_test).float()
#x_train.shape == 4192, 131 * 2^5 = 4192
batch_size = 131 #We pick beforehand a batch_size that we will use for the training


# Pytorch train and test sets
train = torch.utils.data.TensorDataset(x_train,y_train)
test = torch.utils.data.TensorDataset(x_test,y_test)

# data loader
train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = True)


# %%
x_test.shape


# %%
num_classes = 5

# Create CNN Model
class CNNModel(nn.Module):
    def __init__(self):
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

#Definition of hyperparameters
n_iters = 4500
num_epochs = n_iters / (len(x_train) / batch_size)
num_epochs = int(num_epochs)
# Create CNN
model = CNNModel()
#model.cuda()
print(model)

# Binary Cross Entropy Loss for MultiLabel Classfication
error = nn.BCELoss()

# SGD Optimizer
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# %%
# CNN model training
count = 0
loss_list = []
iteration_list = []
accuracy_list = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        
        train = Variable(images.view(batch_size, 1, 29 ,29 ,64))
        labels = Variable(labels)
        # Clear gradients
        optimizer.zero_grad()
        # Forward propagation
        outputs = model(train)
        # Calculate softmax and ross entropy loss
        loss = error(outputs, labels)
        # Calculating gradients
        loss.backward()
        # Update parameters
        optimizer.step()
        
        count += 1
        #32 batches
        if count % 32 == 0:
            # Calculate Accuracy         
            correct = 0
            total = 0
            # Iterate through test dataset
            for images, labels in test_loader:
                
                test = Variable(images.view(len(images), 1, 29, 29 ,64))
                # Forward propagation
                outputs = model(test)
                # Get predictions from the maximum value
                predicted = outputs
                predicted[predicted < 0.5] = 0
                predicted[predicted > 0.5] = 1
               
                # Total number of labels
                total += (len(labels) * num_classes)
                correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / float(total)
            # store loss and iteration
            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)
            # Print Loss
            print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(count, loss.data, accuracy))



