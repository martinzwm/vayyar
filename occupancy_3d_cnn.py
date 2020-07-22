# %%
import os
import numpy as np
import json
import scipy.io as sio
import pickle
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import *
import h5py
from utilities import importDataFromMatFiles, loadData
from models import CNNModel

# %%
x = loadData("/Users/jameshe/Documents/radar_ura/vayyar/x.pickle")
y = loadData("/Users/jameshe/Documents/radar_ura/vayyar/y.pickle")
print(x.shape)
x_shape_dim0 = x.shape[0]
x_shape_dim1 = x.shape[1]
x_shape_dim2 = x.shape[2]
x_shape_dim3 = x.shape[3]
x = np.reshape(x, (x_shape_dim0, x_shape_dim1 * x_shape_dim2 * x_shape_dim3))
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)
x = np.reshape(x, (x_shape_dim0, x_shape_dim1, x_shape_dim2, x_shape_dim3))
#%%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

x_train = torch.from_numpy(x_train).float()
y_train = torch.from_numpy(y_train).float()
x_test = torch.from_numpy(x_test).float()
y_test = torch.from_numpy(y_test).float()
batch_size = 128 

# Pytorch train and test sets
train = torch.utils.data.TensorDataset(x_train,y_train)
test = torch.utils.data.TensorDataset(x_test,y_test)

# data loader
train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = True)
#%%
#Definition of hyperparameters

start = time.time()
num_classes = 5
num_epochs = 10
# Create CNN
model = CNNModel(num_classes)
model.train()
#model.cuda()
print(model)

# Binary Cross Entropy Loss for MultiLabel Classfication
error = nn.BCELoss()

learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# CNN model training
loss_list = []
iteration_count = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        train = Variable(images.view(len(images), 1, 29 ,29 ,64))
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
        loss_list.append(loss.data)
        iteration_count += 1
        print(f'Epoch {epoch + 1} Iteration {iteration_count}: loss = {loss.data}')
torch.save(model.state_dict(), "/Users/jameshe/Documents/radar_ura/vayyar/core_code/occupancy_3d_cnn_model.pt")
end = time.time()
print(f'duration = {end - start}s')
# %%
model = CNNModel(5)
model.load_state_dict(torch.load("/Users/jameshe/Documents/radar_ura/vayyar/core_code/occupancy_3d_cnn_model.pt"))
model.eval()

predictions = []
complete_labels = []
for images, labels in train_loader:
    train = Variable(images.view(len(images), 1, 29 ,29 ,64))
    outputs = model(train)
    outputs[outputs < 0.5] = 0
    outputs[outputs > 0.5] = 1
    predictions.append(outputs)
    complete_labels.append(labels)
from sklearn.metrics import classification_report
from sklearn.metrics import multilabel_confusion_matrix
cfm = multilabel_confusion_matrix(complete_labels[0].detach().numpy(), predictions[0].detach().numpy())
report = classification_report(complete_labels[0].detach().numpy(), predictions[0].detach().numpy())
print(cfm)
print(report)


# %%
