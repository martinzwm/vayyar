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
from torch.utils.data import Dataset, DataLoader, random_split
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import *
from utilities import importDataFromMatFiles, loadData
from models import CNNModel
from torchvision import transforms
from data_prep import vCabDataSet, cropR

#%% Import FirstBatch dataset
transform = transforms.Compose([
            cropR(24),
            transforms.ToTensor(),
            transforms.Normalize(mean=[1.655461726353112e-06],
                                 std=[1.3920989854294221e-05])
        ])
dataset = vCabDataSet('/home/vayyar_data/processed_FirstBatch', transform)

#%% Split training and testing dataset
train_percent = 0.9
validation_percent = 0.05
testing_percent = 0.05
total_num = len(dataset)
training_num = int(train_percent * total_num)
validation_num = int(validation_percent * total_num)
testing_num = int(total_num - training_num - validation_num)

train_set, val_set, test_set = random_split(dataset, [training_num, validation_num, testing_num])

# %%
batch_size = 512
train_loader = DataLoader(
    train_set,
    batch_size=batch_size,
    num_workers=8,
    shuffle=True
)
val_loader = DataLoader(
    val_set,
    batch_size=batch_size,
    num_workers=8,
    shuffle=True
)
test_loader = DataLoader(
    test_set,
    batch_size=batch_size,
    num_workers=8,
    shuffle=True
)
print("finished loaders")
#%%
#Definition of hyperparameters
start = time.time()
model_name = "/home/vayyar_model/first_batch_cnn_20200807.pt"
num_classes = 15
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
#%%
# CNN model training
loss_list = []
iteration_count = 0
for epoch in range(num_epochs):
    for i, (sample) in enumerate(train_loader):
        images = sample["imagePower"].float()
        labels = sample["label"].float()
        train = Variable(images.view(len(images), 1, 29 ,29 ,24))
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
torch.save(model.state_dict(), model_name)
end = time.time()
print(f'duration = {end - start}s')
# %%
model = CNNModel(num_classes)
model.load_state_dict(torch.load(model_name))
model.eval()

predictions = []
complete_labels = []
for sample in train_loader:
    images = sample["imagePower"].float()
    labels = sample["label"].float()
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
