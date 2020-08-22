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
from utilities import importDataFromMatFiles, loadData, scenarioWiseTransformLabels
from models import CNNModel
from torchvision import transforms
from data_prep import vCabDataSet, cropR
import pkbar
import math
from torch.utils.tensorboard import SummaryWriter
import argparse

#%%
writer = SummaryWriter()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_name', help='name of the model', required=True)
parser.add_argument('-d', '--mis_data_filename', help='misclassified data name', required=True,)

args = vars(parser.parse_args())
model_name = args['model_name']
misclassified_filename = args['mis_data_filename']
#%% Import dataset

#first batch
# transform = transforms.Compose([
#             cropR(24),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[1.655461726353112e-06],
#                                  std=[1.3920989854294221e-05])
#         ])
# dataset = vCabDataSet('/home/vayyar_data/processed_FirstBatch', transform)

#vcab_recordings
# transform = transforms.Compose([
#             cropR(24),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[-0.05493089184165001],
#                                  std=[0.035751599818468094])
#         ])
# dataset = vCabDataSet('/home/vayyar_data/processed_vCab_Recordings_clutter_removed', transform)

transform = transforms.Compose([
            cropR(24),
            transforms.ToTensor(),
            transforms.Normalize(mean=[7.608346462249756],
                                 std=[6.12775993347168])
        ])
dataset = vCabDataSet('/home/vayyar_data/processed_vCab_Recordings', transform)


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
# model_name = "/home/vayyar_model/vCab_Recordings_cnn_20200807.pt"
# model_name = "/home/vayyar_model/first_batch_cnn_20200807.pt"
num_classes = 15
num_epochs = 10
# Create CNN
model = CNNModel(num_classes)
model.train()
#model.cuda()
print(model)

# Binary Cross Entropy Loss for MultiLabel Classfication
error = nn.BCELoss()
# learning_rate = 0.001 #FirstBatch
# learning_rate = 0.0001 #Vcab_Recordings with clutter removal
learning_rate = 0.00005 #Vcab_Recordings
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#%%
# CNN model training
loss_list = []
val_loss_list = []
iteration_count = 0
train_per_epoch = math.ceil(len(train_set) / batch_size)
val_per_epoch = math.ceil(len(val_set) / batch_size)

for epoch in range(num_epochs):
    sum_loss = 0
    sum_val_loss = 0
    kbar = pkbar.Kbar(target=train_per_epoch, width=8)
    for i, sample in enumerate(train_loader):
        x_train = sample["imagePower"].float().to(device)
        y_train = sample["label"].float().to(device)
        x_train = Variable(x_train.view(len(x_train), 1, 29 ,29 ,24))
        y_train = Variable(y_train)
        # Clear gradients
        optimizer.zero_grad()
        # Forward propagation
        outputs = model(x_train)
        # Calculate softmax and ross entropy loss
        loss = error(outputs, y_train)
        # Calculating gradients
        loss.backward()
        # Update parameters
        optimizer.step()
        # Write to tensorboard
        writer.add_scalar("Loss/train", loss, epoch)
        sum_loss += loss.data
        # loss_list.append(loss.data)
        iteration_count += 1
        kbar.update(i, values=[("loss", loss)])
        # print(f'Epoch {epoch + 1} Iteration {iteration_count}: loss = {loss.data}')
    loss_list.append(sum_loss/train_per_epoch)
    with torch.no_grad():
        for val_batch in val_loader:
            
            x_val = val_batch['imagePower'].float().to(device)
            y_val = val_batch['label'].float().to(device)
            x_val = Variable(x_val.view(len(x_val), 1, 29 ,29 ,24))
            y_val = Variable(y_val)

            model.eval()

            y_val_pred = model(x_val)
            val_loss = error(y_val_pred, y_val) 
            # Write to tensorboard
            writer.add_scalar("Loss/validation", val_loss, epoch)
            sum_val_loss += val_loss.data
            # val_loss_list.append(val_loss.data)
    val_loss_list.append(sum_val_loss/val_per_epoch)
    kbar.add(1, values=[("loss", loss), ("val_loss", val_loss)])
    print('done validation')
    print("Epoch {}, Loss: {}".format(epoch+1, sum_loss/train_per_epoch))
    print("Epoch {}, Val Loss: {}".format(epoch+1, sum_val_loss/val_per_epoch))
writer.flush()
writer.close()
#%%
torch.save(model.state_dict(), model_name)
end = time.time()
print(f'duration = {end - start}s')
# %%
model = CNNModel(num_classes)
model.load_state_dict(torch.load(model_name))
model.eval()

test_per_epoch = math.ceil(len(test_set) / batch_size)
accuracy = []

f1 = open('all_cnn_arch1.csv', 'w')
f2 = open(misclassified_filename, 'w')
f1.write(','.join(['path', 'label_seat', 'predicted_seat', 'label_type', 'predicted_type', 'seat_prediction_result', 'type_prediction_result\n']))
f2.write(','.join(['path', 'label_seat', 'predicted_seat', 'label_type', 'predicted_type', 'seat_prediction_result', 'type_prediction_result\n']))
f1.close()
f2.close()
for sample in test_loader:
    x_test = sample["imagePower"].float().to(device)
    y_test = sample["label"].float().to(device)
    path = np.array(sample['path'])    
    train = Variable(x_test.view(len(x_test), 1, 29 ,29 ,24))
    outputs = model(train)
    outputs[outputs < 0.5] = 0
    outputs[outputs > 0.5] = 1
    outputs = outputs.detach().numpy()
    y_test = y_test.detach().numpy()
    accuracy.append(accuracy_score(y_test, outputs))
    test_dict = {
        'path': [],
        'label_seat': [],
        'predicted_seat':[],
        'label_type': [],
        'predicted_type': []
    }
    test_dict['path'] = list(path)
    test_dict['predicted_seat'], test_dict['predicted_type'] = scenarioWiseTransformLabels(outputs)
    test_dict['label_seat'], test_dict['label_type'] = scenarioWiseTransformLabels(y_test)
    df = pd.DataFrame.from_dict(test_dict)
    df['seat_prediction_result'] = np.where(df['label_seat'] == df['predicted_seat'], True, False)
    df['type_prediction_result'] = np.where(df['label_type'] == df['predicted_type'], True, False)
    mis_df = df.loc[(df['seat_prediction_result'] == False) | (df['type_prediction_result'] == False)]
    df.to_csv('all_cnn.csv', mode='a', header=False, index=False)
    mis_df.to_csv(misclassified_filename, mode='a', header=False, index=False)
acc = np.average(np.array(accuracy))
print(f'Testing accuracy is {acc}.')

# %%
