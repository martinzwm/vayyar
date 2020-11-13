# from LSTM_Vayyar import *
import pandas as pd
import numpy as np
import os
import pickle
import torch
import json
import copy
from utilities import loadmat
from torch.utils.data import Dataset

import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.autograd import Variable
import torch.nn as nn
from torch.optim import *
from utilities import importDataFromMatFiles, loadData, scenarioWiseTransformLabels, getConfusionMatrices, seatWiseTransformLabels, plot_seat_wise_bar, multiclass_metric
# from torchvision import transforms
from data_prep import rfImageDataSet, cropR
#import pkbar
import math
from torch.utils.tensorboard import SummaryWriter
# import argparse
# import seaborn as sn

class TrainDataSet(Dataset):
    # Allows user to create a dataset with multiple directories
    def __init__(self, rootDir, transform=None):
        # rootDir is a list of directories
        self.rootDir = rootDir
        self.path_labels = []
        self.dataset_sizes = [] # list of num of samples in each individual dataset
        self.grand_dataset_size = 0 # total num of samples
        for i in range(len(rootDir)):
            dataset_label = pd.read_pickle(os.path.join(rootDir[i], 'path_label.pickle'))
            dataset_size = len(dataset_label)
            self.path_labels.append(dataset_label)
            self.dataset_sizes.append(dataset_size+self.grand_dataset_size)
            self.grand_dataset_size += dataset_size
        self.transform = transform

    def __len__(self):
        return self.grand_dataset_size
    
    def __getitem__(self, idx):
        # Pick the corresponding datasets
        for i in range(len(self.dataset_sizes)):
            if idx < self.dataset_sizes[i]:
                dataset_num = i
                if dataset_num != 0:
                    idx -= self.dataset_sizes[dataset_num-1] # set it to the idx of the corresponding dataset
                break
        # Load data
        rfImagePath = os.path.join(self.rootDir[dataset_num], str(self.path_labels[dataset_num].iloc[idx, 3]) + '.npy')
        image_power = np.load(rfImagePath)
        if self.transform:
            image_power = self.transform(image_power)
        label = self.path_labels[dataset_num].iloc[idx, 1]
        sample = {
            'imagePower':image_power, 
            'label':label,
            'path':self.path_labels[dataset_num].iloc[idx, 0],
            'npy_id':self.path_labels[dataset_num].iloc[idx, 3],
            'car_info':self.path_labels[dataset_num].iloc[idx, 5],
            'car_model':self.path_labels[dataset_num].iloc[idx,4]
            }
        return sample
    
    def class_distribution(self):
        class_count = np.zeros((32))
        for i in range(self.grand_dataset_size):
            sample = self.__getitem__(i)
            sample_class = sample['label'][0]
            class_count[sample_class] += 1
        return class_count

    def mean_and_std(self):
        mean, std = 0, 0
        for i in range(self.grand_dataset_size):
            sample = self.__getitem__(i)
            image = sample['imagePower']
            mean += image.mean()
            std += image.std()
        return mean/self.grand_dataset_size, std/self.grand_dataset_size

class CNNModelRC(nn.Module):
    def __init__(self, num_classes):
        super(CNNModelRC, self).__init__()
        self.conv_layer1 = self._conv_layer_set(1, 32)
        self.conv_layer2 = self._conv_layer_set(32, 64)
        self.fc1 = nn.Linear(64*7*7*6, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.LeakyReLU()
        # self.batch=nn.BatchNorm1d(128)
        self.drop=nn.Dropout(p=0.15) 
        self.softmax=nn.LogSoftmax(dim=1)        
        
    def _conv_layer_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=(1,1,1)),
        nn.LeakyReLU(),
        nn.MaxPool3d((2, 2, 2)),
        )
        return conv_layer
    
    def forward(self, x):
        # Set 1
        out = self.conv_layer1(x)
        out = self.drop(out)
        out = self.conv_layer2(out)
        out = out.view(out.size(0), -1) #Flatten it out
        out = self.fc1(out)
        out = self.relu(out)
        # out = self.batch(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out

def train():
    # # For local
    # train_dataset = TrainDataSet([
    #     r'B:\Vayyar_Dataset\small_data\for_martin_vcab\ford1',
    #     r'B:\Vayyar_Dataset\small_data\for_martin_vcab\ford2'
    # ])

    # val_dataset = TrainDataSet([
    #     r'B:\Vayyar_Dataset\small_data\for_martin_vcab\ford1_center'
    # ])

    # For remote
    train_dataset = TrainDataSet([
        '/data/for_martin_vcab/ford1',
        '/data/for_martin_vcab/ford2'
    ])

    val_dataset = TrainDataSet([
        '/data/for_martin_vcab/ford1_center'
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device == torch.device('cuda'):
        batch_size = 256
    else:
        batch_size = 64

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=1,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=1,
        shuffle=True
    )

    # Build model
    num_classes = 32
    model = CNNModelRC(num_classes).to(device)
    # model = torch.load('lean_cnn.pt').to(device)
    model.train() # mode = train
    print(model)

    error = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # CNN model training
    num_epochs = 2
    loss_list, val_loss_list, val_acc_list = [], [], []
    iteration_count = 0
    train_per_epoch = math.ceil(len(train_dataset) / batch_size)
    val_per_epoch = math.ceil(len(val_dataset) / batch_size)

    start = time.time()
    for epoch in range(num_epochs):
        sum_loss = 0
        sum_val_loss = 0
        sum_val_acc = 0
        for i, sample in enumerate(train_loader):
            model.train()
            x_train = sample["imagePower"].float().to(device)
            y_train = sample["label"].view(len(x_train),).long().to(device)
            x_train = Variable(x_train.view(len(x_train), 1, 29 ,29 ,24))
            y_train = Variable(y_train)
            # Forward propagation
            outputs = model(x_train)
            # Calculate softmax and ross entropy loss
            loss = error(outputs, y_train)
            # Clear gradients
            optimizer.zero_grad()
            # Calculating gradients
            loss.backward()
            # Update parameters
            optimizer.step()
            sum_loss += loss.data
            iteration_count += 1
        if device == torch.device('cuda'):
            sum_loss = sum_loss.cpu()
        loss_list.append(sum_loss/train_per_epoch)
        with torch.no_grad():
            for val_batch in val_loader:
                x_val = val_batch['imagePower'].float().to(device)
                y_val = val_batch['label'].view(len(x_val),).long().to(device)
                x_val = Variable(x_val.view(len(x_val), 1, 29 ,29 ,24))
                y_val = Variable(y_val)

                model.eval()
                y_val_pred = model(x_val).detach()
                val_loss = error(y_val_pred, y_val) 
                sum_val_loss += val_loss.data
                if device == torch.device('cuda'):
                    sum_val_acc += accuracy_score(y_val_pred.cpu().numpy().argmax(axis=1), y_val.detach().cpu().numpy())
                else:
                    sum_val_acc += accuracy_score(y_val_pred.numpy().argmax(axis=1), y_val.detach().numpy())

        if len(val_loss_list) == 0 or sum_val_loss/val_per_epoch < min(val_loss_list):
            torch.save(model, 'lean_cnn.pt')
        if device == torch.device('cuda'):
            sum_val_loss = sum_val_loss.cpu()
        val_loss_list.append(sum_val_loss/val_per_epoch)
        val_acc_list.append(sum_val_acc/val_per_epoch)
        print('done validation')
        print("Epoch {}, Loss: {}".format(epoch+1, sum_loss/train_per_epoch))
        print("Epoch {}, Val Loss: {}".format(epoch+1, sum_val_loss/val_per_epoch))
        print("Epoch {}, Val Accuracy: {}".format(epoch+1, sum_val_acc/val_per_epoch))

    end = time.time()
    print(f'duration = {end - start}s')

    # Save train loss and val loss profile
    train_loss_profile = {
        'train_loss': np.array(loss_list),
        'val_loss': np.array(val_loss_list),
        'val_acc': np.array(val_acc_list)
    }
    df = pd.DataFrame.from_dict(train_loss_profile)
    df.to_pickle('train_loss_profile.pickle')

def test():
    test_dataset = TrainDataSet([
        # r'B:\Vayyar_Dataset\small_data\for_martin_vcab\ford2_center'
        '/data/for_martin_vcab/ford2_center'
    ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load('lean_cnn.pt').to(device)
    if device == torch.device('cuda'):
        batch_size = 256
    else:
        batch_size = 64

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=1,
        shuffle=True
    )
    accuracy_list = []
    for test_batch in test_loader:
        x_test = test_batch['imagePower'].float().to(device)
        y_test = test_batch['label'].view(len(x_test),).long().to(device)
        x_test = Variable(x_test.view(len(x_test), 1, 29 ,29 ,24))
        y_test = Variable(y_test)

        model.eval()
        y_test_pred = model(x_test).detach()
        if device == torch.device('cuda'):
            accuracy = accuracy_score(y_test_pred.cpu().numpy().argmax(axis=1), y_test.cpu().detach().numpy())
        else:
            accuracy = accuracy_score(y_test_pred.numpy().argmax(axis=1), y_test.detach().numpy())
        accuracy_list.append(accuracy)
    print(np.mean(accuracy_list))
    
if __name__ == '__main__':
    train()
    profile = pd.read_pickle('train_loss_profile.pickle')
    print(profile)
            




# train_dataset1 = rfImageDataSet(r'B:\Vayyar_Dataset\small_data\for_martin_vcab\ford1') # ford 1
# X_train1, y_train1=sample_label_extraction(train_dataset1)
# y_train1 = np.array(y_train1)
# print('X_train1.shape', X_train1.shape)
# print('y_train1.shape', y_train1.shape)
# print('The Name of training classess - y_train1',np.unique(y_train1))
# # print('train_dataset1[0]',train_dataset1[0])
# # To get a distribution of the classes
# print('train_dataset1.class_distribution()',train_dataset1.class_distribution())
# # To get the mean and std of the entire dataset
# print('train_dataset1.mean_and_std()',train_dataset1.mean_and_std())
# train_dataset2 = rfImageDataSet(r'B:\Vayyar_Dataset\small_data\for_martin_vcab\ford1_center') #ford1_center
# X_train2, y_train2=sample_label_extraction(train_dataset2)
# y_train2 = np.array(y_train2)
# print('X_train2.shape', X_train2.shape)
# print('y_train2.shape', y_train2.shape)
# print('The Name of training classess - y_train2',np.unique(y_train2))
# # print('train_dataset2[0]',train_dataset2[0])
# # To get a distribution of the classes
# print('train_dataset2.class_distribution()',train_dataset2.class_distribution())
# # To get the mean and std of the entire dataset
# print('train_dataset2.mean_and_std()',train_dataset2.mean_and_std())
# train_dataset3 = rfImageDataSet(r'B:\Vayyar_Dataset\small_data\for_martin_vcab\ford2') #ford2
# # if the sampel is Ford 2
# X_train3, y_train3=sample_label_extraction(train_dataset3)
# y_train3 = np.array(y_train3)
# #
# print('X_train3.shape', X_train3.shape)
# print('y_train3.shape', y_train3.shape)
# print('The Name of training classess - y_train3',np.unique(y_train3))
# # print('train_dataset3[0]',train_dataset3[0])
# # To get a distribution of the classes
# print('train_dataset3.class_distribution()',train_dataset3.class_distribution())
# # To get the mean and std of the entire dataset
# print('train_dataset3.mean_and_std()',train_dataset3.mean_and_std())
# X_train=np.concatenate((X_train1, X_train2,X_train3))
# y_train=np.concatenate((y_train1, y_train2, y_train3))


# test_dataset = rfImageDataSet(r'B:\Vayyar_Dataset\small_data\for_martin_vcab\ford2_center') #ford2_center
# X_val, y_val = sample_label_extraction(test_dataset)
# y_val = np.array(y_val)
# # print('test_dataset[0]',test_dataset[0])
# # To get a distribution of the classes
# print('test_dataset.class_distribution()',test_dataset.class_distribution())
# # To get the mean and std of the entire dataset
# print('test_dataset.mean_and_std()',test_dataset.mean_and_std())
# X_train = np.array(X_train)
# y_train = np.array(y_train)
# print('X_train.shape', X_train.shape)
# print('y_train.shape', y_train.shape)
# y_cat = to_categorical(y_train)
# print('y_cat', y_cat)

# print('X_val.shape', X_val.shape)
# print('y_val.shape', y_val.shape)
