# from LSTM_Vayyar import *
import pandas as pd
import numpy as np
import os
import pickle
import torch
import json
import copy
from utilities import loadmat
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import *
from utilities import importDataFromMatFiles, loadData, scenarioWiseTransformLabels, getConfusionMatrices, seatWiseTransformLabels, plot_seat_wise_bar, multiclass_metric
from torchvision import transforms
from data_prep import rfImageDataSet, cropR
import math
from torch.utils.tensorboard import SummaryWriter

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

# Deterministic Auto-Encoder (DAE)
class DAE(nn.Module):

    # declare layers
    def __init__(self, in_channel):
        super(DAE, self).__init__()
        self.fc1 = nn.Conv3d(in_channel, 64, (3,3,3), padding=(1,1,1))
        self.fc2 = nn.Conv3d(64, 128, (3,3,3), padding=(1,1,1))
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    # encoder: one ReLU hidden layer of 400 nodes, one Linear output layer of 20 nodes
    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h)

    # decoder: one ReLU hidden layer of 400 nodes, one sigmoid output layer of 784 nodes
    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    # forward: encoder followed by decoder
    def forward(self, x):
        z = self.encode(x.view(-1, 784))
        return self.decode(z)


class VGGNet(nn.Module):
    def __init__(self, in_channel=1, num_classes=32):
        super(VGGNet, self).__init__()
        self.conv_1 = nn.Conv3d(in_channel, 64, (3,3,3), padding=(1,1,1))
        nn.init.kaiming_normal_(self.conv_1.weight)
        self.conv_2 = nn.Conv3d(64, 64, (3,3,3), padding=1)
        nn.init.kaiming_normal_(self.conv_2.weight)
        self.group_1 = nn.GroupNorm(4,64)

        self.conv_3 = nn.Conv3d(64, 128, (3,3,3), padding=1)
        nn.init.kaiming_normal_(self.conv_3.weight)
        self.conv_4 = nn.Conv3d(128, 128, (3,3,3), padding=1)
        nn.init.kaiming_normal_(self.conv_4.weight)
        self.conv_5 = nn.Conv3d(128, 128, (3,3,3), padding=1)
        nn.init.kaiming_normal_(self.conv_5.weight)
        self.group_2 = nn.GroupNorm(8,128)

        self.conv_6 = nn.Conv3d(128, 256, (3,3,3), padding=1)
        nn.init.kaiming_normal_(self.conv_6.weight)
        self.conv_7 = nn.Conv3d(256, 256, (3,3,3), padding=1)
        nn.init.kaiming_normal_(self.conv_7.weight)
        self.conv_8 = nn.Conv3d(256, 256, (3,3,3), padding=1)
        nn.init.kaiming_normal_(self.conv_8.weight)
        self.group_3 = nn.GroupNorm(16,256)

        self.conv_9 = nn.Conv3d(256, 512, (3,3,3), padding=1)
        nn.init.kaiming_normal_(self.conv_6.weight)
        self.conv_10 = nn.Conv3d(512, 512, (3,3,3), padding=1)
        nn.init.kaiming_normal_(self.conv_7.weight)
        self.conv_11 = nn.Conv3d(512, 512, (3,3,3), padding=1)
        nn.init.kaiming_normal_(self.conv_8.weight)
        self.group_4 = nn.GroupNorm(32,512)

        self.fc_12 = nn.Linear(6912, 4096)
        self.fc_13 = nn.Linear(4096, num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.drop=nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.relu(x)
        x = self.group_1(x)
        x = self.conv_2(x)
        x = self.relu(x)
        x = self.group_1(x)
        x = self.maxpool(x)
        x = self.drop(x)

        x = self.conv_3(x)
        x = self.relu(x)
        x = self.group_2(x)
        x = self.conv_4(x)
        x = self.relu(x)
        x = self.group_2(x)
        x = self.conv_5(x)
        x = self.relu(x)
        x = self.group_2(x)
        x = self.maxpool(x)
        x = self.drop(x)

        x = self.conv_6(x)
        x = self.relu(x)
        x = self.group_3(x)
        x = self.conv_7(x)
        x = self.relu(x)
        x = self.group_3(x)
        x = self.conv_8(x)
        x = self.relu(x)
        x = self.group_3(x)
        x = self.maxpool(x)
        x = self.drop(x)

        # x = self.conv_9(x)
        # x = self.relu(x)
        # x = self.group_4(x)
        # x = self.conv_10(x)
        # x = self.relu(x)
        # x = self.group_4(x)
        # x = self.conv_11(x)
        # x = self.relu(x)
        # x = self.group_4(x)
        # x = self.maxpool(x)

        x = x.view(x.shape[0], -1)
        x = self.fc_12(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc_13(x)
        return x

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, 3, padding = 1)
        self.bn1 = nn.BatchNorm3d(planes)
        
        self.conv2 = nn.Conv3d(planes, planes, 3, padding = 1)
        self.bn2 = nn.BatchNorm3d(planes)
        
        self.shortcut = nn.Sequential()
        if (in_planes != planes):
            self.shortcut = nn.Sequential( nn.Conv3d(in_planes, planes, 3, padding = 1),
                                           nn.BatchNorm3d(planes))
            
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x) #for the input
        out = F.relu(out)
        return out     

class SmallResNet(nn.Module):
    def __init__(self, in_channel, hidden_channels, num_classes):
        super(SmallResNet, self).__init__()
        self.conv = nn.Conv3d(in_channel, hidden_channels[0], 3, padding = 1) #first conv
        self.bn = nn.BatchNorm3d(hidden_channels[0]) #then batchNorm
        #now use 3 residual blocks
        self.res1 = BasicBlock(hidden_channels[0],hidden_channels[1])
        self.res2 = BasicBlock(hidden_channels[1],hidden_channels[2])
        self.res3 = BasicBlock(hidden_channels[2],hidden_channels[3])
        #now do the maxpooling
        self.maxpool = nn.MaxPool3d(2, 2) 
        self.fc = nn.Linear(hidden_channels[3] * 3 * 3 * 3 , num_classes) #from maxpooling

    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        out = self.maxpool(out)
        out = self.fc(out.view(out.size(0), -1))
        return out

class CNNModelRC(nn.Module):
    def __init__(self, num_classes):
        super(CNNModelRC, self).__init__()
        self.conv_layer1 = self._conv_layer_set(1, 32)
        self.conv_layer2 = self._conv_layer_set(32, 64)
        self.fc1 = nn.Linear(64*7*7*6, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.LeakyReLU()
        # self.batch=nn.BatchNorm1d(128)
        self.drop=nn.Dropout(p=0.50)   
        
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
        return out

def train():
    # # Normalization
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.7263228545248579],
    #                             std=[3.6356451880572926])
    # ])

    # # For local
    # train_dataset = TrainDataSet([
    #     r'B:\Vayyar_Dataset\small_data\for_martin_vcab\ford1',
    #     r'B:\Vayyar_Dataset\small_data\for_martin_vcab\ford2'
    # ], transform=transform)

    # val_dataset = TrainDataSet([
    #     r'B:\Vayyar_Dataset\small_data\for_martin_vcab\ford1_center'
    # ], transform=transform)

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
    
    # # To balance the train_dataset
    # label_list = []
    # for i in range(len(train_dataset)):
    #     label_list.append(train_dataset[i]['label'][0])
    # label_list = np.array(label_list)

    # class_distribution = train_dataset.class_distribution()
    # class_weight = 1.0 / (class_distribution + 1e-5)
    
    # label_weight = class_weight[label_list]

    # weighted_sampler = WeightedRandomSampler(
    #     weights=label_weight,
    #     num_samples=len(label_weight),
    #     replacement=True
    # )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=1,
        shuffle=True
        # shuffle=False,
        # sampler=weighted_sampler
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=1,
        shuffle=True
    )

    # Build model
    num_classes = 32
    # model = VGGNet(in_channel=1, num_classes=num_classes).to(device)
    model = CNNModelRC(num_classes).to(device)
    # model = SmallResNet(in_channel=1, hidden_channels=[32, 64, 128, 256], num_classes=num_classes).to(device)
    # model = torch.load('lean_cnn.pt').to(device)
    model.train() # mode = train
    print(model)

    error = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # CNN model training
    num_epochs = 200
    loss_list, val_loss_list, train_acc_list, val_acc_list = [], [], [], []
    iteration_count = 0
    train_per_epoch = math.ceil(len(train_dataset) / batch_size)
    val_per_epoch = math.ceil(len(val_dataset) / batch_size)

    start = time.time()
    for epoch in range(num_epochs):
        sum_loss, sum_val_loss, sum_train_acc, sum_val_acc = 0, 0, 0, 0
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
            if device == torch.device('cuda'):
                sum_train_acc += accuracy_score(outputs.detach().cpu().numpy().argmax(axis=1), y_train.detach().cpu().numpy())
            else:
                sum_train_acc += accuracy_score(outputs.detach().numpy().argmax(axis=1), y_train.detach().numpy())
            iteration_count += 1

        if device == torch.device('cuda'):
            sum_loss = sum_loss.cpu()
        loss_list.append(sum_loss/train_per_epoch)
        train_acc_list.append(sum_train_acc/train_per_epoch)
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
        print("Epoch {}, Train Accuracy: {}".format(epoch+1, sum_train_acc/train_per_epoch))
        print("Epoch {}, Val Accuracy: {}".format(epoch+1, sum_val_acc/val_per_epoch))

    end = time.time()
    print(f'duration = {end - start}s')

    # Save train loss and val loss profile
    train_loss_profile = {
        'train_loss': np.array(loss_list),
        'val_loss': np.array(val_loss_list),
        'train_acc': np.array(train_acc_list),
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
