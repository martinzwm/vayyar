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
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.autograd import Variable
import torch.nn as nn
from torch.optim import *
from utilities import importDataFromMatFiles, loadData, scenarioWiseTransformLabels, getConfusionMatrices, seatWiseTransformLabels, plot_seat_wise_bar, multiclass_metric
from models import CNNModel, CNNModelRC
from torchvision import transforms
from data_prep import rfImageDataSet, cropR
import pkbar
import math
from torch.utils.tensorboard import SummaryWriter
import argparse
import seaborn as sn


#%% Import dataset
writer = SummaryWriter()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#ford2: mean: -0.0657602995634079, standard deviation: 0.051400020718574524
#ford1: mean: -0.059092991054058075, standard deviation: 0.0702936202287674
#5minutes: mean: 0.07992006093263626, standard deviation: 0.1240379586815834


transform = transforms.Compose([
            cropR(24),
            transforms.ToTensor(),
            transforms.Normalize(mean=[-0.0657602995634079],
                                 std=[0.051400020718574524])
        ])
train_set = rfImageDataSet('/home/vayyar_data/processed_vCab_Recordings_ford2_training/', transform) #103897 samples
val_set = rfImageDataSet('/home/vayyar_data/processed_vCab_Recordings_ford2_validation/', transform) #70167 samples
test_set = rfImageDataSet('/home/vayyar_data/processed_vCab_Recordings_ford2_testing/', transform) #24494 samples

# %%
batch_size = 512
train_loader = DataLoader(
    train_set,
    batch_size=batch_size,
    num_workers=8,
    shuffle=False
)
val_loader = DataLoader(
    val_set,
    batch_size=batch_size,
    num_workers=8,
    shuffle=False
)
test_loader = DataLoader(
    test_set,
    batch_size=batch_size,
    num_workers=8,
    shuffle=False
)
print("finished loaders")
#%%
#Definition of hyperparameters
start = time.time()
model_name = "cnn_clutter_removal_ford2.pickle"
# model_name = "/home/vayyar_model/first_batch_cnn_20200807.pt"
num_classes = 15
num_epochs = 10
# Create CNN
model = CNNModelRC(num_classes)
model.train()
#model.cuda()
print(model)

# Binary Cross Entropy Loss for MultiLabel Classfication
error = nn.BCELoss()
# learning_rate = 0.001 #FirstBatch
learning_rate = 0.0001 #Vcab_Recordings with clutter removal
# learning_rate = 0.005 #Vcab_Recordings
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
        iteration_count += 1
        kbar.update(i, values=[("loss", loss)])
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
torch.save(model.state_dict(), model_name)
end = time.time()
print(f'duration = {end - start}s')
# %%
model = CNNModelRC(num_classes)
model.load_state_dict(torch.load(model_name))
model.eval()

test_per_epoch = math.ceil(len(test_set) / batch_size)
accuracy = []
misclassified_filename = "ford2_misclassified.csv"

f1 = open('all_ford2.csv', 'w')
f2 = open(misclassified_filename, 'w')
f1.write(','.join(['path', 
                   'label_seat', 'predicted_seat', 
                   'label_type', 'predicted_type\n']))
f2.write(','.join(['path', 
                   'label_seat', 'predicted_seat', 
                   'label_type', 'predicted_type\n']))
f1.close()
f2.close()
cm_dict = {"cm1": np.zeros((4,4), dtype=np.int64),
           "cm2": np.zeros((4,4), dtype=np.int64),
           "cm3": np.zeros((4,4), dtype=np.int64),
           "cm4": np.zeros((4,4), dtype=np.int64),
           "cm5": np.zeros((4,4), dtype=np.int64)}

for sample in test_loader:
    x_test = sample["imagePower"].float().to(device)
    y_test = sample["label"].float().to(device)
    path = np.array(sample['path'])    
    train = Variable(x_test.view(len(x_test), 1, 29 ,29 ,24))
    outputs = model(train)
    outputs[outputs < 0.5] = 0
    outputs[outputs > 0.5] = 1
    outputs = outputs.detach().numpy().astype('int8')
    y_test = y_test.detach().numpy().astype('int8')
    accuracy.append(accuracy_score(y_test, outputs))
    test_dict = {
        'path': [],
        'label_seat': [], 'predicted_seat':[],
        'label_type': [], 'predicted_type': []
    }
    test_dict['path'] = list(path)
    test_dict['predicted_seat'], test_dict['predicted_type'] = scenarioWiseTransformLabels(outputs)
    test_dict['label_seat'], test_dict['label_type'] = scenarioWiseTransformLabels(y_test)
    pred_seat_label = seatWiseTransformLabels(outputs)
    true_seat_label = seatWiseTransformLabels(y_test)
    for i in range(len(pred_seat_label)):
        label = set(true_seat_label[i] + pred_seat_label[i])
        cm = confusion_matrix(true_seat_label[i], pred_seat_label[i], labels=['ADT', 'KID', 'EMP', 'OTHER'])
        cm_name = "cm" + str(i+1)
        cm_dict[cm_name] = np.add(cm_dict[cm_name], cm)
    df = pd.DataFrame.from_dict(test_dict)
    df['seat_prediction_result'] = np.where(df['label_seat'] == df['predicted_seat'], True, False)
    df['type_prediction_result'] = np.where(df['label_type'] == df['predicted_type'], True, False)
    mis_df = df.loc[(df['seat_prediction_result'] == False) | (df['type_prediction_result'] == False)]
    df.to_csv('all_cnn_clutter_removal.csv', mode='a', header=False, index=False)
    mis_df.to_csv(misclassified_filename, mode='a', header=False, index=False)
acc = np.average(np.array(accuracy))
print(f'Testing accuracy is {acc}.')
# %%
f1_score_array = np.empty((5,3))
for i, cm in enumerate(cm_dict):
    f1_score_array[i] = multiclass_metric(cm_dict[cm], metric='accuracy')
    normalized_cm = cm_dict[cm] / np.array([np.sum(cm_dict[cm], axis=1),]*4).T
    df_cm = pd.DataFrame(list(normalized_cm[:3,:]), index = ["ADULT", "KID", "EMPTY"],
                  columns = ["ADULT", "KID", "EMPTY", "UNKNOWN"])
    plt.figure(figsize = (10,7))
    plt.title(f'Confusion Matrix for Seat {i+1}')
    
    sn.heatmap(df_cm, annot=True, fmt='.2%', cmap='Blues')
    plt.xlabel('Predicted Class')
    plt.ylabel('Target Class')
plot_seat_wise_bar(f1_score_array, metric='Accuracy')
# %%

# %%
