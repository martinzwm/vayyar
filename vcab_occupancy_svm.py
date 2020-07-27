#%%
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from utilities import loadmat, getPreprocessedRFImage
import os
from data_prep import vCabDataSet
import torch
import numpy as np
from torchvision import transforms
import torch.optim as optim
from torch.autograd import Variable
from models import SVM


#%% Import vCab_Recordings dataset
transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[2531.013427734375],
                                 std=[8374.5048828125])
        ])
dataset = vCabDataSet('/home/vayyar_data/processed_vCab_Recordings', transform)
print(dataset[0]['imagePower'].shape)
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
train_loader = DataLoader(
    train_set,
    batch_size=128,
    num_workers=8,
    shuffle=True
)
val_loader = DataLoader(
    val_set,
    batch_size=128,
    num_workers=8,
    shuffle=True
)
#%% Training the SVM
learning_rate = 0.1  # Learning rate
n_epochs = 10  # Number of epochs

def make_train_step(model, loss_fn, optimizer):
    # Builds function that performs a step in the train loop
    def train_step(x, y):
        # Clear gradients
        optimizer.zero_grad()
        # Sets model to TRAIN mode
        model.train()
        # Makes predictions
        y_hat = model(x)
        # Computes loss
        loss = loss_fn(y, y_hat)  # hinge loss
        # Computes gradients
        loss.backward()
        # Updates parameters and zeroes gradients
        optimizer.step()
        optimizer.zero_grad()
        # Returns the loss
        # return loss.item()
        return loss[0].data.cpu().numpy()
    
    # Returns the function that will be called inside the train loop
    return train_step

def hinge_loss(y, y_hat):
    return torch.mean(torch.clamp(1 - y_hat * y, min=0))

model = SVM()  # Our model
optimizer = optim.SGD(model.parameters(), lr=learning_rate)  # Our optimizer
model.train()  # Our model, SVM is a subclass of the nn.Module, so it inherits the train method
losses = []
val_losses = []
sum_loss = 0
sum_val_loss = 0
train_step = make_train_step(model, hinge_loss, optimizer)

for epoch in range(n_epochs):
    for batch in train_loader:
        #TODO: when have CUDA:
        #x_batch = batch['imagePower'].to(device)
        #y_batch = batch['label'].to(device)

        x_batch = torch.flatten(batch['imagePower'])
        y_batch = batch['label']
        print(batch['imagePower'].shape)
        print(y_batch.shape)

        loss = train_step(x_batch, y_batch)
        sum_loss += loss[0].data.cpu().numpy()
        losses.append(loss)
        
    with torch.no_grad():
        for x_val, y_val in val_loader:
            #x_val = x_val.to(device)
            #y_val = y_val.to(device)
            
            model.eval()

            y_val_pred = model(x_val)
            val_loss = hinge_loss(y_val, y_val_pred) 
            sum_val_loss += val_loss[0].data.cpu().numpy()
            val_losses.append(val_loss.item())
    print("Epoch {}, Loss: {}".format(epoch, sum_loss[0]))
    print("Epoch {}, Val Loss: {}".format(epoch, sum_val_loss[0]))

print(model.state_dict(), "/home/jennifer_yu/Desktop/vayyar_4d_radar_intelligience/svm.pt")
#%%
for epoch in range(epoch):
    perm = torch.randperm(N)  # Generate a set of random numbers of length: sample size
    sum_loss = 0  # Loss for each epoch
        
    for i in range(0, N, batch_size):
        x = X[perm[i:i + batch_size]]  # Pick random samples by iterating over random permutation
        y = Y[perm[i:i + batch_size]]  # Pick the correlating class
        
        x = Variable(x)  # Convert features and classes to variables
        y = Variable(y)

        optimizer.zero_grad()  # Manually zero the gradient buffers of the optimizer
        output = model(x)  # Compute the output by doing a forward pass
        
        loss = torch.mean(torch.clamp(1 - output * y, min=0))  # hinge loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Optimize and adjust weights

        sum_loss += loss[0].data.cpu().numpy()  # Add the loss
        
    print("Epoch {}, Loss: {}".format(epoch, sum_loss[0]))
torch.save(model.state_dict(), "/Users/jameshe/Documents/radar_ura/vayyar/core_code/occupancy_3d_cnn_model.pt")

#%% one
mean = 0.
std = 0.
count = 0
for samples in loader:
    samples = samples['imagePower']
    mean += samples.mean()
    std += samples.std()
    count += 1

mean /= len(loader.dataset)/count
std /= len(loader.dataset)/count
print(f'{mean}, {std}')



#%% Making prediction and write misclassified samples to another file

#%% Error analysis