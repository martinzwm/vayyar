#%%
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from utilities import loadmat, getPreprocessedRFImage, scenarioWiseTransformLabels
import os
from data_prep import rfImageDataSet, cropR, clutterRemoval
import torch
import numpy as np
from torchvision import transforms
import torch.optim as optim
from torch.autograd import Variable
from models import SVM
import pkbar
from sklearn.multioutput import MultiOutputClassifier
import pickle
torch.manual_seed(0)
#%% Import vCab_Recordings dataset
transform = transforms.Compose([
            cropR(24),
            transforms.ToTensor(),
            transforms.Normalize(mean=[-0.05493089184165001],
                                 std=[0.035751599818468094])
        ])
dataset = rfImageDataSet('/home/vayyar_data/processed_vCab_Recordings_clutter_removed', transform)

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
#%% Training the SVM pytorch approach
import time
start = time.time()
learning_rate = 0.1  # Learning rate
n_epochs = 100  # Number of epochs

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
        return loss.item()
    
    # Returns the function that will be called inside the train loop
    return train_step

def hinge_loss(y, y_hat):
    return torch.mean(torch.clamp(1 - y_hat * y, min=0))

model = SVM()  # Our model
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Our optimizer
model.train()  # Our model, SVM is a subclass of the nn.Module, so it inherits the train method
losses = []
val_losses = []
train_step = make_train_step(model, hinge_loss, optimizer)
train_per_epoch = int(len(train_set) / batch_size)

for epoch in range(n_epochs):
    sum_loss = 0
    sum_val_loss = 0
    kbar = pkbar.Kbar(target=train_per_epoch, width=8)
    for i, batch in enumerate(train_loader):
        #TODO: when have CUDA:
        #x_batch = batch['imagePower'].to(device)
        #y_batch = batch['label'].to(device)

        x_batch = batch['imagePower']
        y_batch = batch['label']
    
        loss = train_step(x_batch, y_batch)
        sum_loss += loss
        losses.append(loss)
        kbar.update(i, values=[("loss", loss)])

    print('done training')
    with torch.no_grad():
        for val_batch in val_loader:
            #x_val = x_val.to(device)
            #y_val = y_val.to(device)
            
            x_val = val_batch['imagePower']
            y_val = val_batch['label']

            model.eval()

            y_val_pred = model(x_val)
            val_loss = hinge_loss(y_val, y_val_pred) 
            sum_val_loss += val_loss.item()
            val_losses.append(val_loss.item())
            
    kbar.add(1, values=[("loss", loss), ("val_loss", val_loss)])
    print('done validation')
    print("Epoch {}, Loss: {}".format(epoch, sum_loss))
    print("Epoch {}, Val Loss: {}".format(epoch, sum_val_loss))
print(time.time()-start)

#%%
model_path = "/home/vayyar_model/svm_20200727.pt"
torch.save(model.state_dict(), model_path)
#%%
import matplotlib.pyplot as plt
fig = plt.figure()
ax = plt.axes()

x = np.arange(0, len(train_set)*10, batch_size)
ax.plot(x, losses[:len(x)])
ax.plot(x, val_losses[:len(x)])
labels = np.arange(1, 101, 10)
plt.xticks(np.arange(min(x), max(x), (max(x))/10))
# plt.yticks(np.arange(0, 1.1, 0.1))
plt.title('Loss function graph')
ax.legend(['loss', 'val_loss'])
ax.set_xticklabels(labels)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

#%%
model.load_state_dict(torch.load(model_path))
for test_batch in test_loader:
    #x_val = x_val.to(device)
    #y_val = y_val.to(device)
    
    x_test = test_batch['imagePower']
    y_test = test_batch['label']
    path = test_batch['path']

    model.eval()
    y_test_pred = model(x_test)
    y_test_pred[y_test_pred < 0.5] = 0
    y_test_pred[y_test_pred > 0.5] = 1
    y_test_pred.
    print(len(y_test_pred[y_test_pred != y_test]))
    break
#%% This is the sklearn SVM approach 
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import time
custom_classifier1 = SGDClassifier(learning_rate='constant', eta0=0.01)
clf_dict =  {'svm':MultiOutputClassifier(custom_classifier1)}
#%%
training_accuracy = {'svm':[]}
for i, batch in enumerate(train_loader):
    #TODO: when have CUDA:
    #x_batch = batch['imagePower'].to(device)
    #y_batch = batch['label'].to(device)
    
    x_batch = batch['imagePower'].detach().cpu().numpy()
    x_batch = x_batch.reshape(x_batch.shape[0], x_batch.shape[1]*x_batch.shape[2]*x_batch.shape[3])
    y_batch = batch['label'].detach().cpu().numpy()
    for clf in clf_dict:
        clf_dict[clf].partial_fit(x_batch, y_batch, classes=np.array([[0, 1]] * int(y_batch.shape[1])))
        y_pred = clf_dict[clf].predict(x_batch)
        try:
            training_accuracy[clf].append(clf_dict[clf].score(x_batch, y_batch))
        except:
            training_accuracy[clf] = list()
            training_accuracy[clf].append(clf_dict[clf].score(x_batch, y_batch))
# save the model to disk
for clf in clf_dict:
    filename = f'{clf}_vcab_clutter_removal_90.pickle'
    pickle.dump(clf_dict[clf], open(filename, 'wb'))
print("finished training")
#%% 
from sklearn.metrics import accuracy_score
val_accuracy = {'svm':[]}
for clf in clf_dict:
    print(f'{clf}_Vcab_result_removal_90.csv')
    f1 = open(f'{clf}_Vcab_result_removal_90.csv', 'w')
    f2 = open(f'{clf}_misclassified_Vcab_clutter_removal_result_90.csv', 'w')
    f1.write(','.join(['path', 'label_seat', 'predicted_seat', 'label_type', 'predicted_type', 'seat_prediction_result', 'type_prediction_result\n']))
    f2.write(','.join(['path', 'label_seat', 'predicted_seat', 'label_type', 'predicted_type', 'seat_prediction_result', 'type_prediction_result\n']))
    f1.close()
    f2.close()
for i, batch in enumerate(val_loader):
    x_batch = batch['imagePower'].detach().cpu().numpy()
    x_batch = x_batch.reshape(x_batch.shape[0], x_batch.shape[1]*x_batch.shape[2]*x_batch.shape[3])
    y_batch = batch['label'].detach().cpu().numpy()
    path = np.array(batch['path'])        
    for clf in clf_dict:
        loaded_model = pickle.load(open(f'{clf}_vcab_clutter_removal_90.pickle', 'rb'))
        val_dict = {
            'path': [],
            'label_seat': [],
            'predicted_seat':[],
            'label_type': [],
            'predicted_type': []
        }
        loaded_model.partial_fit(x_batch, y_batch, classes=np.array([[0, 1]] * int(y_batch.shape[1])))
        y_pred = loaded_model.predict(x_batch)

        val_dict['path'] = list(path)
        val_dict['predicted_seat'], val_dict['predicted_type'] = scenarioWiseTransformLabels(y_pred)
        val_dict['label_seat'], val_dict['label_type'] = scenarioWiseTransformLabels(y_batch)
        df = pd.DataFrame.from_dict(val_dict)
        df['seat_prediction_result'] = np.where(df['label_seat'] == df['predicted_seat'], True, False)
        df['type_prediction_result'] = np.where(df['label_type'] == df['predicted_type'], True, False)
        mis_df = df.loc[(df['seat_prediction_result'] == False) | (df['type_prediction_result'] == False)]
        df.to_csv(f'{clf}_Vcab_result_removal_90.csv', mode='a', header=False, index=False)
        mis_df.to_csv(f'{clf}_misclassified_Vcab_clutter_removal_result_90.csv', mode='a', header=False, index=False)
        try:
            val_accuracy[clf].append(accuracy_score(y_pred, y_batch))
        except:
            training_accuracy[clf] = list()
            val_accuracy[clf].append(accuracy_score(y_pred, y_batch))
# %%
for classifier in val_accuracy:
    acc = np.average(np.array(val_accuracy[classifier]))
    print('The {} validation accuracy is {}.'.format(classifier, acc))
for classifier in training_accuracy:
    acc = np.average(np.array(training_accuracy[classifier]))
    print('The {} training accuracy is {}.'.format(classifier, acc))


# %%
