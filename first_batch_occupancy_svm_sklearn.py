#%%
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from utilities import loadmat, getPreprocessedRFImage, scenarioWiseTransformLabels
import os
from data_prep import rfImageDataSet, cropR
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
#%% Import dataset
transform = transforms.Compose([
            cropR(24),
            transforms.ToTensor(),
            transforms.Normalize(mean=[1.655461726353112e-06],
                                 std=[1.3920989854294221e-05])
        ])
dataset = rfImageDataSet('/home/vayyar_data/processed_FirstBatch', transform)

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
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import time
custom_classifier1 = SGDClassifier(learning_rate='constant', eta0=0.01)
custom_classifier2 = PassiveAggressiveClassifier()
custom_classifier3 = Perceptron()
clf_dict =  {'svm':MultiOutputClassifier(custom_classifier1),
             'passive_aggressive': MultiOutputClassifier(custom_classifier2),
             'perceptron': MultiOutputClassifier(custom_classifier3)
                }
#%%
training_accuracy = {'svm':[],
                     'passive_aggressive': [],
                     'perceptron': []
                    }
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
    filename = f'{clf}_FirstBatch.pickle'
    pickle.dump(clf_dict[clf], open(filename, 'wb'))
print("finished training")

#%% 
from sklearn.metrics import accuracy_score
val_accuracy = {'svm':[],
                'passive_aggressive': [],
                'perceptron': []
                }
for clf in clf_dict:
    print(f'{clf}_firstBatch_result.csv')
    f1 = open(f'{clf}_firstBatch_result.csv', 'w')
    f2 = open(f'{clf}_misclassified_firstBatch_result.csv', 'w')
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
        loaded_model = pickle.load(open(f'{clf}_FirstBatch.pickle', 'rb'))
        val_dict = dict()
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
        df.to_csv(f'{clf}_firstBatch_result.csv', mode='a', header=False, index=False)
        mis_df.to_csv(f'{clf}_misclassified_firstBatch_result.csv', mode='a', header=False, index=False)
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
