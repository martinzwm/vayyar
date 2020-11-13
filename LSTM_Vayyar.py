import pandas as pd
import numpy as np
import os
import pickle
import torch
import json
import copy

from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, Embedding, LSTM, GlobalMaxPooling1D, SpatialDropout1D, Dropout, GRU, Bidirectional
from keras.layers import TimeDistributed

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import time
from keras.models import Sequential
# from imblearn.over_sampling import SMOTE
import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, mean_absolute_error, r2_score,accuracy_score



def plot_confusion_matrix(model, X, y):
    y_pred = model.predict_classes(X, verbose=0)
    plt.figure(figsize=(8, 6))
    sns.heatmap(pd.DataFrame(confusion_matrix(y, y_pred)), annot=True, fmt='d', cmap='YlGnBu', alpha=0.8, vmin=0)

###### --------------------------------------------------

class rfImageDataSet(Dataset):
    def __init__(self, rootDir, transform = None):
        self.rootDir = rootDir
        self.path_label = pd.read_pickle(os.path.join(rootDir, "path_label.pickle"))
        self.transform = transform

    def __len__(self):
        return len(self.path_label)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()
        # print(idx)
        # print(self.path_label.iloc[idx])
        rfImagePath = os.path.join(self.rootDir, str(self.path_label.iloc[idx, 3]) + '.npy')
        print(rfImagePath)
        image_power = np.load(rfImagePath)
        if self.transform:
            image_power = self.transform(image_power)
        label = self.path_label.iloc[idx, 1]
        sample = {
            'imagePower':image_power,
            'label':label,
            'path':self.path_label.iloc[idx, 0],
            'npy_id':self.path_label.iloc[idx, 3],
            'car_info':self.path_label.iloc[idx, 5],
            'car_model':self.path_label.iloc[idx,4]
            }
        return sample
        # Get a distribution of the 32 classes

    def class_distribution(self):
        '''
        Returns a dictionary (key: class, val: number of appearance in this dataset)
        '''
        class_counts = {}
        for idx in range(len(self.path_label)):
            class_hashable = ''.join([str(i) for i in self.path_label.iloc[idx, 1]])
            if class_hashable not in class_counts:
                class_counts[class_hashable] = 0
            class_counts[class_hashable] += 1
        return class_counts

    def mean_and_std(self):
        '''
        Returns a dictionary (key: 'mean' and 'std', val: the corresponding values)
        '''
        mean, std = 0, 0
        N = len(self.path_label)
        for idx in range(N):
            rfImagePath = os.path.join(self.rootDir, str(self.path_label.iloc[idx, 3]) + '.npy')
            image_power = np.load(rfImagePath)
            mean += image_power.mean()
            std += image_power.std()
        return mean / N, std / N

#
#
#
def sample_label_extraction(T_dataset):
    Xtrain_dataset_main = []
    Ytrain_dataset_main = []
    for i, sample in enumerate(T_dataset):
        # print(i)
        # print('---------------------------------------------------------------------------')

        x = T_dataset.__getitem__(i)
        XTrain = x['imagePower']
        YTrain = x['label']

        Xtrain_dataset_main.append(XTrain)
        Ytrain_dataset_main.append(YTrain)

        # print(x['imagePower'])
        # print('---------------------------------------------------------------------------')

    # print(len(Xtrain_dataset_main))
    X_train = np.array(Xtrain_dataset_main)
    y_train = np.array(Ytrain_dataset_main)

    # print('X_train.shape', X_train.shape)
    # print('y_train.shape', y_train.shape)

    return X_train, y_train
# #---------------------------------importing training samples-----------------------------------------------
# train_dataset1 = rfImageDataSet(r'D:\Vayyar Occupancy Detection\processed_vCab_Recordings_after clutter removal\processed\ford1') # ford 1
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


# train_dataset2 = rfImageDataSet(r'D:\Vayyar Occupancy Detection\processed_vCab_Recordings_after clutter removal\processed\ford1_center') #ford1_center

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

# train_dataset3 = rfImageDataSet(r'D:\Vayyar Occupancy Detection\processed_vCab_Recordings_after clutter removal\processed\ford2') #ford2
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


# test_dataset = rfImageDataSet(r'D:\Vayyar Occupancy Detection\processed_vCab_Recordings_after clutter removal\processed\ford2_center') #ford2_center
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
# #
# X_train = X_train.reshape(len(X_train), 1, 29 * 29 * 24)
# X_train = X_train.reshape(len(X_train), 29 * 29 * 24)

# print('Xin',X_train.shape)
# print('y',y_train.shape)


# # %------------------------------------------------SMOTE--------------------------------------------------------------
# #Resampling SMOTe
# sm = SMOTE(random_state=42)
# X_train, y_train = sm.fit_resample(X_train, y_train)
# print('X_resampling',X_train.shape)
# print('y_resampling',y_train.shape)
# y_train=np.array(y_train)
# print('y_resampling',y_train.shape)
# dd=y_train.transpose()
# Toarray = np.zeros((len(y_train), 1))
# #
# for i in range (0,len(y_train)-1):
#     a=(y_train[i])
#     Toarray[i]=a
# y_train=Toarray
# y_cat = to_categorical(y_train)
# print('y_cat', y_cat)
# #
# #
# #----------------------------------------------------------------------------------------------------------
# #
# #---------------------------------importing test samples-----------------------------------------------

# # test dataset
# X_test, y_test=sample_label_extraction(test_dataset)
# X_test = np.array(X_test)
# y_test = np.array(y_test)
# print('X_test.shape', X_test.shape)
# print('y_test.shape', y_test.shape)
# X_test = X_test.reshape(len(X_test), 1, 29 * 29 * 24)
# X_test = X_test.reshape(len(X_test), 29 * 29 * 24)
# print('X_test.reshaped', X_test.shape)
# print('y_test.shape', y_test.shape)




#  # --------------------------------------scaling/normalization and PCA-----------------------------------------------------------------


# scaler = StandardScaler()
# # scaler=MinMaxScaler()
# scaler.fit(X_train)
# # Apply transform to both the training set and the test set.
# X_train = scaler.transform(X_train)
# print('X_train_split',X_train.shape)
# X_test = scaler.transform(X_test)
# print('X_test_split',X_test.shape)
# # pca = PCA(.95)
# # pca.fit(X_train)
# # print(pca.n_components_)
# # X_train = pca.transform(X_train)
# # X_test = pca.transform(X_test)
# # print('X_train_pca',X_train.shape)
# # print('X_test_pca',X_test.shape)
# #---------------------------------------------- reshaping for time sample ------------------------------
# num_time_sample=1

# X_train=X_train.reshape(len( X_train), num_time_sample, X_train.shape[1] )
# # y_train=y_train.reshape(len( y_train), num_time_sample, y_train.shape[0] )

# print('X_train_pca',X_train.shape)
# print('The Name of training classess - y_train',np.unique(y_train))
# print('The Name of testing classess - y_test',np.unique(y_test))


# X_test=X_test.reshape(len( X_test), 1, X_test.shape[1] )
# print('X_test_pca',X_test.shape)



# # # %________________________________________________________________________________________________
# # #------------------------------ LSTM-------------------------------

# model_lstm = Sequential()
# # model_lstm.add(Embedding(input_dim = (X_train.shape[1],), output_dim = 512, input_length = (X_train.shape[1],)))
# # model_lstm.add(SpatialDropout1D(0.3))
# model_lstm.add(Bidirectional(1024, input_shape=(X_train.shape[1],X_train.shape[2]),activation='relu'))
# model_lstm.add(Dropout(0.3))
# # model_lstm.add(Dense(1024*3, activation = 'relu'))
# # model_lstm.add(Dense(1024*2, activation = 'relu'))
# # model_lstm.add(Dense(1024, activation = 'relu'))
# model_lstm.add(Dense(512,  activation = 'relu'))
# model_lstm.add(Dense(256, activation = 'relu'))
# model_lstm.add(Dense(128,   activation = 'relu'))
# model_lstm.add(Dense(64,  activation = 'relu'))
# model_lstm.add(Dense(32, activation = 'softmax'))


# model_lstm.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
# #  training
# history = model_lstm.fit(X_train,y_cat,validation_split = 0.2, epochs = 10,    batch_size = 512, shuffle=True)

# print(history.history.keys())
# #  "Accuracy"
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper left')
# plt.show()
# # "Loss"
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper left')
# plt.show()



# #------------------------------test
# y_pred = model_lstm.predict_classes(X_test, verbose=1)
# print("Predictions:")
# # print(y_pred)
# # print(np.argmax(y_pred))
# print(classification_report(y_test, y_pred))
# # model_lstm.test_on_batch(X_test, y_test)

# plot_confusion_matrix(model_lstm, X_test, y_test)
# cm=confusion_matrix(y_test,y_pred)
# matrix_proportions = np.zeros((32,32))
# for i in range(0,32):
#         matrix_proportions[i,:] = cm[i,:]/float(cm[i,:].sum())
#         names=['seat0','seat1','seat2','seat3','seat4','seat5','seat12','seat13','seat14','seat15','seat23','seat24','seat25','seat34','seat35','seat45','seat123',
#                'seat124','seat125','seat134','seat135','seat145','seat234','seat235','seat245','seat345','seat1234','seat1235','seat1245','seat1345','seat2345','seat12345']
#         Conf = pd.DataFrame(matrix_proportions, index=names,columns=names)

# sns.heatmap(Conf, annot=True, annot_kws={"size": 10}, cmap='gist_gray_r', cbar=False, square=True, fmt='.2f')
# plt.ylabel(r'True Categories',fontsize=10)
# plt.xlabel(r'Predicted Categories',fontsize=10)
# plt.tick_params(labelsize=10)
# plt.show()
# print(Conf)




# # Evaluate the model on the test data using `evaluate`
# # print("Evaluate on test data")
# # results = model_lstm.evaluate(X_test, y_test, batch_size=128)
# # print("test loss, test acc:", results)
# #
# # # Generate predictions (probabilities -- the output of the last layer)
# # # on new data using `predict`
# # print("predictions shape:", y_pred.shape)

# # _____________________________
# # # ---------------------------

# model_bid = Sequential()
# # model_lstm.add(Embedding(input_dim = (X_train.shape[1],), output_dim = 512, input_length = (X_train.shape[1],)))
# # model_lstm.add(SpatialDropout1D(0.3))
# model_bid.add(Bidirectional(1024, input_shape=(X_train.shape[1],X_train.shape[2]),activation='relu'))
# model_bid.add(Dropout(0.3))
# # model_lstm.add(Dense(1024*3, activation = 'relu'))
# # model_lstm.add(Dense(1024*2, activation = 'relu'))
# # model_lstm.add(Dense(1024, activation = 'relu'))
# model_bid.add(Dense(512,  activation = 'relu'))
# model_bid.add(Dense(256, activation = 'relu'))
# model_bid.add(Dense(128,   activation = 'relu'))
# model_bid.add(Dense(64,  activation = 'relu'))
# model_bid.add(Dense(32, activation = 'softmax'))


# model_bid.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
# #  training
# history_bid = model_bid.fit(X_train,y_cat,validation_split = 0.2, epochs = 10,    batch_size = 512, shuffle=True)

# print(history_bid.history.keys())
# #  "Accuracy"
# plt.plot(history_bid.history['accuracy'])
# plt.plot(history_bid.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper left')
# plt.show()
# # "Loss"
# plt.plot(history_bid.history['loss'])
# plt.plot(history_bid.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper left')
# plt.show()



# #------------------------------test
# y_pred = model_bid.predict_classes(X_test, verbose=1)
# print("Predictions:")
# # print(y_pred)
# # print(np.argmax(y_pred))
# print(classification_report(y_test, y_pred))
# # model_lstm.test_on_batch(X_test, y_test)

# plot_confusion_matrix(model_bid, X_test, y_test)
# cm=confusion_matrix(y_test,y_pred)
# matrix_proportions = np.zeros((32,32))
# for i in range(0,32):
#         matrix_proportions[i,:] = cm[i,:]/float(cm[i,:].sum())
#         names=['seat0','seat1','seat2','seat3','seat4','seat5','seat12','seat13','seat14','seat15','seat23','seat24','seat25','seat34','seat35','seat45','seat123',
#                'seat124','seat125','seat134','seat135','seat145','seat234','seat235','seat245','seat345','seat1234','seat1235','seat1245','seat1345','seat2345','seat12345']
#         Conf = pd.DataFrame(matrix_proportions, index=names,columns=names)

# sns.heatmap(Conf, annot=True, annot_kws={"size": 10}, cmap='gist_gray_r', cbar=False, square=True, fmt='.2f')
# plt.ylabel(r'True Categories',fontsize=10)
# plt.xlabel(r'Predicted Categories',fontsize=10)
# plt.tick_params(labelsize=10)
# plt.show()
# print(Conf)




# # Evaluate the model on the test data using `evaluate`
# # print("Evaluate on test data")
# # results = model_lstm.evaluate(X_test, y_test, batch_size=128)
# # print("test loss, test acc:", results)
# #
# # # Generate predictions (probabilities -- the output of the last layer)
# # # on new data using `predict`
# # print("predictions shape:", y_pred.shape)


# #
# #
# # --------------------------------------------------------------------------------------------------
# #

# def full_multiclass_report(model,
#                            x,
#                            y_true,
#                            classes,
#                            batch_size=32,
#                            binary=False):
#     # 1. Transform one-hot encoded y_true into their class number
#     if not binary:
#         y_true = np.argmax(y_true, axis=1)

#     # 2. Predict classes and stores in y_pred
#     y_pred = model.predict_classes(x, batch_size=batch_size)

#     # 3. Print accuracy score
#     print("Accuracy : " + str(accuracy_score(y_true, y_pred)))

#     print("")

#     # 4. Print classification report
#     print("Classification Report")
#     print(classification_report(y_true, y_pred, digits=5))

#     # 5. Plot confusion matrix
#     cnf_matrix = confusion_matrix(y_true, y_pred)
#     print(cnf_matrix)
#     # plot_confusion_matrix(cnf_matrix, classes=classes)



# classes=np.unique(y_test)

# # full_multiclass_report (model_lstm,X_test,y_test, classes,batch_size=32,binary=False)

# print("Classification Report")
# print(classification_report(y_test, y_pred))










# print('Your code is successfuly done- Brava ')
# print('https://towardsdatascience.com/building-a-multi-output-convolutional-neural-network-with-keras-ed24c7bc1178')
# print('Building a multi-output Convolutional Neural Network with Keras-dont forget, '
#       'you can use it one for number of passengers, the other one for type')