# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
#%%%
import numpy as np
import pandas as pd
from matplotlib import style
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import cross_val_score
import h5py
from utilities import plot_confusion_matrix, importDataFromMatFiles, importDataOccupancyType, saveData, loadData, seatWiseTransformLabels, scenarioWiseTransformLabels, getConfusionMatrices
#%%
with h5py.File('training_dataset.hdf5', 'r') as f:
    X = f['x'][:]
    Y = f['y'][:]
    print(f['path'][:])
#%%print(X.shape)
print(Y.shape)
X = np.squeeze(np.sum(X, axis=3))
# X = np.reshape(X, (X.shape[0],X.shape[1] * X.shape[2] * X.shape[3]))
print('Xin.shape = ',X.shape)
print(X.dtype)

# %%
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.05)
# apply PCA to reduce the size
# Fit on training set only.
scaler = StandardScaler()
scaler.fit(X_train)

# Apply transform to both the training set and the test set.
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
print('X_train_split',X_train.shape)
print('X_test_split',X_test.shape)
pca = PCA(.95)
pca.fit(X_train)
print(pca.n_components_)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)
print('X_train_pca',X_train.shape)
print('X_test_pca',X_test.shape)

# svm classifier
clf = OneVsRestClassifier(svm.SVC(kernel='linear', C=1))
clf.fit(X_train, y_train)


# %%
y_pred = clf.predict(X_test)
#%%

y_pred_seat_result = np.array(seatWiseTransformLabels(y_pred))
y_test_seat_result = np.array(seatWiseTransformLabels(y_test))

y_pred_scenario_result = scenarioWiseTransformLabels(y_pred)
y_test_scenario_result = scenarioWiseTransformLabels(y_test)

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoder.fit(y_pred_scenario_result)
y_pred_trans_result = np.array([encoder.transform(y_pred_scenario_result)]).T
y_test_trans_result = np.array([encoder.transform(y_test_scenario_result)]).T

thirtytwo_classes_cf = confusion_matrix(y_test_trans_result, y_pred_trans_result)
confusionMatrices = getConfusionMatrices(y_pred_seat_result, y_test_seat_result)
# %%

# plot_confusion_matrix(clf, y_pred_trans_result, y_test_trans_result)  # doctest: +SKIP
class_names = list(set(y_test_scenario_result + y_pred_scenario_result))
plot_confusion_matrix(y_test_trans_result, y_pred_trans_result, thirtytwo_classes_cf, classes=np.array(class_names), title='32 classes Confusion Matrix')


#%%
for i in range(5):
        print(classification_report(y_test[i], y_pred[i]))
# %%
report = classification_report(y_test, y_pred)
cm=multilabel_confusion_matrix(y_test,y_pred)
print(report)
print(cm)