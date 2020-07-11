# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
#%%%
import numpy as np
import pandas as pd
import seaborn as sn
from matplotlib import style
style.use("ggplot")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix, plot_confusion_matrix
import seaborn
from sklearn.model_selection import cross_val_score
import pickle



# %%
X = loadData("/Users/jameshe/Documents/radar_ura/vayyar/x.pickle")
Y = loadData("/Users/jameshe/Documents/radar_ura/vayyar/y.pickle")
print(X.shape)
print(Y.shape)
X = np.reshape(X, (X.shape[0],X.shape[1] * X.shape[2] * X.shape[3]))
print('Xin',X.shape)
print(X.dtype)


# %%
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20)
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

# %%
AC = clf.score(X_test,y_test)
report = classification_report(y_test, y_pred)
cm=multilabel_confusion_matrix(y_test,y_pred)
print('report',AC)
df_cm = pd.DataFrame(cm, range(8),range(8))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, annot=True,annot_kws={"size": 16})# font size
plt.show()
matrix_proportions =  np.zeros((8,8))
for i in range(0,8):
        matrix_proportions[i,:] = cm[i,:]/float(cm[i,:].sum())
        names=['S0', 'S1', 'S2' , 'S3', 'S4' 'S7', 'S6', 'S7']
        Conf = pd.DataFrame(matrix_proportions, index=names,columns=names)

seaborn.heatmap(Conf, annot=True, annot_kws={"size": 10}, cmap='gist_gray_r', cbar=False, square=True, fmt='.2f')
plt.ylabel(r'True Categories',fontsize=10)
plt.xlabel(r'Predicted Categories',fontsize=10)
plt.tick_params(labelsize=10)
plt.show()
print(Conf)

