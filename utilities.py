import scipy.io as sio
import pickle
import numpy as np
import os
import json
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], sio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

def saveData(data, path):
    """
    open file mode w: write, if the file exist, erase it
    open file mode b: open the file as a binary file
    """
    with open(path,'wb') as pickleFileHandle:
        pickle.dump(data, pickleFileHandle)
        pickleFileHandle.close()

def loadData(path):
    """
    open file mode b: open the file as a binary file
    open file mode r: read file
    """
    with open(path, 'rb') as pickleFileHandle:
        data = pickle.load(pickleFileHandle)
        return data

def importDataFromMatFiles(rootDir):
    """
    Import data from the small dataset (FirstBatch)
    Param: rootDir: The parent directory to the directories that start with Copy ....
                    For example, in this case, rootDir = "/Users/jameshe/Documents/radar_ura/FirstBatch"
    """
    xList = list()
    yList = list()
    mlb = MultiLabelBinarizer()
    for f in os.scandir(rootDir):
        if f.is_dir():
            with open(os.path.join(f.path, "test_data.json")) as labelFile:
                labels = json.load(labelFile)
                occupancyLabel = labels["Occupied_Seats"]
            for file in os.scandir(os.path.join(f.path, "SavedVars_RF_Data")):
                if '.mat' in file.name:
                    frame = sio.loadmat(file)
                    imagePower = np.absolute(np.power(frame["Image"],2))
                    imageMaxPower = np.max(imagePower)
                    maskG = frame["Mask"].astype(bool)
                    allowedDropRelPeak = 5
                    maskT = (imagePower >= imageMaxPower/allowedDropRelPeak)
                    imagePower[~ (maskG & maskT)] = 0
                    xList.append(imagePower)
                    yList.append(occupancyLabel)
    yList = mlb.fit_transform(yList)
    xList = np.array(xList)
    return (xList, yList)

def importDataOccupancyType(rootDir):
    xList = list()
    yList = list()
    occupiedSeatList = list()
    occupantTypeList = list()
    pathList = list()
    
    for f in os.scandir(rootDir):
        if f.is_dir():
            with open(os.path.join(f.path, "test_data.json")) as labelFile:
                labels = json.load(labelFile)
                occupancyLabel = labels["Occupied_Seats"]
                occupancyTypeLabel = labels["Occupant_Type"]
            for file in os.scandir(os.path.join(f.path, "SavedVars_RF_Data")):
                if '.mat' in file.name:
                    frame  = sio.loadmat(file)
                    imagePower = np.absolute(np.power(frame["Image"], 2))
                    imageMaxPower = np.max(imagePower)
                    maskG = frame["Mask"].astype(bool)
                    allowedDropRelPeak = 5
                    maskT = (imagePower >= imageMaxPower/allowedDropRelPeak)
                    imagePower[~ (maskG & maskT)] = 0
                    xList.append(imagePower)
                    yLabel = makeOccupancyLabelsWithType(occupancyLabel, occupancyTypeLabel)
                    yList.append(yLabel)
                    occupiedSeatList.append(str(occupancyLabel).replace('[','').replace(']','').replace(' ',''))
                    occupantTypeList.append(str(occupancyTypeLabel).replace('[','').replace(']','').replace(' ','').replace("'",""))
                    pathList.append('/'.join(f.path.split('/')[-2:]))
                    # print(f'{occupancyLabel}, {occupancyTypeLabel}')
    xList = np.array(xList, dtype='float32')
    yList = np.array(yList, dtype='int8')
    occupiedSeatList = np.array(occupiedSeatList, dtype='S')
    occupantTypeList = np.array(occupantTypeList, dtype='S')
    pathList = np.array(pathList, dtype='S')
    return xList, yList, occupiedSeatList, occupantTypeList, pathList

def makeOccupancyLabelsWithType(occupancyLabel, occupancyTypeLabel):
    label = list()
    assert len(occupancyLabel) == len(occupancyTypeLabel), "occupancyLabel length does not equal to occupancyTypeLabel length"
    for item in range(1,6):
        if item in occupancyLabel:
            index = occupancyLabel.index(item)
            occupancyType = occupancyTypeLabel[index]
            if occupancyType == 'ADT':
                # label.extend([0,0])
                label.extend([0,0,1])
            elif occupancyType == 'SCD' or occupancyType == 'MCD' or occupancyType == 'LCD' or occupancyType == 'IFT':
                # label.extend([0,1])
                label.extend([0,1,0])
            else:
                # label.extend([1,0])
                label.extend([1,0,0])
        else:
            # label.extend([1,0])
            label.extend([1,0,0])
    return label
#%%
def seatWiseTransformLabels(fifteenClassLabels):
    """
    tenClassLables: n x 15 label
    return: a tuple of five element, (n x 3 for seat one, n x 3 for seat two, ..., n x 3 for seat five)
            first element for empty, second element for adult, the third element for children 
    """
    fifteenClassLabels = fifteenClassLabels.astype('str')
    seat_1_label = [''.join(row) for row in fifteenClassLabels[:,0:3]]
    seat_2_label = [''.join(row) for row in fifteenClassLabels[:,3:6]]
    seat_3_label = [''.join(row) for row in fifteenClassLabels[:,6:9]]
    seat_4_label = [''.join(row) for row in fifteenClassLabels[:,9:12]]
    seat_5_label = [''.join(row) for row in fifteenClassLabels[:,12:15]]
    encoder = LabelEncoder()
    encoder.fit(['100', '010', '001'])
    seat_1_label = encoder.transform(seat_1_label)
    seat_2_label = encoder.transform(seat_2_label)
    seat_3_label = encoder.transform(seat_3_label)
    seat_4_label = encoder.transform(seat_4_label)
    seat_5_label = encoder.transform(seat_5_label)
    return [seat_1_label, seat_2_label, seat_3_label, seat_4_label, seat_5_label]

def scenarioWiseTransformLabels(fifteenClassLabels):
    """
    tenClassLables: n x 15 label
    return: n x 1 labels, 
    e.g. [0,0,1, 0,0,1, 1,0,0, 1,0,0, 0,1,0] -> [1,2,5]
    """
    result = list()
    for label in fifteenClassLabels:
        transform_str = ""
        for i in range(0, int(label.shape[0]), 3):
            if (list(label[i:i+3]) == [0,0,1]) or (list(label[i:i+3]) == [0,1,0]):
                transform_str += str(int(i/3 + 1)) #people present, append seat number
                transform_str += ","
            elif (list(label[i:i+3]) == [1,0,0]):
                transform_str += ""
                pass
            else:
                transform_str += str(int(i/3 + 1))
                transform_str += ("n/a")
                transform_str += ","
        if not transform_str: transform_str = "empty"
        if transform_str[-1] == ",": result.append(transform_str[:-1])  
    return result


def getConfusionMatrices(prediction: list, truth):
    confusionMatrices = list()
    for i in range(5):
        confusionMatrices.append(confusion_matrix(truth[i], prediction[i]))
    return confusionMatrices

def plot_confusion_matrix(y_true, y_pred, cm, classes, 
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig_cm, ax_cm = plt.subplots()
    im = ax_cm.imshow(cm, interpolation='nearest', cmap=cmap)
    ax_cm.figure.colorbar(im, ax=ax_cm)
    # We want to show all ticks...
    ax_cm.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax_cm.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax_cm.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig_cm.tight_layout()
    return ax_cm


# %%
