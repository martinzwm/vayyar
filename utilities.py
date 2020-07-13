import scipy.io as sio
import pickle
import numpy as np
import os
import json
from sklearn.preprocessing import MultiLabelBinarizer
#%%
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
    xList = np.array(xList)
    yList = np.array(yList)
    return xList, yList

def makeOccupancyLabelsWithType(occupancyLabel, occupancyTypeLabel):
    label = list()
    assert len(occupancyLabel) == len(occupancyTypeLabel), "occupancyLabel length does not equal to occupancyTypeLabel length"
    for item in range(1,6):
        if item in occupancyLabel:
            index = occupancyLabel.index(item)
            occupancyType = occupancyTypeLabel[index]
            if occupancyType == 'ADT':
                label.extend([0,1])
            elif occupancyType == 'SCD' or occupancyType == 'MCD' or occupancyType == 'LCD' or occupancyType == 'IFT':
                label.extend([1,0])
            else:
                label.extend([0,0])
        else:
            label.extend([0,0])
    return label
