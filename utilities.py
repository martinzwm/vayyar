#%%
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

    example: loadmat(.../168.mat) return a dict data = dict_keys(['__header__', '__version__', '__globals__', 'rfImageStruct'])
    data['rfImageStruct'] returns a dict: dict_keys(['image_DxDyR', 'mask', 'dx_grid', 'dy_grid', 'r_grid', 'axis_names'])
    further indexing by 'image_DxDyR' returns the actual rfImage
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
                # occupancyLabel is a list of int
                occupancyLabel = labels["Occupied_Seats"]
                occupancyTypeLabel = labels["Occupant_Type"]
            for file in os.scandir(os.path.join(f.path, "SavedVars_RF_Data")):
                if '.mat' in file.name:
                    frame  = sio.loadmat(file)
                    imagePower = getPreprocessedRFImage(frame["Image"], frame["Mask"])
                    xList.append(imagePower)
                    yLabel = makeOccupancyLabelsWithType(occupancyLabel, occupancyTypeLabel)
                    yList.append(yLabel)
                    occupiedSeatList.append(str(occupancyLabel).replace('[','').replace(']','').replace(' ',''))
                    occupantTypeList.append(str(occupancyTypeLabel).replace('[','').replace(']','').replace(' ','').replace("'",""))
                    pathList.append('/'.join(f.path.split('/')[-2:]))
    xList = np.array(xList, dtype='float32')
    yList = np.array(yList, dtype='int8')
    occupiedSeatList = np.array(occupiedSeatList, dtype='S')
    occupantTypeList = np.array(occupantTypeList, dtype='S')
    pathList = np.array(pathList, dtype='S')
    return xList, yList, occupiedSeatList, occupantTypeList, pathList

def makeOccupancyLabelsWithType(occupancyLabel, occupancyTypeLabel):
    """
    Create a label in the form of a numpy array of int. length = 15.
    """ 
    label = list()
    assert len(occupancyLabel) == len(occupancyTypeLabel), "occupancyLabel length does not equal to occupancyTypeLabel length"
    for item in range(1,6):
        if item in occupancyLabel:
            index = occupancyLabel.index(item)
            occupancyType = occupancyTypeLabel[index]
            if occupancyType == 'ADT':
                label.extend([0,0,1])
            elif occupancyType == 'SCD' or occupancyType == 'MCD' or occupancyType == 'LCD' or occupancyType == 'IFT':
                label.extend([0,1,0])
            else:
                label.extend([1,0,0])
        else:
            label.extend([1,0,0])
    label = np.array(label).astype('uint8')
    return label

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
    map_dict = {'100': 'EMP', '010':'KID','001':'ADT','000':'OTHER', '011':'OTHER', '101':'OTHER', '110':'OTHER', '111':'OTHER'}
    seat_1_label = list(map(map_dict.get, seat_1_label))
    seat_2_label = list(map(map_dict.get, seat_2_label))
    seat_3_label = list(map(map_dict.get, seat_3_label))
    seat_4_label = list(map(map_dict.get, seat_4_label))
    seat_5_label = list(map(map_dict.get, seat_5_label))
    return [seat_1_label, seat_2_label, seat_3_label, seat_4_label, seat_5_label]

def scenarioWiseTransformLabels(fifteenClassLabels):
    """
    tenClassLables: n x 15 label
    return: n x 1 labels, 
    e.g. [0,0,1, 0,0,1, 1,0,0, 1,0,0, 0,1,0] -> [1,2,5], [ADT,ADT,KID]
         [1,0,0, 1,0,0, 1,0,0, 1,0,0, 1,0,0] -> EMP, EMP

    """
    result_seat = list()
    result_type = list()
    for label in fifteenClassLabels:
        seat_transform_str = ""
        type_transform_str = ""
        for i in range(0, int(label.shape[0]), 3):
            if (list(label[i:i+3]) == [0,0,1]):
                seat_transform_str += str(int(i/3 + 1)) #people present, append seat number
                seat_transform_str += ","
                type_transform_str += 'ADT,'
            elif (list(label[i:i+3]) == [0,1,0]):
                seat_transform_str += str(int(i/3 + 1)) #people present, append seat number
                seat_transform_str += ","
                type_transform_str += 'KID,'
            elif (list(label[i:i+3]) == [1,0,0]):
                pass
            else:
                seat_transform_str += str(int(i/3 + 1))
                seat_transform_str += ("n/a,")
                type_transform_str += str(int(i/3 + 1))
                type_transform_str += ("n/a,")
        if not seat_transform_str: seat_transform_str = "EMP,"
        if not type_transform_str: type_transform_str = "EMP,"
        if seat_transform_str[-1] == ",": result_seat.append(seat_transform_str[:-1])  
        if type_transform_str[-1] == ",": result_type.append(type_transform_str[:-1])  

    return result_seat, result_type

def getConfusionMatrices(prediction: list, truth):
    confusionMatrices = list()
    for i in range(5):
        confusionMatrices.append(confusion_matrix(truth[i], prediction[i]))
    return confusionMatrices

def makeVcabPickleFile(remove_clutter=True):
    """
    One-time function calling function, can be called using python interpreter or jupyter notebook
    Code to generate the pickle file that will later be used to construct the custom pytorch dataset
    Works with the vCab dataset
    remove_clutter: True by default. If remove_clutter is enabled, it will subtract every single frame from the first frame at miniute level folder.
    """
    import os
    import pandas as pd
    import json
    from utilities import makeOccupancyLabelsWithType
    #create dictionary with two keys: path and label, "path" indexes a list of path. "label" indexes a list of labels
    data = {"path_original": [],
            "label": [],
            "processed_filename": []}

    #for loop to traverse the entire dataset and append to the two lists
    rootDir = '/home/vayyar_data/vCab_Recordings'
    index = 0
    for dayLevelItem in os.scandir(rootDir): #recording time level
        if dayLevelItem.is_dir():
            # omit out of position detectionf or now
            if 'OOP' not in dayLevelItem.name:
                for carLevelItem in os.scandir(dayLevelItem.path):#car level
                    if carLevelItem.is_dir():
                        for minuteLevelItem in os.scandir(carLevelItem.path):#minute level, e.g. v_Copy (12) - Copy__04-11-2019 15-23-54
                            if minuteLevelItem.is_dir():
                                with open(os.path.join(minuteLevelItem.path, "test_data.json")) as labelFile:
                                    labels = json.load(labelFile)
                                    occupancyLabel = labels["Occupied_Seats"]
                                    occupancyTypeLabel = labels["Occupant_Type"]
                                    first_frame_path = os.path.join(minuteLevelItem.path, "rfImages", "001")
                                    first_rfImage_struct = loadmat(first_frame_path)['rfImageStruct']
                                    first_frame = getPreprocessedRFImage(first_rfImage_struct['image_DxDyR'], first_rfImage_struct['mask'])
                                    for file in os.scandir(os.path.join(minuteLevelItem.path, "rfImages")):
                                        if file.name.endswith('.mat'):
                                            try:
                                                rfImageStruct = loadmat(file.path)['rfImageStruct']
                                                imagePower = getPreprocessedRFImage(rfImageStruct['image_DxDyR'], rfImageStruct['mask'])
                                                if remove_clutter == True:
                                                    if '001' in file.name:
                                                        continue
                                                    else:
                                                        imagePower = np.subtract(imagePower, first_frame)
                                                data["path_original"].append(file.path)
                                                data["processed_filename"].append(index)
                                                data["label"].append(makeOccupancyLabelsWithType(occupancyLabel, occupancyTypeLabel))
                                                # Store processed files into a separate folder. E.g. '/home/vayyar_data/processed_vCab_Recordings_clutter_removed/...' 
                                                processed_csv = '/home/vayyar_data/processed_vCab_Recordings_clutter_removed/%s.npy' % (str(index))
                                                np.save(processed_csv, imagePower)
                                                index += 1
                                            except Exception as ex:
                                                template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                                                message = template.format(type(ex).__name__, ex.args)
                                                print(message)
                                                print(file.path)
    # Create a pandas dataframe out from this dictionary. "path" will be the first column, "label" will be the second column.
    # Each row will contain info for a sample
    # Store the index dataframe into a pickle fle. E.g. '/home/vayyar_data/processed_vCab_Recordings_clutter_removed/path_label.pickle'
    df = pd.DataFrame.from_dict(data) 
    df.to_pickle('/home/vayyar_data/processed_vCab_Recordings_clutter_removed/path_label.pickle')

#create files for seperate session tests
def makeSeperatedVcabPickleFile(remove_clutter=True):
    """
    One-time function calling function, can be called using python interpreter or jupyter notebook
    Code to generate the pickle file that will later be used to construct the custom pytorch dataset
    Works with the vCab dataset
    remove_clutter: True by default. If remove_clutter is enabled, it will subtract every single frame from the first frame at miniute level folder.
    """
    import os
    import pandas as pd
    import json
    from utilities import makeOccupancyLabelsWithType
    #create dictionary with two keys: path and label, "path" indexes a list of path. "label" indexes a list of labels
    data = {"path_original": [],
            "label": [],
            "processed_filename": []}

    #for loop to traverse the entire dataset and append to the two lists
    rootDir = '/home/vayyar_data/vCab_Recordings'
    index = 0
    # car_set = list()
    minute_level_set = list()
    for dayLevelItem in os.scandir(rootDir): #recording time level
        if dayLevelItem.is_dir():
            # omit out of position detectionf or now
            if 'OOP' not in dayLevelItem.name:
                for carLevelItem in os.scandir(dayLevelItem.path):#car level
                    if carLevelItem.is_dir() and ('ford2' in carLevelItem.path):
                        for minuteLevelItem in os.scandir(carLevelItem.path):#minute level, e.g. v_Copy (12) - Copy__04-11-2019 15-23-54
                            if minuteLevelItem.is_dir():
                                minute_level_set.append(minuteLevelItem.path)
                                # with open(os.path.join(minuteLevelItem.path, "test_data.json")) as labelFile:
                                #     labels = json.load(labelFile)
                                #     car_set.append(labels['Car_Model'][0])
    print(len(minute_level_set))
    training_num = int(0.7 * len(minute_level_set))
    validation_num = int(0.05 * len(minute_level_set))
    testing_num = len(minute_level_set) - training_num - validation_num
    training_set = minute_level_set[0:training_num]
    validation_set = minute_level_set[training_num:training_num+validation_num]
    testing_set = minute_level_set[training_num+validation_num:]

    for dayLevelItem in os.scandir(rootDir): #recording time level
        if dayLevelItem.is_dir():
            # omit out of position detectionf or now
            if 'OOP' not in dayLevelItem.name:
                for carLevelItem in os.scandir(dayLevelItem.path):#car level
                    if carLevelItem.is_dir() and ('ford2' in carLevelItem.path):
                        for minuteLevelItem in os.scandir(carLevelItem.path):#minute level, e.g. v_Copy (12) - Copy__04-11-2019 15-23-54
                            if minuteLevelItem.is_dir() and minuteLevelItem.path in testing_set:
                                with open(os.path.join(minuteLevelItem.path, "test_data.json")) as labelFile:
                                    labels = json.load(labelFile)
                                    occupancyLabel = labels["Occupied_Seats"]
                                    occupancyTypeLabel = labels["Occupant_Type"]
                                    first_frame_path = os.path.join(minuteLevelItem.path, "rfImages", "001")
                                    first_rfImage_struct = loadmat(first_frame_path)['rfImageStruct']
                                    first_frame = getPreprocessedRFImage(first_rfImage_struct['image_DxDyR'], first_rfImage_struct['mask'])
                                    for file in os.scandir(os.path.join(minuteLevelItem.path, "rfImages")):
                                        if file.name.endswith('.mat'):
                                            try:
                                                rfImageStruct = loadmat(file.path)['rfImageStruct']
                                                imagePower = getPreprocessedRFImage(rfImageStruct['image_DxDyR'], rfImageStruct['mask'])
                                                if remove_clutter == True:
                                                    if '001' in file.name:
                                                        continue
                                                    else:
                                                        imagePower = np.subtract(imagePower, first_frame)
                                                data["path_original"].append(file.path)
                                                data["processed_filename"].append(index)
                                                data["label"].append(makeOccupancyLabelsWithType(occupancyLabel, occupancyTypeLabel))
                                                # Store processed files into a separate folder. E.g. '/home/vayyar_data/processed_vCab_Recordings_clutter_removed/...' 
                                                processed_csv = '/home/vayyar_data/processed_vCab_Recordings_ford2_testing/%s.npy' % (str(index))
                                                np.save(processed_csv, imagePower)
                                                index += 1
                                            except Exception as ex:
                                                template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                                                message = template.format(type(ex).__name__, ex.args)
                                                print(message)
                                                print(file.path)
    # Create a pandas dataframe out from this dictionary. "path" will be the first column, "label" will be the second column.
    # Each row will contain info for a sample
    # Store the index dataframe into a pickle fle. E.g. '/home/vayyar_data/processed_vCab_Recordings_clutter_removed/path_label.pickle'
    df = pd.DataFrame.from_dict(data) 
    df.to_pickle('/home/vayyar_data/processed_vCab_Recordings_ford2_testing/path_label.pickle')

def makeFirstBatchPickleFile():
    """
    One-time function calling function, can be called using python interpreter or jupyter notebook
    Code to generate the pickle file that will later be used to construct the custom pytorch dataset
    Works with the FirstBatch dataset
    Note: the clutter removal processing technique has been applied on FirstBatch dataset already, so no more processing is needed
    """
    import os
    import pandas as pd
    import json
    from utilities import makeOccupancyLabelsWithType
    #create dictionary with two keys: path and label, "path" indexes a list of path. "label" indexes a list of labels
    data = {"path_original": [],
            "label": [],
            "processed_filename": []}

    #for loop to traverse the entire dataset and append to the two lists
    rootDir = '/home/vayyar_data/FirstBatch'
    index = 0
    for f in os.scandir(rootDir):
        if f.is_dir():
            with open(os.path.join(f.path, "test_data.json")) as labelFile:
                labels = json.load(labelFile)
                # occupancyLabel is a list of int
                occupancyLabel = labels["Occupied_Seats"]
                occupancyTypeLabel = labels["Occupant_Type"]
            for file in os.scandir(os.path.join(f.path, "SavedVars_RF_Data")):
                if '.mat' in file.name:
                    try:
                        frame  = sio.loadmat(file)
                        imagePower = getPreprocessedRFImage(frame["Image"], frame["Mask"])
                        yLabel = makeOccupancyLabelsWithType(occupancyLabel, occupancyTypeLabel)
                        # Store processed files into a separate folder. E.g. '/home/vayyar_data/processed_vCab_Recordings_clutter_removed/...' 
                        processed_csv = '/home/vayyar_data/processed_FirstBatch_nonthreshold/%s.npy' % (str(index))
                        data["path_original"].append(file.path)
                        data["processed_filename"].append(index)
                        data["label"].append(yLabel)
                        np.save(processed_csv, imagePower)
                        index += 1
                    except Exception as ex:
                        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                        message = template.format(type(ex).__name__, ex.args)
                        print(message)
                        print(file.path)

    # Create a pandas dataframe out from this dictionary. "path" will be the first column, "label" will be the second column.
    # Each row will contain info for a sample
    # Store the index dataframe into a pickle fle. E.g. '/home/vayyar_data/processed_FirstBatch_nonthreshold/path_label.pickle'
    df = pd.DataFrame.from_dict(data)
    df.to_pickle('/home/vayyar_data/processed_FirstBatch_nonthreshold/path_label.pickle')

def getPreprocessedRFImage(rfImage, mask, threshold=True):
    """
    Apply Geometric and threshold masking to the input rf image
    Then take the magnitude of the complex values.
    convert data type to float32
    """
    imagePower = np.absolute(np.power(rfImage, 2)).astype('float32')
    maskG = mask.astype(bool)
    if threshold == True:
        imageMaxPower = np.max(imagePower)
        allowedDropRelPeak = 5
        maskT = (imagePower >= imageMaxPower/allowedDropRelPeak)
    imagePower[~ (maskG)] = 0
    return imagePower

def cal_accuracy(tp, total):
    return tp / total

def cal_precision(tp, fp):
    return tp / (tp + fp)

def cal_recall(tp, fn):
    return tp / (tp + fn)

def cal_f1_score(tp, fp, fn):
    return 2 * (cal_precision(tp, fp) * cal_recall(tp, fn)) / (cal_precision(tp, fp) + cal_recall(tp, fn))

def multiclass_metric(cm, metric='accuracy'):
    """
    Calculate the metric for each class.
    Input: confusion matrix (4x4), type of the metric (accuracy as default)
    Output: three different scores for each class (adult, kid and empty)
    """
    result = list()    #[ADT, KID, EMP]
    for i, row in enumerate(cm[:3,:]):
        tp = row[i]
        fp = np.sum(cm[:,i])-tp
        fn = np.sum(cm[i,:])-tp
        if metric == 'accuracy': score = cal_accuracy(tp, np.sum(row))
        elif metric == 'precision': score = cal_precision(tp, fp)
        elif metric == 'recall': score = cal_recall(tp, fn)
        elif metric == 'f1_score': score = cal_f1_score(tp, fp, fn)
        result.append(score)
    return np.array(result)

def plot_seat_wise_bar(score_array, metric='Accuracy'):
    """
    Plot metrics (e.g. accuracy, f1 score, precision etc.) for each seat
    """
    N = score_array.shape[0]
    ind = np.arange(N)  # the x locations for the groups
    width = 0.27       # the width of the bars

    fig = plt.figure()
    ax = fig.add_subplot(111)
    legend_labels = ["ADULT", "KID", "EMPTY"]
    xticks_labels = [f'Seat{x+1}' for x in range(N)]
    # for i in range(score_array.shape[1]):
    #     yvals = list(score_array[:,i])
    #     rects1 = ax.bar(ind, yvals, width, color=colors[i])
    #     rects1.set_label(legend_labels[i])
    
    yvals = list(score_array[:,0])
    rects1 = ax.bar(ind, yvals, width, color='r')
    rects1.set_label(legend_labels[0])
    zvals = list(score_array[:,1])
    rects2 = ax.bar(ind+width, zvals, width, color='g')
    rects2.set_label(legend_labels[1])
    kvals = list(score_array[:,2])
    rects3 = ax.bar(ind+width*2, kvals, width, color='b')
    rects3.set_label(legend_labels[2])

    ax.set_title(f'Seat Wise {metric}')
    ax.set_ylabel(metric)
    ax.set_xticks(ind+width)
    ax.set_xticklabels(xticks_labels)
    ax.legend(bbox_to_anchor=(1.05, 1))
# %%
