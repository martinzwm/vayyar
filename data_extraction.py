import pandas as pd
import numpy as np
import os
import pickle
import torch
import json
import copy
from utilities import loadmat
from torch.utils.data import Dataset


def makeVcabPickleFile2(
    root_dir,
    output_dir,
    crop_r=True,
    num_frame_sum=10,
    remove_clutter=True,
    ):
    '''
    A 'processed' folder will be created under the output_dir. A 'car_name' folder will be
    created under the 'processed' folder for each type of car (e.x. ford1, ford2, ...).

    Inputs:
        root_dir: the subfolder should be the 'date level' folder
        output_dir: a folder named 'processed' will be created to store the processed data
        crop_r: whether r dimension should be cropped to 24, note that summing over frames
                may not work if the r dimension is not cropped
        num_frame_sum: number of frame we want to sum over
    Outputs: null
    '''

    output_dir += '\processed'
    if os.path.isdir(output_dir):
        raise NameError("{} already exists!".format(output_dir))
    else:
        os.mkdir(output_dir)

    # We want to separate dataset by different cars, so we could train on one and test on the other.
    # The key of data is the car_name, the value is the data_template (make sure you use deepcopy here
    # to avoid reference semantics).
    data = {} 
    data_template = {"path_original": [],
            "seat_occupied": [],
            "occupancy_type": [],
            "processed_filename": [],
            "car_model": [],
            "car_info": []}

    index = 0
    for dayLevelItem in os.scandir(root_dir): #recording time level
        if dayLevelItem.is_dir():
            # omit out of position detectionf or now, 'processed' is usually where we keep the data
            if 'OOP' not in dayLevelItem.name and dayLevelItem.name != 'processed':
                for carLevelItem in os.scandir(dayLevelItem.path):#car level
                    if carLevelItem.is_dir():
                        print('Currently processing {}'.format(carLevelItem.path))
                        if carLevelItem.name not in data:
                            data[carLevelItem.name] = copy.deepcopy(data_template)
                            processed_car_path = os.path.join(output_dir, carLevelItem.name)
                            os.mkdir(processed_car_path)
                        for minuteLevelItem in os.scandir(carLevelItem.path):#minute level, e.g. v_Copy (12) - Copy__04-11-2019 15-23-54
                            if minuteLevelItem.is_dir():
                                with open(os.path.join(minuteLevelItem.path, "test_data.json")) as labelFile:
                                    # Get labels for this recording
                                    labels = json.load(labelFile)
                                    occupancyLabel = make_labels(labels["Occupied_Seats"])
                                    occupancyTypeLabel = labels["Occupant_Type"]
                                    carModel = labels["Car_Model"] # to improve performance at *this stage*, we may not want to mix car models

                                    # Get first frame for clutter removal
                                    first_frame_path = os.path.join(minuteLevelItem.path, "rfImages", "001")
                                    first_rfImage_struct = loadmat(first_frame_path)['rfImageStruct']
                                    first_frame = first_rfImage_struct['image_DxDyR']
                                    if crop_r == True:
                                        first_frame = first_frame[:,:,:24]

                                    ctr = 0 # counter for taking the sum of num_frame_sum
                                    image_sum = np.zeros((29,29,24), dtype=np.complex128)
                                    for file in os.scandir(os.path.join(minuteLevelItem.path, "rfImages")): #frame level
                                        if file.name.endswith('.mat'):
                                            try:
                                                rfImageStruct = loadmat(file.path)['rfImageStruct']
                                                image = rfImageStruct['image_DxDyR']
                                                if crop_r == True:
                                                    image = image[:,:,:24]
                                                if remove_clutter == True:
                                                    if '001' in file.name:
                                                        continue
                                                    else:
                                                        image -= first_frame
                                                image_sum += image
                                                ctr += 1
                                            except Exception as ex:
                                                template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                                                message = template.format(type(ex).__name__, ex.args)
                                                print(message)
                                                print(file.path)
                                            
                                            if ctr % num_frame_sum == 0:
                                                data[carLevelItem.name]["path_original"].append(file.path)
                                                data[carLevelItem.name]["processed_filename"].append(index)
                                                data[carLevelItem.name]["seat_occupied"].append(occupancyLabel)
                                                data[carLevelItem.name]["occupancy_type"].append(occupancyTypeLabel)
                                                data[carLevelItem.name]["car_model"].append(carModel)
                                                data[carLevelItem.name]["car_info"].append(carLevelItem.name) # e.g. ford1
                                                # Note the '\%s.npy' needs to be replaced by '/%s.npy' if running on linux or mac system
                                                processed_csv = os.path.join(output_dir, carLevelItem.name) + '\%s.npy' % (str(index))
                                                np.save(processed_csv, np.absolute(image_sum))
                                                index += 1
                                                image_sum = np.zeros((29,29,24), dtype=np.complex128)
        
    # Create a pandas dataframe out from this dictionary.
    for car_name in data:
        df = pd.DataFrame.from_dict(data[car_name])
        path_label = os.path.join(output_dir, car_name) + '\path_label.pickle'
        df.to_pickle(path_label)

def makeVtrigPickleFile2(
    root_dir,
    output_dir,
    crop_r=True,
    num_frame_sum=10,
    remove_clutter=True,
    ):
    '''
    Note that the vTrig data is organized in a different folder structure than vCab.
    
    A 'processed' folder will be created under the output_dir. A 'car_name' folder will be
    created under the 'processed' folder for each type of car (e.x. ford1, ford2, ...).

    Inputs:
        root_dir: the subfolder should be the 'date level' folder
        output_dir: a folder named 'processed' will be created to store the processed data
        crop_r: whether r dimension should be cropped to 24, note that summing over frames
                may not work if the r dimension is not cropped
        num_frame_sum: number of frame we want to sum over
    Outputs: null
    '''

    output_dir += '\processed'
    if os.path.isdir(output_dir):
        raise NameError("{} already exists!".format(output_dir))
    else:
        os.mkdir(output_dir)

    # We want to separate dataset by different cars, so we could train on one and test on the other.
    # The key of data is the car_name, the value is the data_template (make sure you use deepcopy here
    # to avoid reference semantics).
    data = {} 
    data_template = {"path_original": [],
            "seat_occupied": [],
            "occupancy_type": [],
            "processed_filename": [],
            "car_model": [],
            "car_info": []}

    for carLevelItem in os.scandir(root_dir):#car level
        # 'processed' folder is where we keep the processed data
        if carLevelItem.is_dir() and carLevelItem.name != 'processed':
            print('Currently processing {}'.format(carLevelItem.path))
            index = 0
            if carLevelItem.name not in data:
                data[carLevelItem.name] = copy.deepcopy(data_template)
                processed_car_path = os.path.join(output_dir, carLevelItem.name)
                os.mkdir(processed_car_path)
            for dayLevelItem in os.scandir(carLevelItem.path): #recording time level
                if dayLevelItem.is_dir():
                    # omit out of position detectionf or now
                    if 'OOP' not in dayLevelItem.name:
                        for minuteLevelItem in os.scandir(dayLevelItem.path):#minute level, e.g. v_Copy (12) - Copy__04-11-2019 15-23-54
                            if minuteLevelItem.is_dir():
                                with open(os.path.join(minuteLevelItem.path, "test_data.json")) as labelFile:
                                    # Get labels for this recording
                                    labels = json.load(labelFile)
                                    occupancyLabel = make_labels(labels["Occupied_Seats"])
                                    occupancyTypeLabel = labels["Occupant_Type"]
                                    carModel = labels["Car_Model"] # to improve performance at *this stage*, we may not want to mix car models

                                    # Get first frame for clutter removal
                                    first_frame_path = os.path.join(minuteLevelItem.path, "rfImages", "001")
                                    first_rfImage_struct = loadmat(first_frame_path)['rfImageStruct']
                                    first_frame = first_rfImage_struct['image_DxDyR']
                                    if crop_r == True:
                                        first_frame = first_frame[:,:,:24]

                                    ctr = 0 # counter for taking the sum of num_frame_sum
                                    image_sum = np.zeros((29,29,24), dtype=np.complex128)
                                    for file in os.scandir(os.path.join(minuteLevelItem.path, "rfImages")): #frame level
                                        if file.name.endswith('.mat'):
                                            try:
                                                rfImageStruct = loadmat(file.path)['rfImageStruct']
                                                image = rfImageStruct['image_DxDyR']
                                                if crop_r == True:
                                                    image = image[:,:,:24]
                                                if remove_clutter == True:
                                                    if '001' in file.name:
                                                        continue
                                                    else:
                                                        image -= first_frame
                                                image_sum += image
                                                ctr += 1
                                            except Exception as ex:
                                                template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                                                message = template.format(type(ex).__name__, ex.args)
                                                print(message)
                                                print(file.path)
                                            
                                            if ctr % num_frame_sum == 0:
                                                data[carLevelItem.name]["path_original"].append(file.path)
                                                data[carLevelItem.name]["processed_filename"].append(index)
                                                data[carLevelItem.name]["seat_occupied"].append(occupancyLabel)
                                                data[carLevelItem.name]["occupancy_type"].append(occupancyTypeLabel)
                                                data[carLevelItem.name]["car_model"].append(carModel)
                                                data[carLevelItem.name]["car_info"].append(carLevelItem.name) # e.g. ford1
                                                # Note the '\%s.npy' needs to be replaced by '/%s.npy' if running on linux or mac system
                                                processed_csv = os.path.join(output_dir, carLevelItem.name) + '\%s.npy' % (str(index))
                                                np.save(processed_csv, np.absolute(image_sum))
                                                index += 1
                                                image_sum = np.zeros((29,29,24), dtype=np.complex128)
        
    # Create a pandas dataframe out from this dictionary.
    for car_name in data:
        df = pd.DataFrame.from_dict(data[car_name])
        path_label = os.path.join(output_dir, car_name) + '\path_label.pickle'
        df.to_pickle(path_label)


# def make_labels(occupancyLabel):
#     '''
#     Convert occupied seat information from Vayyar format to our format
#     Inputs:
#         occupancyLabel(list): the list of seats (1-5) that are occupied
#     Ouputs:
#         label(numpy array of length 5): each number represents whether the corresponding seat (1-5) is occupied
#     '''
#     label = []
#     for seat_id in range(1,6):
#         if seat_id in occupancyLabel:
#             label.append(1)
#         else:
#             label.append(0)
#     return np.array(label).astype('uint8')


def make_labels(occupancyLabel):
    '''
    Written by Hajar
    Convert occupied seat information from Vayyar format to a number (0-31)
    '''
    label = []
    a = occupancyLabel
    if a == [1]:
        label.append(1)
    elif a == [2]:
        label.append(2)
    elif a == []:
        label.append(0)
    elif a == [3]:
        label.append(3)
    elif a == [4]:
        label.append(4)
    elif a == [5]:
        label.append(5)
    elif a == [12]:
        label.append(6)
    elif a == [1, 3]:
        label.append(7)
    elif a == [1, 4]:
        label.append(8)
    elif a == [1, 5]:
        label.append(9)
    elif a == [2, 3]:
        label.append(10)
    elif a == [2, 4]:
        label.append(11)
    elif a == [2, 5]:
        label.append(12)
    elif a == [3, 4]:
        label.append(13)
    elif a == [3, 5]:
        label.append(14)
    elif a == [4, 5]:
        label.append(15)
    elif a == [1, 2, 3]:
        label.append(16)
    elif a == [1, 2, 4]:
        label.append(17)
    elif a == [1, 2, 5]:
        label.append(18)
    elif a == [1, 3, 4]:
        label.append(19)
    elif a == [1, 3, 5]:
        label.append(20)
    elif a == [1, 4, 5]:
        label.append(21)
    elif a == [2, 3, 4]:
        label.append(22)
    elif a == [2, 3, 5]:
        label.append(23)
    elif a == [2, 4, 5]:
        label.append(24)
    elif a == [3, 4, 5]:
        label.append(25)
    elif a == [1, 2, 3, 4]:
        label.append(26)
    elif a == [1, 2, 3, 5]:
        label.append(27)
    elif a == [1, 2, 4, 5]:
        label.append(28)
    elif a == [1, 3, 4, 5]:
        label.append(29)
    elif a == [2, 3, 4, 5]:
        label.append(30)
    elif a == [1, 2, 3, 4, 5]:
        label.append(31)
    return np.array(label)

class rfImageDataSet2(Dataset):
    def __init__(self, rootDir, transform = None):
        self.rootDir = rootDir
        self.path_label = pd.read_pickle(os.path.join(rootDir, "path_label.pickle"))
        self.transform = transform

    def __len__(self):
        return len(self.path_label)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()
        rfImagePath = os.path.join(self.rootDir, str(self.path_label.iloc[idx, 3]) + '.npy')
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
        return mean/N, std/N

# # To extract 'processed' data:
# # A folder named 'processed' will be created under the output_dir to store results

# makeVcabPickleFile2(
#     root_dir=r'B:\Vayyar_Dataset\vCab_Recordings',
#     output_dir=r'B:\Vayyar_Dataset\vCab_Recordings'
#     )

# OR

# makeVtrigPickleFile2(
#     root_dir=r'B:\Vayyar_Dataset\Multicar',
#     output_dir=r'B:\Vayyar_Dataset\Multicar'
#     )

# # To create a dataset for training and testing
# dataset = rfImageDataSet2(r'B:\Vayyar_Dataset\Multicar\processed\Hyundai_i10')
# # Each sample in the dataset consists image, the label, and other info
# print(dataset[0])
# # To get a distribution of the classes
# print(dataset.class_distribution())
# # To get the mean and std of the entire dataset
# print(dataset.mean_and_std())