# %%
import os
import numpy as np
import json
import scipy.io as sio
import pickle
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.autograd import Variable
import torch.nn as nn
from torch.optim import *
from utilities import importDataFromMatFiles, loadData, scenarioWiseTransformLabels, getConfusionMatrices, seatWiseTransformLabels, plot_seat_wise_bar, multiclass_metric, makeVcabPickleFile2
from models import CNNModel, CNNModelRC
from torchvision import transforms
from data_prep import rfImageDataSet, cropR
#import pkbar
import math
from torch.utils.tensorboard import SummaryWriter
import argparse
import seaborn as sn


makeVcabPickleFile2()
raise NameError('Pause!')

#vcab_recordings
transform = transforms.Compose([
            cropR(24),
            transforms.ToTensor(),
            transforms.Normalize(mean=[-0.05493089184165001],
                                 std=[0.035751599818468094])
        ])
