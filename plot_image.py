#%% This script is to plot the rfImage of FirstBatch and vCab_Recordings. Provided by Vayyar Imaging.
from scipy import io
import numpy as np
import matplotlib.pyplot as plt

UseLkg = True
FlattenAxis = 0     # 0, 1, 2
Batch = 2           # 0 - Sample, 1 - vCab, 2 - vTrig

if Batch == 0:
    file = "/home/vayyar_data/FirstBatch/Copy (104) - Copy__30-10-2019 14-03-26/SavedVars_RF_Data/Vars013.mat"
    lkgFile = "/home/vayyar_data/FirstBatch/Copy (104) - Copy__30-10-2019 14-03-26/SavedVars_RF_Data/Vars001.mat"
elif Batch == 1:
    file = "/home/vayyar_data/vCab_Recordings/16.10/ford_1/Copy (149) - Copy__16-10-2019 13-04-34/rfImages/037.mat"
    lkgFile = "/home/vayyar_data/vCab_Recordings/16.10/ford_1/Copy (149) - Copy__16-10-2019 13-04-34/rfImages/001.mat"
elif Batch == 2:
    file = "/home/vayyar_data/vTrig_Recordings/GeneralOccupancy/13-14.11.19/2019_11_13__16_30_44/rfImages/043.mat"
    lkgFile = "/home/vayyar_data/vTrig_Recordings/GeneralOccupancy/13-14.11.19/2019_11_13__16_30_44/rfImages/001.mat"

mat = io.loadmat(file)
if 'rfImageStruct' in mat:
    imageFormat = 'NEW'
    img = mat['rfImageStruct']['image_DxDyR'][0, 0]
    if UseLkg:
        matLkg = io.loadmat(lkgFile)
        lkg = matLkg['rfImageStruct']['image_DxDyR'][0, 0]
else:
    imageFormat = 'OLD'
    img = mat['Image']
    if UseLkg:
        matLkg = io.loadmat(lkgFile)
        lkg = mat['Image']

shape = img.shape
lkg = np.load("/home/vayyar_data/processed_vCab_Recordings_nonthreshold/empty.npy").reshape(29,29,24)

if UseLkg:
    img = np.subtract(img, lkg)

print(f'Mat loaded. Image format: {imageFormat}, Shape: {shape}')
print(f'Image Min = {np.min(np.abs(img))}')
print(f'Image Max = {np.max(np.abs(img))}')

flattened = np.rot90(np.sum(np.abs(img), axis=FlattenAxis), k=3)

plt.imshow(np.real(flattened))
plt.show()

# %%
