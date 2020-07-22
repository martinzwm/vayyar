#%%
import h5py
import numpy as np
from utilities import importDataOccupancyType

#%%
x, y, occupiedSeat, occupantType, path = importDataOccupancyType("/Users/jameshe/Documents/radar_ura/vayyar/FirstBatch")

#%%
with h5py.File('training_dataset.hdf5', 'w') as f:
    f.create_dataset('x', data=x)
    f.create_dataset('y', data=y)
    f.create_dataset('occupiedSeat', data=occupiedSeat)
    f.create_dataset('occupantType', data=occupantType)
    f.create_dataset('path', data=path)


# %%
