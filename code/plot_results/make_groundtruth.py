import h5py
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import gridspec, colors  # Import gridspec and colors from matplotlib
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from IPython.display import display

# please adjust here if you wish to save the ground truth data to another directory
BASE_SAVE_DIR = os.path.join(os.getcwd(), os.pardir, os.pardir, "exp1-4_data", "data_prepped_for_models")
os.makedirs(BASE_SAVE_DIR, exist_ok=True)

# neural data ------------------------------------------------------------------------------------------------------------------------------------------------
# if you wish to reconstruct the matched neural and tail data files, point these to the raw data folders...
DATA_DIR = os.path.join(os.getcwd(), os.pardir, os.pardir)
fish9_data_folder = f"{DATA_DIR}/raw_data_f9"
fish10_data_folder = f"{DATA_DIR}/raw_data_f10"
fish11_data_folder = f"{DATA_DIR}/raw_data_f11"
fish12_data_folder = f"{DATA_DIR}/raw_data_f12"
fish13_data_folder = f"{DATA_DIR}/raw_data_f13"
num_planes = 6

# !!! CHANGE TO FISH YOU ARE BUILDING DATA FOR...---------------------------------------------------------------------------
fish = 13
# --------------------------------------------------------------------------------------------------------------------------

arrays = []
for plane_idx in range(num_planes):
    if fish==9:
        plane_path = os.path.join(fish9_data_folder, f'plane_{plane_idx}', 'F.npy')
    elif fish==10:
        plane_path = os.path.join(fish10_data_folder, f'plane_{plane_idx}', 'F.npy')
    elif fish==11:
        plane_path = os.path.join(fish11_data_folder, f'plane_{plane_idx}', 'F.npy')
    elif fish==12:
        plane_path = os.path.join(fish12_data_folder, f'plane_{plane_idx}', 'F.npy')
    elif fish==13:
        plane_path = os.path.join(fish13_data_folder, f'plane_{plane_idx}', 'F.npy')
    if not os.path.exists(plane_path):
        raise FileNotFoundError(f"File not found: {plane_path}")
    array = np.load(plane_path)
    arrays.append(array)
    print(f"Plane {plane_idx} loaded with shape: {array.shape}")

# Concatenate all planes (neurons x time_bins)
full_array = np.concatenate(arrays, axis=0)
print(f"Full array shape: {full_array.shape}")

# Standardize (zero mean, unit variance) and then normalize to [0,1]
scaler_standard = StandardScaler()
standardized_array = scaler_standard.fit_transform(full_array)
print("Standardization complete.")

scaler_minmax = MinMaxScaler()
normalized_array = scaler_minmax.fit_transform(standardized_array)
print("Normalization complete.")

print(normalized_array.shape) # num_neurons by num_time bins
np.save(f"{BASE_SAVE_DIR}/fish{fish}_neural_data_full.npy", normalized_array)
np.save(f"{BASE_SAVE_DIR}/fish{fish}_neural_data_matched.npy", normalized_array[:,:4462])


# tail data --------------------------------------------------------------------------------------------------------------------------------------------------
if fish==9:
    tail_data_loc = f'{fish9_data_folder}/plane_0/tail_df.h5'
elif fish==10:
    tail_data_loc = f'{fish10_data_folder}/plane_0/tail_df.h5'
elif fish==11:
    tail_data_loc = f'{fish11_data_folder}/plane_0/tail_df.h5'
elif fish==12:
    tail_data_loc = f'{fish12_data_folder}/plane_0/tail_df.h5'
elif fish==13:
    tail_data_loc = f'{fish13_data_folder}/plane_0/tail_df.h5'

# extract tail data from h5file
with h5py.File(tail_data_loc, "r") as h5file:
    block0_items = [item.decode('utf-8') for item in h5file["tail/block0_items"][:]]
    block0_values = h5file["tail/block0_values"][:]
    block1_items = [item.decode('utf-8') for item in h5file["tail/block1_items"][:]]
    block1_values = h5file["tail/block1_values"][:]
    block2_items = [item.decode('utf-8') for item in h5file["tail/block2_items"][:]]
    block2_values = h5file["tail/block2_values"][:]
    df_block0 = pd.DataFrame(block0_values, columns=block0_items)
    df_block1 = pd.DataFrame(block1_values, columns=block1_items)
    df_block2 = pd.DataFrame(block2_values, columns=block2_items)
    tail_df = pd.concat([df_block0, df_block1, df_block2], axis=1)

# display useful tail data df
tail_df_useful = tail_df[["theta_00", "theta_01","theta_02", "theta_03", "theta_04", "theta_05", "t","frame"]]
display("Tail dataframe display: \n", tail_df_useful)

# just use the array for simulation
tail_array = tail_df_useful.values
print(tail_array.shape)
np.save(f"{BASE_SAVE_DIR}/fish{fish}_tail_data_full.npy", tail_array)

# merge data in given frame
# Define the tail component columns
tail_components = ["theta_00", "theta_01", "theta_02", "theta_03", "theta_04", "theta_05"]
# Group by 'frame' and compute the mean for each tail component
tail_avg = tail_df_useful.groupby("frame")[tail_components].mean()
print(tail_avg.values.shape)
print(tail_avg.values)
np.save(f"{BASE_SAVE_DIR}/fish{fish}_tail_data_matched.npy", tail_avg.values)
