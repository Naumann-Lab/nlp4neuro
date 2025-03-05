import numpy as np
import pandas as pd
import h5py

from IPython.display import display

tail_data_loc = '../../data/fish11_iMm_complete_stim/fish11_iMm_complete_stim/plane_0/tail_df.h5'

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
tail_df_useful2 = tail_df[["theta_00", "theta_01","theta_02", "theta_03", "theta_04", "theta_05"]]
display("Tail dataframe display: \n", tail_df_useful2)

# just use the array for simulation
tail_array = tail_df_useful2.values[0:200000,:]
print(tail_array.shape)
np.save("tail_component_array.npy", tail_array)
