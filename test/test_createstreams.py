import pyxdf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import sys

sys.path.append('data')


## Load the xdf file
data, header = pyxdf.load_xdf('data/teest.xdf')

## Create the figure
fig, ax = plt.subplots()

## loop thru all of the separate streams 
for stream in data:
    ## Stream name (e.g. go1_state; spot_state; etc.)
    print("stream[name] = ", stream['info']['name'])
    ## ex: spot/go1_state type = xy_rxryrzrw
    print("stream[type] = ", stream['info']['type'])
    ## Stream dtype
    print("stream[channel_format] = ", stream['info']['channel_format'])
    ## The actual data points
    y = stream['time_series']
    ## spot/go1_state shape is (n_timesteps,6), i think EDA/ECG maybe similar? 
    print("y.shape = ",y.shape) 

    ## Data in this example comes as np arrays
    if isinstance(y, np.ndarray):

        print("stream['time_stamps'].shape = ", stream['time_stamps'].shape)
        print("y[:,0].shape = ",y[:,0].shape)
    else:
        raise RuntimeError('Unknown stream format')

plt.show()