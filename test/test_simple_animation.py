import pyxdf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import sys
import os
sys.path.append('data')

## Update function for the matplotlib animation
def update(frame):
    # update the line plot:
    line1.set_xdata(stream['time_stamps'][:frame])
    line1.set_ydata(y[:frame,0])
    # update the line plot:
    line2.set_xdata(stream['time_stamps'][:frame])
    line2.set_ydata(y[:frame,1])
    # update the line plot:
    line3.set_xdata(stream['time_stamps'][:frame])
    line3.set_ydata(y[:frame,2])
    # update the line plot:
    line4.set_xdata(stream['time_stamps'][:frame])
    line4.set_ydata(y[:frame,3])
    # update the line plot:
    line5.set_xdata(stream['time_stamps'][:frame])
    line5.set_ydata(y[:frame,4])
    # update the line plot:
    line6.set_xdata(stream['time_stamps'][:frame])
    line6.set_ydata(y[:frame,5])
    return (line1, line2, line3, line4, line5, line6)

## Load the xdf file
data, header = pyxdf.load_xdf('data/teest.xdf')

## Create the figure
fig, ax = plt.subplots()

## loop thru all of the separate streams (e.g. go1_state; spot_state; etc.)
for stream in data:
    ## Stream name
    print("stream[name] = ", stream['info']['name'])
    ## ex: spot/go1_state type = xy_rxryrzrw
    print("stream[type] = ", stream['info']['type'])
    ## Stream dtype
    print("stream[channel_format] = ", stream['info']['channel_format'])
    ## The actual data points
    y = stream['time_series']
    ## spot/go1_state shape is (n_timesteps,6), i think EDA/ECG maybe similar? 
    print("y.shape = ",y.shape) 

    ## Comes from the pyxdf README example, but unused here
    if isinstance(y, list):
        # list of strings, draw one vertical line for each marker
        for timestamp, marker in zip(stream['time_stamps'], y):
            plt.axvline(x=timestamp)
            print(f'Marker "{marker[0]}" @ {timestamp:.2f}s')
    ## Data in this example comes as np arrays
    elif isinstance(y, np.ndarray):
        ## Plot a simple line plot of all of the time series in this stream
        # plt.plot(stream['time_stamps'], y)

        print("stream['time_stamps'].shape = ", stream['time_stamps'].shape)
        print("y[:,0].shape = ",y[:,0].shape)

        ## We want one line for each of the spot_state time series in this example
        line1 = ax.plot(stream['time_stamps'], y[:,0])[0]
        line2 = ax.plot(stream['time_stamps'], y[:,1])[0]
        line3 = ax.plot(stream['time_stamps'], y[:,2])[0]
        line4 = ax.plot(stream['time_stamps'], y[:,3])[0]
        line5 = ax.plot(stream['time_stamps'], y[:,4])[0]
        line6 = ax.plot(stream['time_stamps'], y[:,5])[0]
        # ax.legend()

         

    else:
        raise RuntimeError('Unknown stream format')

ani = animation.FuncAnimation(fig=fig, func=update, interval=1) ## interval=1 speeds up the animation
plt.show()

