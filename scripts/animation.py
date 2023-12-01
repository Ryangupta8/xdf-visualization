import pyxdf
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from matplotlib.markers import MarkerStyle
from matplotlib.text import TextPath
from matplotlib.transforms import Affine2D
import numpy as np
import sys
import os
import math

sys.path.append('data')
sys.path.append('map')

def rot(x):
    # r = np.array([[-1,0],[0,-1]])
    r = np.array([[1,0],[0,1]])
    return r@x


vectormap_txt = "map/AHG_vectormap.txt"
def plot_vectormap(fname,figure,axes,facecolor="white",linecolor="black"):
    with open(fname,"r") as f:
        lines = f.readlines()
    axes.set_facecolor(facecolor)
    axes.set_xticks([])
    axes.set_yticks([])
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.spines['bottom'].set_visible(False)
    axes.spines['left'].set_visible(False)
    for line in lines:
        pts = [float(pt) for pt in [ln.split(", ") for ln in line.splitlines()][0]]
        x_,x = rot(np.array([[pts[0]],[pts[1]]])), rot(np.array([[pts[2]],[pts[3]]]))
        ln = axes.plot([x_[0],x[0]],[x_[1],x[1]],color=linecolor,zorder=0)
    fig.set_figwidth(20)
    fig.set_figheight(10)
    #fig.tight_layout()
    return fig,axes,ln


## Update function for the matplotlib animation
def update(frame):
    # update the scatter plot:
    data = np.stack([y1[frame,0], y1[frame,1]]).T
    scat1.set_offsets(data)
    data = np.stack([y1[:frame,0], y1[:frame,1]]).T
    scat2.set_offsets(data)
    yaw_ = math.atan2(2.0 * (y1[frame,5] * y1[frame,4]), y1[frame,5] * y1[frame,5] - y1[frame,4] * y1[frame,4])
    t = Affine2D().scale(7).rotate_deg(np.rad2deg(yaw_))
    m = MarkerStyle(TextPath((0, -3.1), ">"), transform=t)
    scat1.set_paths([MarkerStyle(m).get_path().transformed(MarkerStyle(m).get_transform())])

    # update the scatter plot:
    data = np.stack([y2[frame,0], y2[frame,1]]).T
    scat3.set_offsets(data)
    data = np.stack([y2[:frame,0], y2[:frame,1]]).T
    scat4.set_offsets(data)
    yaw_ = math.atan2(2.0 * (y2[frame,5] * y2[frame,4]), y2[frame,5] * y2[frame,5] - y2[frame,4] * y2[frame,4])
    t = Affine2D().scale(7).rotate_deg(np.rad2deg(yaw_))
    m = MarkerStyle(TextPath((0, -3.1), ">"), transform=t)
    scat3.set_paths([MarkerStyle(m).get_path().transformed(MarkerStyle(m).get_transform())])
    
    # print("ecg_srate: ",ecg_srate)
    # print("state_srate: ",state_srate)
    # print("ecg_srate/state_srate: ",ecg_srate/state_srate)
    # update the line plot:
    line1.set_xdata(stream3['time_stamps'][:int(frame*ecg_srate/state_srate)])
    line1.set_ydata(y3[:int(frame*ecg_srate/state_srate)])
    # print("stream3['time_stamps'][frame] = ",stream3['time_stamps'][frame])
    # print("y3[frame] = ",y3[frame][0])
    data = np.stack([stream3['time_stamps'][int(frame*ecg_srate/state_srate)], y3[int(frame*ecg_srate/state_srate)][0]]).T
    scat5.set_offsets(data)
    sub2.set_xlim(stream3['time_stamps'][0], stream3['time_stamps'][0] + frame / 14.5)

    # # update the line plot:
    line2.set_xdata(stream4['time_stamps'][:int(frame*ecg_srate/state_srate)])
    line2.set_ydata(y4[:int(frame*ecg_srate/state_srate)])
    data = np.stack([stream4['time_stamps'][int(frame*ecg_srate/state_srate)], y4[int(frame*ecg_srate/state_srate)][0]]).T
    scat6.set_offsets(data)
    sub3.set_xlim(stream4['time_stamps'][0], stream4['time_stamps'][0] + frame / 14.5)
    
    return (scat1) # (line1, line2, line3, line4, line5, line6)

xdf_file = 'session1/session1-trial2-social-separated-search.xdf'
# xdf_file = 'session1-trial8-isolated-P0livroom-P1smallroom-together-search.xdf'
## Load the xdf file
data, header = pyxdf.load_xdf('data/' + xdf_file)

## Create the figures
fig = plt.figure(figsize=(20,20))
fig.subplots_adjust(bottom=0.015, left=0.015, top = 0.985, right=0.985)
# sub1 = plt.subplot(1,2,1)
# sub2 = plt.subplot(3,2,5)
# sub3 = plt.subplot(3,2,6)
# sub4 = plt.subplot(3,2,7)
# sub5 = plt.subplot(3,2,8)
##
sub1 = plt.subplot(2,1,1)
sub2 = plt.subplot(4,2,5)
sub2.set_xticks([])
sub2.set_yticks([])
sub3 = plt.subplot(4,2,6)
sub3.set_xticks([])
sub3.set_yticks([])
sub4 = plt.subplot(4,2,7)
sub4.set_xticks([])
sub4.set_yticks([])
sub5 = plt.subplot(4,2,8)
sub5.set_xticks([])
sub5.set_yticks([])

# X = [ (1,2,1), (3,2,2), (3,2,4), (3,2,6) ]
# for nrows, ncols, plot_number in X:
#     plt.subplot(nrows, ncols, plot_number)
#     plt.xticks([])
#     plt.yticks([])

plot_vectormap(vectormap_txt, fig, sub1)

if 'social' in xdf_file:
    t = Affine2D().scale(2)
    m = MarkerStyle("^", transform=t)
    sub1.scatter(5.75, 3.2, c='#008000', marker=m)
    sub1.scatter(6.75, 3.2, c='#008000', marker=m)
else:
    t = Affine2D().scale(2)
    m = MarkerStyle("^", transform=t)
    sub1.scatter(5.75, 3.2, c='#008000', marker=m)
    t = Affine2D().scale(2).rotate_deg(-40)
    m = MarkerStyle("^", transform=t)
    sub1.scatter(10.0, 1.2, c='#008000', marker=m)


## loop thru all of the separate streams (e.g. go1_state; spot_state; etc.)
for stream in data:
    
    y = stream['time_series']
    # print("stream['time_stamps'].shape = ", stream['time_stamps'].shape)
    # print("y.shape = ",y.shape)
    

    ## Data in this example comes as np arrays
    if isinstance(y, np.ndarray):
        
        if "spot" in str(stream['info']['name']):
            ## Stream name
            print("stream name: ", stream['info']['name'])
            ## ex: spot/go1_state type = xy_rxryrzrw
            print("stream type: ", stream['info']['type'])
            ## Stream dtype
            print("stream channel format: ", stream['info']['channel_format'])
            ## effective rate
            print("Actual sample rate: ",stream['info']['effective_srate'])
            state_srate = float(stream['info']['effective_srate'])
            print('-----------------------')
            ## The actual data points
            y1 = stream['time_series']
            ## spot/go1_state shape is (n_timesteps,6), i think EDA/ECG maybe similar? 
            print("y.shape = ",y1.shape) 

            _yaw = math.atan2(2.0 * (y1[0,5] * y1[0,4]), y1[0,5] * y1[0,5] - y1[0,4] * y1[0,4])
            # print("yaw = ", _yaw)
            # print("np.rad2deg(_yaw) = ",np.rad2deg(_yaw))
            t = Affine2D().scale(7).rotate_deg(np.rad2deg(_yaw))
            m = MarkerStyle(TextPath((0, 0), ">"), transform=t)
            ## For plotting the oriented robot position
            scat1 = sub1.scatter(y1[0,0], y1[0,1], c='#0000FF', marker=m)
            ## For plotting the full robot path as it moves
            scat2 = sub1.scatter(y1[0,0], y1[0,1], c='#0000FF', marker=".")
            


        if "go1" in str(stream['info']['name']):
            ## Stream name
            print("stream name: ", stream['info']['name'])
            ## ex: spot/go1_state type = xy_rxryrzrw
            print("stream type: ", stream['info']['type'])
            ## Stream dtype
            print("stream channel format: ", stream['info']['channel_format'])
            ## effective rate
            print("Actual sample rate: ",stream['info']['effective_srate'])
            print('-----------------------')

            ## The actual data points
            y2 = stream['time_series']

            _yaw = math.atan2(2.0 * (y2[0,5] * y2[0,4]), y2[0,5] * y2[0,5] - y2[0,4] * y2[0,4])
            # print("yaw = ", _yaw)
            # print("np.rad2deg(_yaw) = ",np.rad2deg(_yaw))
            t = Affine2D().scale(7).rotate_deg(np.rad2deg(_yaw))
            m = MarkerStyle(TextPath((0, 0), ">"), transform=t)
            ## For plotting the oriented robot position
            scat3 = sub1.scatter(y2[0,0], y2[0,1], c='#FF0000', marker=m)
            ## For plotting the full robot path as it moves
            scat4 = sub1.scatter(y2[0,0], y2[0,1], c='#FF0000', marker=".")

        
        if ("ECG-D4" in str(stream['info']['name'])) and (y.shape[0] > 1):
            ## Stream name
            print("stream name: ", stream['info']['name'])
            ## ex: spot/go1_state type = xy_rxryrzrw
            print("stream type: ", stream['info']['type'])
            ## Stream dtype
            print("stream channel format: ", stream['info']['channel_format'])
            
            y3 = stream['time_series']
            print("y3.shape = ",y3.shape)
            ## effective rate
            print("Actual sample rate: ",stream['info']['effective_srate'])
            ecg_srate = float(stream['info']['effective_srate'])
            print('-----------------------')

            line1 = sub2.plot(stream['time_stamps'], y3[:], color="green")[0]
            scat5 = sub2.scatter(stream['time_stamps'][0], y3[0], c='orange')
            stream3 = stream

        if ("ECG-FA" in str(stream['info']['name'])) and (y.shape[0] > 1):
            ## Stream name
            print("stream name: ", stream['info']['name'])
            ## ex: spot/go1_state type = xy_rxryrzrw
            print("stream type: ", stream['info']['type'])
            ## Stream dtype
            print("stream channel format: ", stream['info']['channel_format'])
            y4 = stream['time_series']
            print("y4.shape = ",y4.shape)
            ## effective rate
            print("Actual sample rate: ",stream['info']['effective_srate'])
            print('-----------------------')

            line2 = sub3.plot(stream['time_stamps'], y4[:], color="green")[0]
            scat6 = sub3.scatter(stream['time_stamps'][0], y4[0], c='orange')
            stream4 = stream

    else:
        raise RuntimeError('Unknown stream format')

ani = animation.FuncAnimation(fig=fig, func=update, interval=1) ## interval=1 speeds up the animation
plt.show()

