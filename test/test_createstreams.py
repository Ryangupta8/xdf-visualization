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
    # update the spot state plots:
    data = np.stack([spot_series[frame,0], spot_series[frame,1]]).T
    scat1.set_offsets(data)
    data = np.stack([spot_series[:frame,0], spot_series[:frame,1]]).T
    scat2.set_offsets(data)
    yaw_ = math.atan2(2.0 * (spot_series[frame,5] * spot_series[frame,4]), spot_series[frame,5] * spot_series[frame,5] - spot_series[frame,4] * spot_series[frame,4])
    t = Affine2D().scale(7).rotate_deg(np.rad2deg(yaw_))
    m = MarkerStyle(TextPath((0, -3.1), ">"), transform=t)
    scat1.set_paths([MarkerStyle(m).get_path().transformed(MarkerStyle(m).get_transform())])

    # update go1 state plots:
    data = np.stack([go1_series[frame,0], go1_series[frame,1]]).T
    scat3.set_offsets(data)
    data = np.stack([go1_series[:frame,0], go1_series[:frame,1]]).T
    scat4.set_offsets(data)
    yaw_ = math.atan2(2.0 * (go1_series[frame,5] * go1_series[frame,4]), go1_series[frame,5] * go1_series[frame,5] - go1_series[frame,4] * go1_series[frame,4])
    t = Affine2D().scale(7).rotate_deg(np.rad2deg(yaw_))
    m = MarkerStyle(TextPath((0, -3.1), ">"), transform=t)
    scat3.set_paths([MarkerStyle(m).get_path().transformed(MarkerStyle(m).get_transform())])

    ## 
    line1.set_xdata(spot_stream['time_stamps'][:frame])
    line1.set_ydata(spot_moving[:frame])
    sub2.set_ylim(bottom=-0.5, top=1.5)
    line2.set_xdata(go1_stream['time_stamps'][:frame])
    line2.set_ydata(go1_moving[:frame])
    sub3.set_ylim(bottom=-0.5, top=1.5)

    

    
    return (scat1) # (line1, line2, line3, line4, line5, line6)

## Load the xdf file
# xdf_file = 'session1-trial8-isolated-P0livroom-P1smallroom-together-search.xdf'
xdf_file = 'session1/session1-trial2-social-separated-search.xdf'
data, header = pyxdf.load_xdf('data/' + xdf_file)

## Create the figure
fig = plt.figure(figsize=(4,6))
fig.subplots_adjust(bottom=0.015, left=0.015, top = 0.985, right=0.985)

## Arrange subplots
sub1 = plt.subplot(2,1,1)
sub2 = plt.subplot(4,2,5)
sub2.set_xticks([])
# sub2.set_yticks([])
sub2.set_title("Spot Moving Boolean")
sub3 = plt.subplot(4,2,6)
sub3.set_xticks([])
# sub3.set_yticks([])
sub3.set_title("Go1 Moving Boolean")
sub4 = plt.subplot(4,2,7)
sub4.set_title("Participant 1 SCG Signals")
sub4.set_xticks([])
sub4.set_yticks([])
sub5 = plt.subplot(4,2,8)
sub5.set_title("Participant 2 SCG Signals")
sub5.set_xticks([])
sub5.set_yticks([])


plot_vectormap(vectormap_txt, fig, sub1)

if 'social' in xdf_file:
    subj1_pos = [5.75, 3.2]
    subj2_pos = [6.75,3.2]
    t = Affine2D().scale(2)
    m = MarkerStyle("^", transform=t)
    sub1.scatter(5.75, 3.2, c='#008000', marker=m)
    sub1.scatter(6.75, 3.2, c='#008000', marker=m)
else:
    subj1_pos = [5.75, 3.2]
    subj2_pos = [10.0, 1.2]
    t = Affine2D().scale(2)
    m = MarkerStyle("^", transform=t)
    sub1.scatter(5.75, 3.2, c='#008000', marker=m)
    t = Affine2D().scale(2).rotate_deg(-40)
    m = MarkerStyle("^", transform=t)
    sub1.scatter(10.0, 1.2, c='#008000', marker=m)



## loop thru all of the separate streams (e.g. go1_state; spot_state; etc.)
for stream in data:

    # print("stream name: ", stream['info']['name'])
    
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
            ## spot/go1_state shape is (n_timesteps,6), i think EDA/ECG maybe similar? 
            # print("y.shape = ",spot_series.shape) 
            print('-----------------------')
            ## The actual data points
            spot_series = stream['time_series']
            
            _yaw = math.atan2(2.0 * (spot_series[0,5] * spot_series[0,4]), spot_series[0,5] * spot_series[0,5] - spot_series[0,4] * spot_series[0,4])
            # print("np.rad2deg(_yaw) = ",np.rad2deg(_yaw))
            t = Affine2D().scale(7).rotate_deg(np.rad2deg(_yaw))
            m = MarkerStyle(TextPath((0, 0), ">"), transform=t)
            ## For plotting the oriented robot position
            scat1 = sub1.scatter(spot_series[0,0], spot_series[0,1], c='#0000FF', marker=m)
            ## For plotting the full robot path as it moves
            scat2 = sub1.scatter(spot_series[0,0], spot_series[0,1], c='#0000FF', marker=".", label='Spot')
            sub1.legend(labelcolor='linecolor', loc='lower left')

            spot_stream = stream
            

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
            go1_series = stream['time_series']

            _yaw = math.atan2(2.0 * (go1_series[0,5] * go1_series[0,4]), go1_series[0,5] * go1_series[0,5] - go1_series[0,4] * go1_series[0,4])
            # print("yaw = ", _yaw)
            # print("np.rad2deg(_yaw) = ",np.rad2deg(_yaw))
            t = Affine2D().scale(7).rotate_deg(np.rad2deg(_yaw))
            m = MarkerStyle(TextPath((0, 0), ">"), transform=t)
            ## For plotting the oriented robot position
            scat3 = sub1.scatter(go1_series[0,0], go1_series[0,1], c='#FF0000', marker=m)
            ## For plotting the full robot path as it moves
            scat4 = sub1.scatter(go1_series[0,0], go1_series[0,1], c='#FF0000', marker=".", label='Go1')

            go1_stream = stream

    else:
        raise RuntimeError('Unknown stream format')

## Determine if robot is moving
spot_moving = np.zeros(spot_series.shape[0])
go1_moving = np.zeros(go1_series.shape[0])

for i in range(spot_series.shape[0]-20):
    
    _yaw1 = math.atan2(2.0 * (spot_series[i+20,5] * spot_series[i+20,4]), spot_series[i+20,5] * spot_series[i+20,5] - spot_series[i+20,4] * spot_series[i+20,4])

    _yaw0 = math.atan2(2.0 * (spot_series[i,5] * spot_series[i,4]), spot_series[i,5] * spot_series[i,5] - spot_series[i,4] * spot_series[i,4])

    ## if position is changing
    if np.sqrt((spot_series[i+20,0] - spot_series[i,0])**2) + np.sqrt((spot_series[i+20,1] - spot_series[i,1])**2) > 0.075:
        spot_moving[i+20] = 1
    ## if yaw is changing
    elif abs(_yaw1 - _yaw0) > 0.1:
        spot_moving[i+20] = 1

for i in range(go1_series.shape[0]-20):
    _yaw1 = math.atan2(2.0 * (go1_series[i+20,5] * go1_series[i+20,4]), go1_series[i+20,5] * go1_series[i+20,5] - go1_series[i+20,4] * go1_series[i+20,4])

    _yaw0 = math.atan2(2.0 * (go1_series[i,5] * go1_series[i,4]), go1_series[i,5] * go1_series[i,5] - go1_series[i,4] * go1_series[i,4])

    ## if position is changing
    if np.sqrt((go1_series[i+20,0] - go1_series[i,0])**2) + np.sqrt((go1_series[i+20,1] - go1_series[i,1])**2) > 0.075:
        go1_moving[i+20] = 1
    ## if yaw is changing
    elif abs(_yaw1 - _yaw0) > 0.1:
        go1_moving[i+20] = 1

## Determine if robot is in line of sight



line1 = sub2.plot(spot_stream['time_stamps'], spot_moving[:])[0]
line2 = sub3.plot(go1_stream['time_stamps'], go1_moving[:])[0]
## Plot animation
ani = animation.FuncAnimation(fig=fig, func=update, interval=1) ## interval=1 speeds up the animation
plt.show()
