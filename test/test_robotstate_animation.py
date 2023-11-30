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
def plot_vectormap(fname,facecolor="white",linecolor="black"):
    with open(fname,"r") as f:
        lines = f.readlines()
    fig,ax = plt.subplots(facecolor=facecolor)
    ax.set_facecolor(facecolor)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    for line in lines:
        pts = [float(pt) for pt in [ln.split(", ") for ln in line.splitlines()][0]]
        x_,x = rot(np.array([[pts[0]],[pts[1]]])), rot(np.array([[pts[2]],[pts[3]]]))
        ln = ax.plot([x_[0],x[0]],[x_[1],x[1]],color=linecolor,zorder=0)
    fig.set_figwidth(20)
    fig.set_figheight(10)
    #fig.tight_layout()
    return fig,ax,ln


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
    return (scat1) # (line1, line2, line3, line4, line5, line6)

# xdf_file = 'session1-trial1-social-separated-navigation.xdf'
xdf_file = 'session1-trial8-isolated-P0livroom-P1smallroom-together-search.xdf'
## Load the xdf file
data, header = pyxdf.load_xdf('data/session1/' + xdf_file)

## Create the figure
# fig, ax = plt.subplots()
fig, ax, _ = plot_vectormap(vectormap_txt)

if 'social' in xdf_file:
    t = Affine2D().scale(4)
    m = MarkerStyle("^", transform=t)
    ax.scatter(5.75, 3, c='#008000', marker=m)
    ax.scatter(6.75, 3, c='#008000', marker=m)
else:
    t = Affine2D().scale(4)
    m = MarkerStyle("^", transform=t)
    ax.scatter(5.75, 3.2, c='#008000', marker=m)
    t = Affine2D().scale(4).rotate_deg(-40)
    m = MarkerStyle("^", transform=t)
    ax.scatter(10.0, 1.2, c='#008000', marker=m)


## loop thru all of the separate streams (e.g. go1_state; spot_state; etc.)
for stream in data:
    ## Stream name
    print("stream name: ", stream['info']['name'])
    ## ex: spot/go1_state type = xy_rxryrzrw
    print("stream type: ", stream['info']['type'])
    ## Stream dtype
    print("stream channel format: ", stream['info']['channel_format'])
    y = stream['time_series']
    print("stream['time_stamps'].shape = ", stream['time_stamps'].shape)
    print("y.shape = ",y.shape)
    print('-----------------------')

    ## Data in this example comes as np arrays
    if isinstance(y, np.ndarray):
        
        if "spot" in str(stream['info']['name']):
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
            scat1 = ax.scatter(y1[0,0], y1[0,1], c='#0000FF', marker=m)
            ## For plotting the full robot path as it moves
            scat2 = ax.scatter(y1[0,0], y1[0,1], c='#0000FF', marker=".")


        if "go1" in str(stream['info']['name']):
            ## The actual data points
            y2 = stream['time_series']

            _yaw = math.atan2(2.0 * (y2[0,5] * y2[0,4]), y2[0,5] * y2[0,5] - y2[0,4] * y2[0,4])
            # print("yaw = ", _yaw)
            # print("np.rad2deg(_yaw) = ",np.rad2deg(_yaw))
            t = Affine2D().scale(7).rotate_deg(np.rad2deg(_yaw))
            m = MarkerStyle(TextPath((0, 0), ">"), transform=t)
            ## For plotting the oriented robot position
            scat3 = ax.scatter(y2[0,0], y2[0,1], c='#FF0000', marker=m)
            ## For plotting the full robot path as it moves
            scat4 = ax.scatter(y2[0,0], y2[0,1], c='#FF0000', marker=".")

    else:
        raise RuntimeError('Unknown stream format')

ani = animation.FuncAnimation(fig=fig, func=update, interval=1) ## interval=1 speeds up the animation
plt.show()

