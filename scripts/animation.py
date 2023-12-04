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
import getch

sys.path.append('data')
sys.path.append('map')


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
 
class line:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
 
def onLine(l1, p):
    # Check whether p is on the line or not
    if (
        p.x <= max(l1.p1.x, l1.p2.x)
        and p.x >= min(l1.p1.x, l1.p2.x)
        and (p.y <= max(l1.p1.y, l1.p2.y) and p.y >= min(l1.p1.y, l1.p2.y))
    ):
        return True
    return False
 
def direction(a, b, c):
    val = (b.y - a.y) * (c.x - b.x) - (b.x - a.x) * (c.y - b.y)
    if val == 0:
        # Collinear
        return 0
    elif val < 0:
        # Anti-clockwise direction
        return 2
    # Clockwise direction
    return 1
 
def isIntersect(l1, l2):
    # Four direction for two lines and points of other line
    dir1 = direction(l1.p1, l1.p2, l2.p1)
    dir2 = direction(l1.p1, l1.p2, l2.p2)
    dir3 = direction(l2.p1, l2.p2, l1.p1)
    dir4 = direction(l2.p1, l2.p2, l1.p2)
 
    # When intersecting
    if dir1 != dir2 and dir3 != dir4:
        return True
 
    # When p2 of line2 are on the line1
    if dir1 == 0 and onLine(l1, l2.p1):
        return True
 
    # When p1 of line2 are on the line1
    if dir2 == 0 and onLine(l1, l2.p2):
        return True
 
    # When p2 of line1 are on the line2
    if dir3 == 0 and onLine(l2, l1.p1):
        return True
 
    # When p1 of line1 are on the line2
    if dir4 == 0 and onLine(l2, l1.p2):
        return True
 
    return False
 
def checkInside(poly, n, p):
    # When polygon has less than 3 edge, it is not polygon
    if n < 3:
        return False
 
    # Create a point at infinity, y is same as point p
    exline = line(p, Point(9999, p.y))
    count = 0
    i = 0
    while True:
        # Forming a line from two consecutive points of poly
        side = line(poly[i], poly[(i + 1) % n])
        if isIntersect(side, exline):
            # If side is intersects ex
            if (direction(side.p1, p, side.p2) == 0):
                return onLine(side, p)
            count += 1
         
        i = (i + 1) % n
        if i == 0:
            break
 
    # When count is odd
    return count & 1


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
    
    
    # update the ecgd4 plot:
    line1.set_xdata(ecgd4_stream['time_stamps'][:int(frame*ecg_srate/state_srate)])
    line1.set_ydata(ecgd4_series[:int(frame*ecg_srate/state_srate)])
    # print("ecgd4_stream['time_stamps'][frame] = ",ecgd4_stream['time_stamps'][frame])
    # print("ecgd4_series[frame] = ",ecgd4_series[frame][0])
    data = np.stack([ecgd4_stream['time_stamps'][int(frame*ecg_srate/state_srate)], ecgd4_series[int(frame*ecg_srate/state_srate)][0]]).T
    scat5.set_offsets(data)
    sub2.set_xlim(ecgd4_stream['time_stamps'][0], ecgd4_stream['time_stamps'][0] + frame / 14.5)

    # update the ecgfa plot:
    line2.set_xdata(ecgfa_stream['time_stamps'][:int(frame*ecg_srate/state_srate)])
    line2.set_ydata(ecgfa_series[:int(frame*ecg_srate/state_srate)])
    data = np.stack([ecgfa_stream['time_stamps'][int(frame*ecg_srate/state_srate)], ecgfa_series[int(frame*ecg_srate/state_srate)][0]]).T
    scat6.set_offsets(data)
    sub3.set_xlim(ecgfa_stream['time_stamps'][0], ecgfa_stream['time_stamps'][0] + frame / 14.5)

    # update the scgxd4 plot:
    line3.set_xdata(scgxd4_stream['time_stamps'][:int(frame*scg_srate/state_srate)])
    line3.set_ydata(scgxd4_series[:int(frame*scg_srate/state_srate)])
    data = np.stack([scgxd4_stream['time_stamps'][int(frame*scg_srate/state_srate)], scgxd4_series[int(frame*scg_srate/state_srate)][0]]).T
    scat7.set_offsets(data)
    sub4.set_xlim(scgxd4_stream['time_stamps'][0], scgxd4_stream['time_stamps'][0] + frame / 14.5)

    # update the scgyd4 plot:
    line4.set_xdata(scgyd4_stream['time_stamps'][:int(frame*scg_srate/state_srate)])
    line4.set_ydata(scgyd4_series[:int(frame*scg_srate/state_srate)])
    data = np.stack([scgyd4_stream['time_stamps'][int(frame*scg_srate/state_srate)], scgyd4_series[int(frame*scg_srate/state_srate)][0]]).T
    scat8.set_offsets(data)
    sub4.set_xlim(scgyd4_stream['time_stamps'][0], scgyd4_stream['time_stamps'][0] + frame / 14.5)

    # update the scgzd4 plot:
    line5.set_xdata(scgzd4_stream['time_stamps'][:int(frame*scg_srate/state_srate)])
    line5.set_ydata(scgzd4_series[:int(frame*scg_srate/state_srate)])
    data = np.stack([scgzd4_stream['time_stamps'][int(frame*scg_srate/state_srate)], scgzd4_series[int(frame*scg_srate/state_srate)][0]]).T
    scat9.set_offsets(data)
    sub4.set_xlim(scgzd4_stream['time_stamps'][0], scgzd4_stream['time_stamps'][0] + frame / 14.5)

    # update the scgxfa plot:
    line6.set_xdata(scgxfa_stream['time_stamps'][:int(frame*scg_srate/state_srate)])
    line6.set_ydata(scgxfa_series[:int(frame*scg_srate/state_srate)])
    data = np.stack([scgxfa_stream['time_stamps'][int(frame*scg_srate/state_srate)], scgxfa_series[int(frame*scg_srate/state_srate)][0]]).T
    scat10.set_offsets(data)
    sub5.set_xlim(scgxfa_stream['time_stamps'][0], scgxfa_stream['time_stamps'][0] + frame / 14.5)

    # update the scgyd4 plot:
    line7.set_xdata(scgyfa_stream['time_stamps'][:int(frame*scg_srate/state_srate)])
    line7.set_ydata(scgyfa_series[:int(frame*scg_srate/state_srate)])
    data = np.stack([scgyfa_stream['time_stamps'][int(frame*scg_srate/state_srate)], scgyfa_series[int(frame*scg_srate/state_srate)][0]]).T
    scat11.set_offsets(data)
    sub5.set_xlim(scgyfa_stream['time_stamps'][0], scgyfa_stream['time_stamps'][0] + frame / 14.5)

    # update the scgzd4 plot:
    line8.set_xdata(scgzfa_stream['time_stamps'][:int(frame*scg_srate/state_srate)])
    line8.set_ydata(scgzfa_series[:int(frame*scg_srate/state_srate)])
    data = np.stack([scgzfa_stream['time_stamps'][int(frame*scg_srate/state_srate)], scgzfa_series[int(frame*scg_srate/state_srate)][0]]).T
    scat12.set_offsets(data)
    sub5.set_xlim(scgzfa_stream['time_stamps'][0], scgzfa_stream['time_stamps'][0] + frame / 14.5)

    # update the audio signal smallroom
    data = np.stack([9.45, 1.65]).T
    scat14.set_offsets(data)
    t = Affine2D().scale(250*smallroom_series[int(frame*smallroom_srate/state_srate)])
    m = MarkerStyle(TextPath((-int((250*smallroom_series[int(frame*smallroom_srate/state_srate)])/2), -int((250*smallroom_series[int(frame*smallroom_srate/state_srate)])/2)), "o"), transform=t)
    scat14.set_paths([MarkerStyle(m).get_path().transformed(MarkerStyle(m).get_transform())])

    # update the audio signal livroom
    data = np.stack([4.75, 3.15]).T
    scat13.set_offsets(data)
    t = Affine2D().scale(425*livroom_series[int(frame*livroom_srate/state_srate)])
    m = MarkerStyle(TextPath((-int((425*livroom_series[int(frame*livroom_srate/state_srate)])/2), -int((425*livroom_series[int(frame*livroom_srate/state_srate)])/2)), "o"), transform=t)
    scat13.set_paths([MarkerStyle(m).get_path().transformed(MarkerStyle(m).get_transform())])
    
    if spot_subj1_los[frame]:
        t = Affine2D().scale(4)
        m = MarkerStyle(TextPath((0, 0), "!"), transform=t)
    else:
        t = Affine2D().scale(0)
        m = MarkerStyle(TextPath((0, 0), "!"), transform=t)
    scat15.set_paths([MarkerStyle(m).get_path().transformed(MarkerStyle(m).get_transform())])
    
    if go1_subj1_los[frame]:
        t = Affine2D().scale(4)
        m = MarkerStyle(TextPath((0, 0), "!"), transform=t)
    else:
        t = Affine2D().scale(0)
        m = MarkerStyle(TextPath((0, 0), "!"), transform=t)
    scat17.set_paths([MarkerStyle(m).get_path().transformed(MarkerStyle(m).get_transform())])
    
    if 'social' in xdf_file:
        if spot_subj2_los[frame]:
            t = Affine2D().scale(4)
            m = MarkerStyle(TextPath((0, 0), "!"), transform=t)
        else:
            t = Affine2D().scale(0)
            m = MarkerStyle(TextPath((0, 0), "!"), transform=t)
        scat16.set_paths([MarkerStyle(m).get_path().transformed(MarkerStyle(m).get_transform())])

        if go1_subj2_los[frame]:
            t = Affine2D().scale(4)
            m = MarkerStyle(TextPath((0, 0), "!"), transform=t)
        else:
            t = Affine2D().scale(0)
            m = MarkerStyle(TextPath((0, 0), "!"), transform=t)
        scat18.set_paths([MarkerStyle(m).get_path().transformed(MarkerStyle(m).get_transform())])
    else:
        if spot_subj2_los[frame]:
            t = Affine2D().scale(4).rotate_deg(-40)
            m = MarkerStyle(TextPath((0, 0), "!"), transform=t)
        else:
            t = Affine2D().scale(0).rotate_deg(-40)
            m = MarkerStyle(TextPath((0, 0), "!"), transform=t)
        scat16.set_paths([MarkerStyle(m).get_path().transformed(MarkerStyle(m).get_transform())])

        if go1_subj2_los[frame]:
            t = Affine2D().scale(4).rotate_deg(-40)
            m = MarkerStyle(TextPath((0, 0), "!"), transform=t)
        else:
            t = Affine2D().scale(0).rotate_deg(-40)
            m = MarkerStyle(TextPath((0, 0), "!"), transform=t)
        scat18.set_paths([MarkerStyle(m).get_path().transformed(MarkerStyle(m).get_transform())])


    return (scat1) # (line1, line2, line3, line4, line5, line6)

## Load the xdf file
# xdf_file = 'session1/session1-trial8-isolated-P0livroom-P1smallroom-together-search.xdf'
xdf_file = 'session1/session1-trial2-social-separated-search.xdf'
data, header = pyxdf.load_xdf('data/' + xdf_file)

## Create the figure
fig = plt.figure(figsize=(4,6))
fig.subplots_adjust(bottom=0.015, left=0.015, top = 0.985, right=0.985)

## Arrange subplots
sub1 = plt.subplot(2,1,1)
sub2 = plt.subplot(4,2,5)
sub2.set_xticks([])
sub2.set_yticks([])
sub2.set_title("Participant 1 ECG Signal")
sub3 = plt.subplot(4,2,6)
sub3.set_xticks([])
sub3.set_yticks([])
sub3.set_title("Participant 2 ECG Signal")
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
    sub1.scatter(5.75, 3.2, c='#008000', marker=m, label="Human Participant")
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
            spot_stream = stream
        
            _yaw = math.atan2(2.0 * (spot_series[0,5] * spot_series[0,4]), spot_series[0,5] * spot_series[0,5] - spot_series[0,4] * spot_series[0,4])
            # print("yaw = ", _yaw)
            # print("np.rad2deg(_yaw) = ",np.rad2deg(_yaw))
            t = Affine2D().scale(7).rotate_deg(np.rad2deg(_yaw))
            m = MarkerStyle(TextPath((0, 0), ">"), transform=t)
            ## For plotting the oriented robot position
            scat1 = sub1.scatter(spot_series[0,0], spot_series[0,1], c='#0000FF', marker=m)
            ## For plotting the full robot path as it moves
            scat2 = sub1.scatter(spot_series[0,0], spot_series[0,1], c='#0000FF', marker=".", label='Spot')
            


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
            go1_stream = stream

            _yaw = math.atan2(2.0 * (go1_series[0,5] * go1_series[0,4]), go1_series[0,5] * go1_series[0,5] - go1_series[0,4] * go1_series[0,4])
            # print("yaw = ", _yaw)
            # print("np.rad2deg(_yaw) = ",np.rad2deg(_yaw))
            t = Affine2D().scale(7).rotate_deg(np.rad2deg(_yaw))
            m = MarkerStyle(TextPath((0, 0), ">"), transform=t)
            ## For plotting the oriented robot position
            scat3 = sub1.scatter(go1_series[0,0], go1_series[0,1], c='#FF0000', marker=m)
            ## For plotting the full robot path as it moves
            scat4 = sub1.scatter(go1_series[0,0], go1_series[0,1], c='#FF0000', marker=".", label='Go1')

        
        if ("ECG-D4" in str(stream['info']['name'])) and (y.shape[0] > 1):
            ## Stream name
            print("stream name: ", stream['info']['name'])
            ## ex: spot/go1_state type = xy_rxryrzrw
            print("stream type: ", stream['info']['type'])
            ## Stream dtype
            print("stream channel format: ", stream['info']['channel_format'])
            
            ecgd4_series = stream['time_series']
            print("ecgd4_series.shape = ",ecgd4_series.shape)
            ## effective rate
            print("Actual sample rate: ",stream['info']['effective_srate'])
            ecg_srate = float(stream['info']['effective_srate'])
            print('-----------------------')

            line1 = sub2.plot(stream['time_stamps'], ecgd4_series[:], color="green")[0]
            scat5 = sub2.scatter(stream['time_stamps'][0], ecgd4_series[0], c='orange')
            ecgd4_stream = stream

        if ("SCGX-D4" in str(stream['info']['name'])) and (y.shape[0] > 1):
            ## Stream name
            print("stream name: ", stream['info']['name'])
            ## ex: spot/go1_state type = xy_rxryrzrw
            print("stream type: ", stream['info']['type'])
            ## Stream dtype
            print("stream channel format: ", stream['info']['channel_format'])
            scgxd4_series = stream['time_series']
            print("scgxd4_series.shape = ",scgxd4_series.shape)
            ## effective rate
            print("Actual sample rate: ",stream['info']['effective_srate'])
            scg_srate = float(stream['info']['effective_srate'])
            print('-----------------------')

            line3 = sub4.plot(stream['time_stamps'], scgxd4_series[:])[0]
            scat7 = sub4.scatter(stream['time_stamps'][0], scgxd4_series[0], c='orange')
            scgxd4_stream = stream
            sub4.legend(labelcolor='linecolor')

        if ("SCGY-D4" in str(stream['info']['name'])) and (y.shape[0] > 1):
            ## Stream name
            print("stream name: ", stream['info']['name'])
            ## ex: spot/go1_state type = xy_rxryrzrw
            print("stream type: ", stream['info']['type'])
            ## Stream dtype
            print("stream channel format: ", stream['info']['channel_format'])
            scgyd4_series = stream['time_series']
            print("scgyd4_series.shape = ",scgyd4_series.shape)
            ## effective rate
            print("Actual sample rate: ",stream['info']['effective_srate'])
            print('-----------------------')

            line4 = sub4.plot(stream['time_stamps'], scgyd4_series[:])[0]
            scat8 = sub4.scatter(stream['time_stamps'][0], scgyd4_series[0], c='orange')
            scgyd4_stream = stream

        if ("SCGZ-D4" in str(stream['info']['name'])) and (y.shape[0] > 1):
            ## Stream name
            print("stream name: ", stream['info']['name'])
            ## ex: spot/go1_state type = xy_rxryrzrw
            print("stream type: ", stream['info']['type'])
            ## Stream dtype
            print("stream channel format: ", stream['info']['channel_format'])
            scgzd4_series = stream['time_series']
            print("scgzd4_series.shape = ",scgzd4_series.shape)
            ## effective rate
            print("Actual sample rate: ",stream['info']['effective_srate'])
            print('-----------------------')

            line5 = sub4.plot(stream['time_stamps'], scgzd4_series[:])[0]
            scat9 = sub4.scatter(stream['time_stamps'][0], scgzd4_series[0], c='orange')
            scgzd4_stream = stream


        if ("ECG-FA" in str(stream['info']['name'])) and (y.shape[0] > 1):
            ## Stream name
            print("stream name: ", stream['info']['name'])
            ## ex: spot/go1_state type = xy_rxryrzrw
            print("stream type: ", stream['info']['type'])
            ## Stream dtype
            print("stream channel format: ", stream['info']['channel_format'])
            ecgfa_series = stream['time_series']
            print("ecgfa_series.shape = ",ecgfa_series.shape)
            ## effective rate
            print("Actual sample rate: ",stream['info']['effective_srate'])
            print('-----------------------')

            line2 = sub3.plot(stream['time_stamps'], ecgfa_series[:], color="green")[0]
            scat6 = sub3.scatter(stream['time_stamps'][0], ecgfa_series[0], c='orange')
            ecgfa_stream = stream
        
        if ("SCGX-FA" in str(stream['info']['name'])) and (y.shape[0] > 1):
            ## Stream name
            print("stream name: ", stream['info']['name'])
            ## ex: spot/go1_state type = xy_rxryrzrw
            print("stream type: ", stream['info']['type'])
            ## Stream dtype
            print("stream channel format: ", stream['info']['channel_format'])
            scgxfa_series = stream['time_series']
            print("scgxfa_series.shape = ",scgxfa_series.shape)
            ## effective rate
            print("Actual sample rate: ",stream['info']['effective_srate'])
            print('-----------------------')

            line6 = sub5.plot(stream['time_stamps'], scgxfa_series[:])[0]
            scat10 = sub5.scatter(stream['time_stamps'][0], scgxfa_series[0], c='orange')
            scgxfa_stream = stream
            sub5.legend(labelcolor='linecolor', loc="right")

        if ("SCGY-FA" in str(stream['info']['name'])) and (y.shape[0] > 1):
            ## Stream name
            print("stream name: ", stream['info']['name'])
            ## ex: spot/go1_state type = xy_rxryrzrw
            print("stream type: ", stream['info']['type'])
            ## Stream dtype
            print("stream channel format: ", stream['info']['channel_format'])
            scgyfa_series = stream['time_series']
            print("scgyfa_series.shape = ",scgyfa_series.shape)
            ## effective rate
            print("Actual sample rate: ",stream['info']['effective_srate'])
            print('-----------------------')

            line7 = sub5.plot(stream['time_stamps'], scgyfa_series[:])[0]
            scat11 = sub5.scatter(stream['time_stamps'][0], scgyfa_series[0], c='orange')
            scgyfa_stream = stream

        if ("SCGZ-FA" in str(stream['info']['name'])) and (y.shape[0] > 1):
            ## Stream name
            print("stream name: ", stream['info']['name'])
            ## ex: spot/go1_state type = xy_rxryrzrw
            print("stream type: ", stream['info']['type'])
            ## Stream dtype
            print("stream channel format: ", stream['info']['channel_format'])
            scgzfa_series = stream['time_series']
            print("scgzfa_series.shape = ",scgzfa_series.shape)
            ## effective rate
            print("Actual sample rate: ",stream['info']['effective_srate'])
            print('-----------------------')

            line8 = sub5.plot(stream['time_stamps'], scgzfa_series[:])[0]
            scat12 = sub5.scatter(stream['time_stamps'][0], scgzfa_series[0], c='orange')
            scgzfa_stream = stream

        if "mic-livroom" in str(stream['info']['name']):
            ## Stream name
            print("stream name: ", stream['info']['name'])
            ## ex: spot/go1_state type = xy_rxryrzrw
            print("stream type: ", stream['info']['type'])
            ## Stream dtype
            print("stream channel format: ", stream['info']['channel_format'])
            ## effective rate
            print("Actual sample rate: ",stream['info']['effective_srate'])
            livroom_srate = float(stream['info']['effective_srate'])

            ## The actual data points
            livroom_series = stream['time_series']
            print('livroom_series.shape = ', livroom_series.shape)
            print('-----------------------')

            t = Affine2D().scale(livroom_series[0])
            m = MarkerStyle(TextPath((0, 0), "."), transform=t)
            ## For plotting the audio signal strength
            scat13 = sub1.scatter(4.78, 3.23, marker=m, label='Microphone Living Room')

        if "mic-smallroom" in str(stream['info']['name']):
            ## Stream name
            print("stream name: ", stream['info']['name'])
            ## ex: spot/go1_state type = xy_rxryrzrw
            print("stream type: ", stream['info']['type'])
            ## Stream dtype
            print("stream channel format: ", stream['info']['channel_format'])
            ## effective rate
            print("Actual sample rate: ",stream['info']['effective_srate'])
            smallroom_srate = float(stream['info']['effective_srate'])

            ## The actual data points
            smallroom_series = stream['time_series']
            print('livroom_series.shape = ', smallroom_series.shape)
            print('-----------------------')

            t = Affine2D().scale(smallroom_series[0])
            m = MarkerStyle(TextPath((0, 0), "."), transform=t)
            ## For plotting the audio signal strength
            scat14 = sub1.scatter(9.7, 1.77, marker=m, label='Microphone Small Room')
            


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
spot_subj1_los = np.zeros(spot_series.shape[0])
spot_subj2_los = np.zeros(spot_series.shape[0])
go1_subj1_los = np.zeros(go1_series.shape[0])
go1_subj2_los = np.zeros(go1_series.shape[0])

# Points of the polygon
polygon_livroom1 = [ Point(1,7), Point(1,2), Point(5,2), Point(5,7)]
polygon_livroom2 = [ Point(1,4), Point(1,7), Point(14,7), Point(14,4)]
polygon_smallroom = [  Point(10,1), Point(12,0), Point(14,0), Point(14,7)]

if 'social' in xdf_file:
    for i in range(spot_series.shape[0]):
        p = Point(int(spot_series[i,0]),int(spot_series[i,1]))
        n = 4
        if checkInside(polygon_livroom1, n, p):
            spot_subj1_los[i] = 1
            spot_subj2_los[i] = 1
        elif checkInside(polygon_livroom2, n, p):
            spot_subj1_los[i] = 1
            spot_subj2_los[i] = 1
    for i in range(go1_series.shape[0]):
        p = Point(int(go1_series[i,0]),int(go1_series[i,1]))
        n = 4
        if checkInside(polygon_livroom1, n, p):
            go1_subj1_los[i] = 1
            go1_subj2_los[i] = 1
        elif checkInside(polygon_livroom2, n, p):
            go1_subj1_los[i] = 1
            go1_subj2_los[i] = 1
else:
    for i in range(spot_series.shape[0]):
        p = Point(int(spot_series[i,0]),int(spot_series[i,1]))
        n = 4
        if checkInside(polygon_livroom1, n, p):
            spot_subj1_los[i] = 1
        elif checkInside(polygon_livroom2, n, p):
            spot_subj1_los[i] = 1
        n = 4
        if(checkInside(polygon_smallroom, n, p)):
            spot_subj2_los[i] = 1
    for i in range(go1_series.shape[0]):
        p = Point(int(go1_series[i,0]),int(go1_series[i,1]))
        n = 4
        if checkInside(polygon_livroom1, n, p):
            go1_subj1_los[i] = 1
        elif checkInside(polygon_livroom2, n, p):
            go1_subj1_los[i] = 1
        n = 4
        if(checkInside(polygon_smallroom, n, p)):
            go1_subj2_los[i] = 1


## Scatter plots for the Spot Line of Sight Booleans
t = Affine2D().scale(4)
m = MarkerStyle(TextPath((0, 0), "!"), transform=t)
scat15 = sub1.scatter(subj1_pos[0]-0.3, subj1_pos[1], c='#0000FF', marker=m, label='Spot Line of Sight of Participant')
if 'social' in xdf_file:
    t = Affine2D().scale(4)
    m = MarkerStyle(TextPath((0, 0), "!"), transform=t)
    scat16 = sub1.scatter(subj2_pos[0]-0.3, subj2_pos[1], c='#0000FF', marker=m)
else:
    t = Affine2D().scale(4).rotate_deg(-40)
    m = MarkerStyle(TextPath((0, 0), "!"), transform=t)
    scat16 = sub1.scatter(subj2_pos[0]-0.2, subj2_pos[1]+0.25, c='#0000FF', marker=m)
## Scatter plots for the Go1 Line of Sight Booleans
t = Affine2D().scale(4)
m = MarkerStyle(TextPath((0, 0), "!"), transform=t)
scat17 = sub1.scatter(subj1_pos[0]+0.1, subj1_pos[1], c='#FF0000', marker=m, label='Go1 Line of Sight of Participant')
if 'social' in xdf_file:
    t = Affine2D().scale(4)
    m = MarkerStyle(TextPath((0, 0), "!"), transform=t)
    scat18 = sub1.scatter(subj2_pos[0]+0.1, subj2_pos[1], c='#FF0000', marker=m)
else:
    t = Affine2D().scale(4).rotate_deg(-40)
    m = MarkerStyle(TextPath((0, 0), "!"), transform=t)
    scat18 = sub1.scatter(subj2_pos[0]+0.1, subj2_pos[1]-0.15, c='#FF0000', marker=m)
## Finally add the legend for top plot
sub1.legend(labelcolor='linecolor', loc='upper left')

input("Press Enter to continue...")

## Run the animation
ani = animation.FuncAnimation(fig=fig, func=update, interval=1) ## interval=1 speeds up the animation
plt.show()


