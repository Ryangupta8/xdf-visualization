# xdf-visualization

## Build Instructions

### Conda
Assuming that conda is installed:
```
conda create --name xdf python=3.11
conda activate xdf
pip install -r requirements.txt
```

## Scripts

### 1. test_simple_animation
This script animates a single stream of data from the xdf with 6 time series in it. 
In this case the example is the spot_state from a test xdf file.
This is how we want to plot the biosignal data plots.
``` 
cd xdf-visualization
python test_simple_animation.py
```


### 2. test_robotstate_animation
This script animates the same robot state, but does so as position and otientation of the robot over the map.
This is how we want to visualize the robot state, however we need both of them.
``` 
cd xdf-visualization
python test_robotstate_animation.py
```

### 2. test_createstreams
This script is the first attempt to generate stream data for HS
Namely: 1) robots line of sight; 2) robots moving; 3) robots nearby 
``` 
cd xdf-visualization
python test_createstreams.py
```



## Random Helpful
Plotting the data with orientation:
https://matplotlib.org/stable/gallery/lines_bars_and_markers/multivariate_marker_plot.html