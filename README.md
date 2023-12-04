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

### 1. animation.py
The completed animation. Plots the robots moving at the top with synchronized ECG and SCG data on the plots below. Also on the robot state plot is the participant location, indicators for whether the robots are in line of sight of a participants and cirlces with magnitude representing microphone stream.
``` 
cd xdf-visualization
python scripts/animation.py
```

## Test Files

### 1. test_simple_animation
This script animates a single stream of data from the xdf with 6 time series in it. 
In this case the example is the spot_state from a test xdf file.
This is how we want to plot the biosignal data plots.
``` 
cd xdf-visualization
python test/test_simple_animation.py
```

### 2. test_robotstate_animation
This script animates the same robot state, but does so as position and otientation of the robot over the map.
This is how we want to visualize the robot state, however we need both of them.
``` 
cd xdf-visualization
python test/test_robotstate_animation.py
```

### 3. test_robotstate_animation
This script animates the same robot state, but does so as position and otientation of the robot over the map.
This is how we want to visualize the robot state, however we need both of them.
``` 
cd xdf-visualization
python test/test_robotstate_animation.py
```

### 4. test_createstreams
This script is the first attempt to generate stream data and visualizes it with the robots moving 
Namely: 1) robots line of sight; 2) robots moving;;;;;; 3) robots nearby (not currently included)
``` 
cd xdf-visualization
python test/test_createstreams.py
```




## Random Helpful
Plotting Multivariate data:
https://matplotlib.org/stable/gallery/lines_bars_and_markers/multivariate_marker_plot.html