# IrregularCilinderVolumeEstimation
Python code to estimate the volume of a truncated-cylindrical object by fitting profiles to ellipses.

This tool is used for estimating the volume of objects represented as point clouds. The tool performs a principal component analysis (PCA) on the point cloud to align all profiles, splits the point cloud into 2D profiles, fits an ellipse to each profile, and then calculates the volume of the object from the fitted ellipses.

# Requirements
ellipse==0.6.0
lsq_ellipse==2.2.1
matplotlib==3.6.3
numpy==1.23.3
open3d==0.16.0
scikit_learn==1.2.1
scipy==1.10.0

# Usage
The input data should be a point cloud file in XYZ format with columns for X, Y, and Z coordinates. To use the tool, run the "VolumeEstimationTool.py" script and provide the path to the input file. The tool will then generate a series of plots to visualize the point cloud, the aligned point cloud, and the fitted ellipses.

# Parameters
The tool has several parameters that can be adjusted:

DRAW_PROFILES: If True, the tool will plot each 2D profile in a separate subplot.

PERFORM_PCA: If True, the tool will perform PCA to align all profiles.

MIN_ASPECT_RATIO: The minimum aspect ratio of the fitted ellipse.

MAX_ASPECT_RATIO: The maximum aspect ratio of the fitted ellipse.


These parameters can be adjusted by changing their values in the "VolumeEstimationTool.py" script.
