import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
import datetime

#Vars
DRAW_PROFILES = False
PERFORM_PCA = True
MIN_ASPECT_RATIO = 0.2
MAX_ASPECT_RATIO = 0.8

# save and print actual timestamp
datetimeStart = datetime.datetime.now()    
# print('Start Time = ' + datetimeStart.strftime("%Y-%m-%d %H:%M:%S.%f"))

# Define functions

def fit_ellipse(pointcloud, x_axis_length, isCentered=True, min_aspect_ratio=0.5, max_aspect_ratio=1):
    # Define the cost function to minimize
    def cost_function(params):
        x0, y0, a, b = params
        ellipse = ((pointcloud[:, 0] - x0) / a)**2 + ((pointcloud[:, 1] - y0) / b)**2
        return np.sum((ellipse - 1)**2)

    # Define the initial guess
    x0 = np.mean(pointcloud[:, 0])
    y0 = np.mean(pointcloud[:, 1])
    a = x_axis_length / 2
    b = np.sqrt(np.sum((pointcloud[:, 1] - y0)**2) / len(pointcloud))

    # Define the bounds for the optimization
    if (isCentered):
        bounds = [(-a, a), (-2*a, 2*a), (a-5, a+5), (min_aspect_ratio*a, max_aspect_ratio*a)]
    else:
        bounds = [(None, None), (None, None), (a-5, a+5), (min_aspect_ratio*a, max_aspect_ratio*a)]

    # Minimize the cost function
    result = minimize(cost_function, [x0, y0, a, b], bounds=bounds)

    # get a value of fitting error
    error = cost_function(result.x)

    # Return the parameters that define the best fitting ellipse
    x0, y0, a, b = result.x
    theta = 0.0
    return x0, y0, a, b, theta, error

def split_pointcloud_in_profiles(pointcloud, tol=2):
    # Sort the point cloud by x-coordinate
    pointcloud = pointcloud[pointcloud[:, 0].argsort()]

    # Split the point cloud into 2D profiles
    profiles = []
    current_profile = []
    mean_x = []
    prev_x = None
    for i in range(len(pointcloud)):
        x, y, z = pointcloud[i]
        if i==0:            
            mean_x.append(x) 
        if prev_x is None or abs(x - prev_x) < tol:
            # Add point to current profile
            current_profile.append([y, z])
        else:
            # Start new profile
            profiles.append(np.array(current_profile))
            current_profile = [[y, z]]
            # Add x-coordinate to mean_x
            mean_x.append(x)
        prev_x = x

    # Add last profile to list
    if current_profile:
        profiles.append(np.array(current_profile))

    return profiles, mean_x

# Load point cloud from file and truncate decimals
# pointcloud = np.loadtxt('pts_1perfil.xyz', delimiter=' ', usecols=(1,2), dtype=int)
pointcloud = np.loadtxt(r'F:\PointClouds\pts.xyz', delimiter=' ', usecols=(0,1,2),  dtype=int)

# use the function split_pointcloud to split the pointcloud in profiles
profiles, profile_x = split_pointcloud_in_profiles(pointcloud)

# plotting
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
ax1.set_aspect('equal')
ax2.set_aspect('equal')
if(DRAW_PROFILES):    
    fig2, axs = plt.subplots(nrows=len(profiles), ncols=2, figsize=(10,10))

pcas = []

# clean the .xyz file for ellipse points saving
f = open('.\PointClouds\ellipse_points.xyz', 'w')
f.close()

#do a for loop to do the code bellow for each profile
for i, profile in enumerate(profiles):

    # Perform PCA to align all profiles
    if(PERFORM_PCA):

        # Perform PCA   
        pca = PCA(n_components=2)
        pca.fit(profile)

        # Get eigenvalues and eigenvectors
        eigenvalues = pca.explained_variance_
        eigenvectors = pca.components_

        # Project point cloud onto principal components
        projected = pca.transform(profile)

        # save pca
        pcas.append(pca)

        # move profile to the origin    
        # projected = profile - np.mean(profile, axis=0)
        # projected = projected - np.mean(projected, axis=1)

    else:

        projected = profile

    # calculate the range of the data in the proyection plane
    x_min = projected[:, 0].min()
    x_max = projected[:, 0].max()
    y_min = projected[:, 1].min()
    y_max = projected[:, 1].max()

    # get 4 points of the roi
    roi = np.array([[x_min, y_min], [x_min, y_max], [x_max, y_max], [x_max, y_min]])

    # define a variable as the horizontal_spread of the roi
    horizontal_spread = x_max - x_min

    # define a variable as the vertical_spread of the roi
    vertical_spread = y_max - y_min

    # Fit ellipse to projected point cloud
    ellipse = fit_ellipse(projected, horizontal_spread, isCentered=PERFORM_PCA, min_aspect_ratio=MIN_ASPECT_RATIO, max_aspect_ratio=MAX_ASPECT_RATIO)

    # Generate x and y values for the ellipse
    center = (ellipse[0], ellipse[1])
    a = ellipse[2]
    b = ellipse[3]
    angle = 0
    t = np.linspace(0, 2*np.pi, 100)
    ellipse_points_untransformed = np.zeros((100, 2))
    x = center[0] + a*np.cos(t)*np.cos(angle) - b*np.sin(t)*np.sin(angle)
    ellipse_points_untransformed[:, 0] = x
    y = center[1] + a*np.cos(t)*np.sin(angle) + b*np.sin(t)*np.cos(angle)
    ellipse_points_untransformed[:, 1] = y

    # transform the ellipse points to the original space using pcas[i] inverted transformation with 100 points in the ellipse
    if(PERFORM_PCA):
        ellipse_points_untransformed = pcas[i].inverse_transform(ellipse_points_untransformed)

    # create a 3D pointCloud of the ellipse x=x_mean[i], y and z are the points of the ellipse
    ellipse_points = np.zeros((len(ellipse_points_untransformed), 3))
    ellipse_points[:, 0] = profile_x[i]
    ellipse_points[:, 1] = ellipse_points_untransformed[:, 0]
    ellipse_points[:, 2] = ellipse_points_untransformed[:, 1]
    
    # append ellipse points to the existing points in a .xyz file
    with open('.\PointClouds\ellipse_points.xyz', 'a') as f:
        np.savetxt(f, ellipse_points, fmt='%f')

    # Plot original point cloud
    ax1.scatter(profile[:, 0], profile[:, 1] + i*50, s=2)
    ax1.set_title('Original Point Cloud')

    # Plot projected point cloud
    ax2.scatter(projected[:, 0], projected[:, 1] + i*50, s=2)
    ax2.set_title('Projected Point Cloud')

    #add ellipse to the plot
    # Generate x and y values for the ellipse
    center = (ellipse[0], ellipse[1])
    a = ellipse[2]
    b = ellipse[3]
    angle = 0
    t = np.linspace(0, 2*np.pi, 100)
    x = center[0] + a*np.cos(t)*np.cos(angle) - b*np.sin(t)*np.sin(angle)
    y = center[1] + a*np.cos(t)*np.sin(angle) + b*np.sin(t)*np.cos(angle)
    error = ellipse[5]

    # Plot the ellipse with a color depending on the error  
    if error < 4:
        ax2.plot(x, y + i*50, color='g', linestyle='--', linewidth=1)
    elif error < 6:
        ax2.plot(x, y + i*50, color='y', linestyle='--', linewidth=1)
    else:   
        ax2.plot(x, y + i*50, color='r', linestyle='--', linewidth=1)

    if(DRAW_PROFILES):
        
        # Plot original point cloud
        axs[i, 0].scatter(profile[:, 0], profile[:, 1], color='b', s=1)
        axs[i, 0].set_title('Profile {}'.format(i+1))

        # Plot projected point cloud
        axs[i, 1].scatter(projected[:, 0], projected[:, 1], color='b', s=1)
        axs[i, 1].set_title('Projected Profile {}'.format(i+1))

        # Plot the ellipse
        axs[i, 1].plot(x, y, color='r', linestyle='--', linewidth=1)

        # Plot the x-axis of the ellipse
        axs[i, 1].plot([center[0]-a, center[0]+a], [center[1], center[1]], color='b', linestyle='-', linewidth=1)

        axs[i, 0].set_aspect('equal')
        axs[i, 1].set_aspect('equal')

        
# Print actual timestamp
datetimeEnd = datetime.datetime.now()
# print('End Time = ' + datetimeEnd.strftime("%Y-%m-%d %H:%M:%S.%f"))

# Print elapsed Time
print('Elapsed Time = ' + str(datetimeEnd - datetimeStart))

# Show plot
plt.show()