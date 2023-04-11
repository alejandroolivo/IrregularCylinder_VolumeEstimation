import open3d as o3d
from scipy.spatial.transform import Rotation as R
import numpy as np
from ellipse import LsqEllipse
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# origin 
origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=80, origin=[0,0,0])

# Load two mesh from ply file
# pcd = o3d.io.read_point_cloud(r'G:\bcnvision\FACCSA\3D Jamones\Default\PCs\3\pc0_rot.ply')
# pcd2 = o3d.io.read_point_cloud(r'G:\bcnvision\FACCSA\3D Jamones\Default\PCs\3\pc1_rot.ply')

# Load a point cloud from a txt file. The txt file has 3 columns: x, y, z
pcd3 = o3d.io.read_point_cloud(r'F:\PointClouds\pts.xyz', format='xyz')
pcd4 = o3d.io.read_point_cloud(r'F:\PointClouds\pts.xyz', format='xyz')

# remove points between X=0 and X=300
points = np.asarray(pcd3.points)
points = points[points[:,0] < 100]
pcd3.points = o3d.utility.Vector3dVector(points)

# remove points between X=0 and X=300
points = np.asarray(pcd3.points)
points = points[points[:,0] > 100]
pcd4.points = o3d.utility.Vector3dVector(points)

# Load a point cloud from a txt file. The txt file has 3 columns: x, y, z
pcd5 = o3d.io.read_point_cloud(r'.\PointClouds\ellipse_points.xyz', format='xyz')

# calculate best fit ellipse
# ellipse = pcd3.compute_ellipsoid()
# pcd4 = ellipse.sample_points_uniformly(number_of_points=1000)

# cortar jamones
# points = np.asarray(pcd.points)
# points = points[points[:,1] < -58]
# points = points[points[:,1] > -80]
# points = points[points[:,0] > 300]
# pcd.points = o3d.utility.Vector3dVector(points)

# points2 = np.asarray(pcd2.points)
# points2 = points2[points2[:,1] < -58]
# points2 = points2[points2[:,1] > -80]
# points2 = points2[points2[:,0] > 300]
# pcd2.points = o3d.utility.Vector3dVector(points2)

# pcd.paint_uniform_color([1, 0.706, 0])
# pcd2.paint_uniform_color([1, 0.0, 0.0])
pcd3.paint_uniform_color([0, 0.2, 1])
pcd4.paint_uniform_color([0.2, 0.9, 0.7])
pcd5.paint_uniform_color([0.6, 0.4, 0.7])

# draw
# o3d.visualization.draw_geometries([pcd, pcd2, pcd3, pcd4, pcd5, origin])
o3d.visualization.draw_geometries([pcd3, pcd4, pcd5, origin])

# exportar matrix
