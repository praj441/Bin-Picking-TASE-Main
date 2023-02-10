import open3d as o3d
import numpy as np

md = np.loadtxt('median_depth_map.txt').astype(np.float32)
path='real_data/noise_samples'
for i in range(10):
	md = np.loadtxt(path+'/{0:06d}.txt'.format(i)).astype(np.float32)
	pcd = o3d.geometry.create_point_cloud_from_depth_image(o3d.geometry.Image(md),o3d.camera.PinholeCameraIntrinsic(640,480,614.72,614.14,320.0,240.0),depth_scale=1.0)
	o3d.visualization.draw_geometries([pcd])