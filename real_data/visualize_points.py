import numpy as np 
import open3d as o3d 
import cv2

scene = 5
# preds = np.loadtxt('preds.txt')
root = 'test_data_mix'
while 1:
	

	pc_file = root + '/{0:06d}_pc_complete.npy'.format(scene)
	pc_complete = np.load(pc_file)
	image = cv2.imread(root + '/{0:06d}_ref_image.png'.format(scene))

	pc_complete= pc_complete.astype(np.float32).reshape(-1,3)
	pc_color = image.astype(np.float32).reshape(-1,3)
	pc_color_complete = pc_color/255


	d = np.sum(np.abs(pc_complete)**2,axis=-1)**(1./2)
	pc_arr = pc_complete[np.where(d>0)]
	pc_color = pc_color_complete[np.where(d>0)]

	temp = pc_color[:,2].copy()
	pc_color[:,2] = pc_color[:,0].copy()
	pc_color[:,0] = temp

	# pc_color = 0*pc_color + np.array([0.5,0.5,0.5])
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(pc_arr)
	pcd.colors = o3d.utility.Vector3dVector(pc_color)
	o3d.visualization.draw_geometries([pcd])


	filter_mask = np.load(root + '/{0:06d}_filter_mask.npy'.format(scene)).reshape(-1).astype(bool)
	pc_color = pc_color_complete[filter_mask]
	pc_arr = pc_complete[filter_mask]

	# pc_arr = pc_filter.astype(np.float32)[:,0:3]


	d = np.sum(np.abs(pc_arr)**2,axis=-1)**(1./2)
	pc_arr = pc_arr[np.where(d>0)]
	pc_color = pc_color[np.where(d>0)]

	temp = pc_color[:,2].copy()
	pc_color[:,2] = pc_color[:,0].copy()
	pc_color[:,0] = temp

	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(pc_arr)
	pcd.colors = o3d.utility.Vector3dVector(pc_color)
	o3d.visualization.draw_geometries([pcd])