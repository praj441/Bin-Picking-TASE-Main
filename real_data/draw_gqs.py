import cv2
import numpy as np
import random
import copy
import open3d as o3d

w = 640 #320
h = 480 #240

def draw_graspability_labels(gqs_arr,image):
	print(I.shape, gqs_arr.shape)
	# input()
	image1 = copy.deepcopy(image)
	for i in range(w):
		for j in range(h):
			if gqs_arr[j,i] > 0:
				csize = int(gqs_arr[j,i]/20) + 1
				cv2.circle(image, (i, j), csize, (255,0,0), -1)
	# for i in range(w):
	# 	for j in range(h):			
	# 		if gqs_arr[j,i] >= 0 and random.random() > 0.5:
	# 			cv2.circle(image1, (i, j), 1, (0,0,255), -1)
	return image, image1



scenes = 50
data_path = 'test_data_level_1'

for scene in range()
grasp_quality_score_arr = np.loadtxt(data_path+'/{0:06d}_gqs_array.txt'.format(scene))
I = cv2.imread(data_path+'/{0:06d}_ref_image.png'.format(scene))
darray = np.loadtxt(data_path+'/{0:06d}_depth_array.txt'.format(scene)).astype(np.float32)
gqs_draw, gqs_samples = draw_graspability_labels(grasp_quality_score_arr,I)
cv2.imwrite(data_path+'/{0:06d}_gqs_draw.png'.format(scene),gqs_draw)
cv2.imwrite(data_path+'/{0:06d}_gqs_samples.png'.format(scene),gqs_samples)

# md = np.loadtxt(data_path+'/median_depth_map.txt').astype(np.float32)
# pcd = o3d.geometry.create_point_cloud_from_depth_image(o3d.geometry.Image(darray),o3d.camera.PinholeCameraIntrinsic(640,480,614.72,614.14,320.0,240.0),depth_scale=1.0)
# o3d.visualization.draw_geometries([pcd])