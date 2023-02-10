
import numpy as np
import cv2
import time
import sys, os
from math import *
sys.path.append('../commons')
from utils_gs import Parameters
from utils_gs import draw_rectified_rect
from grasp_evaluation import calculate_GDI2
import copy
import open3d as o3d

data_path = 'test_data_level_2'
out_path = 'test_data_level_2/baseline'
w = 640 #320
h = 480 #240
param = Parameters(w,h)
param.gripper_height = int(param.mh*70) 
idx = '000019'
k = 106
# index = 3

# idx = '000001'
# k = 1
# index = 0

dump_dir = out_path #+ '/dump/' + idx
darray = np.loadtxt(os.path.join(data_path, idx)+'_depth_array.txt').astype(np.float32)
pc_points = np.load(os.path.join(data_path, idx)+'_pc.npy')
rectangle_pixels = np.loadtxt(dump_dir+'/grasp_pose_info'+'/rectangle_{0}.txt'.format(k)).astype(np.int32)
angle = np.loadtxt(dump_dir+'/grasp_pose_info'+'/angle_{0}.txt'.format(k))
image = cv2.imread(os.path.join(data_path, idx)+'_ref_image.png')
# pc_file = '20/{0:03d}_pc.npy'.format(int(idx))
# pc_arr = np.load(pc_file)
# # pcd = o3d.geometry.PointCloud()
# # pcd.points = o3d.utility.Vector3dVector(pc_arr)
# # o3d.geometry.estimate_normals(pcd,search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,max_nn=30))
# pcd = o3d.geometry.create_point_cloud_from_depth_image(o3d.geometry.Image(darray),o3d.camera.PinholeCameraIntrinsic(320,240,307.36,307.07,160.0,120.0),depth_scale=1.0)
# o3d.visualization.draw_geometries([pcd])

inputs = {'darray':darray}
inputs['param'] = param
# inputs['pc_arr'] = pc_arr
inputs['slanted_pose_detection'] = False #True
inputs['cone_detection'] = True
st = time.time()
bmap,gdi,gdi_plus,gdi2, bmap_denoised ,cx,cy = calculate_GDI2(inputs,rectangle_pixels,angle)

cv2.imwrite(dump_dir+'/bmaps'+'/bmap{0}.jpg'.format(k),bmap)#.astype(np.uint8))
cv2.imwrite(dump_dir+'/bmaps'+'/bmap{0}_denoised.jpg'.format(k),bmap_denoised)#.astype(np.uint8))
draw_rectified_rect(image,rectangle_pixels)
cv2.imwrite(dump_dir+'/directions'+'/gpose{0}.jpg'.format(k),image)
print('time in gdi2',time.time()-st)
# zy, zx = np.gradient(gdi2.dmap)


# normal calculation
# st = time.time()
# pmap = gdi2.pmap
# gw,gh,_ = pmap.shape
# pc_points = pmap.reshape(gw*gh,3)
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(pc_points)
# o3d.geometry.estimate_normals(pcd,search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,max_nn=400))
# o3d.geometry.orient_normals_towards_camera_location(pcd)
# print('time in normals',time.time()-st)
# o3d.visualization.draw_geometries([pcd])
# nm = np.asarray(pcd.normals).reshape(gw,gh,3)
# centroid_normal = nm[:,int(gh/2)]
# print('normal vector',centroid_normal)
# cam_dir = np.array([0,0,-1])
# angles = np.zeros(centroid_normal.shape[0])
# for i in range(centroid_normal.shape[0]):
# 	angles[i] = acos(np.dot(centroid_normal[i],cam_dir))/3.14*180

# 	print('angle',angles[i])
