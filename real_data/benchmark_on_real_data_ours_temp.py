import numpy as np
import cv2
import time
import sys, os
from datetime import datetime
from math import *
from general_predictor import Predictor
import open3d as o3d
sys.path.append('../commons')
from utils_gs import Parameters, draw_grasp_map, draw_top_N_points, draw_grasp_pose_as_a_line, draw_rectified_rect
from grasp_evaluation import calculate_GDI2
from utils_cnn import draw_final_pose
import copy

data_path = 'results_ros_policy'
out_path = 'test_temp' #'test_data_level_1/ours'

# data_path = '../simulation/novel_objects'
# out_path = 'test_data_simulation/novel'

w = 320
h = 240
param = Parameters(w,h)
# self.raw_data_path = os.path.join(ROOT_DIR, 'sunrgbd/sunrgbd_trainval')
scan_names = sorted(list(set([os.path.basename(x)[0:6] \
	for x in os.listdir(data_path)])))

sampler = 'fcn_depth'
data_type = 'mid'
predictor = Predictor(sampler,data_type,Nt=10)

gqcnn_score_list = []
# scan_names = ['000023']
inputs_np = {}
inputs_np['slanted_pose_detection'] = False #False
inputs_np['cone_detection'] = False #False

def process_dmap_for_vis(dmap):
	dmap[dmap<0.4] = dmap.max()
	md = np.loadtxt('20/median_depth_map.txt')
	md = cv2.resize(md,(w,h))
	obj_region = ((md - dmap) > 0.01)
	dmap = np.where(obj_region,dmap-0.4,dmap)
	dv = (dmap/dmap.max()*255)

	obj_region = obj_region 

	dv = np.where(obj_region,np.power(dv,1.5),dv)
	omax = dv[obj_region].max()
	omin = dv[obj_region].min()
	dv = np.where(obj_region,((dv-omin)/omax*255),dv)
	return dv

def process_image_before_rendering(image):
	# image = cv2.GaussianBlur(image,filter_G,0)
	image = 0.8*image
	return image



scan_names = ['000006'] #,'000009']
avg_time = 0.0
scenes = 1
for i in range(64):
	# if i == 50:
	# 	break
	idx = str(i)
	print(idx)
	dump_dir = out_path #+ '/' + idx
	sample_path = os.path.join(data_path, idx)
	dmap = np.loadtxt(os.path.join(data_path, idx)+'_depth_array.txt')
	# dv = process_dmap_for_vis(copy.deepcopy(dmap))
	dv = None
	img = cv2.imread(os.path.join(data_path, idx)+'_ref_image.png')
	# pc_cloud = np.load(os.path.join(data_path, idx)+'_pc.npy')
	# filter_mask = np.load(os.path.join(data_path, idx)+'_filter_mask.npy')
	# pc_complete = np.load(os.path.join(data_path, idx)+'_pc_complete.npy')
	
	st = time.time()

	inputs_np['image'] = cv2.resize(img,(w,h))
	inputs_np['darray'] = cv2.resize(dmap,(w,h))

	# pcd = o3d.geometry.create_point_cloud_from_depth_image(o3d.geometry.Image(inputs_np['darray'].astype(np.float32)),o3d.camera.PinholeCameraIntrinsic(640,480,614.72,614.14,320.0,240.0),depth_scale=1.0)		
	# 	# o3d.visualization.draw_geometries([pcd])
	# pc_arr = np.asarray(pcd.points).reshape(h,w,3)


	inputs_np['depth_image'] = dv
	inputs_np['dump_dir'] = dump_dir
	inputs_np['pc_cloud'] = None #pc_cloud
	inputs_np['param'] = param
	# inputs_np['filter_mask'] = filter_mask
	# inputs_np['pc_complete'] = pc_complete
	inputs_np['pc_arr'] = None #pc_arr

	grasp_pose_info = predictor.predict(inputs_np)

	final_image, depth_image_copy, gqs_score = draw_final_pose(grasp_pose_info,copy.deepcopy(img),dmap,scale=2)
	# final_pose_rectangle = grasp_pose_info['final_pose_rectangle']
	# cv2.imwrite(dump_dir+'/{0}_raw_final_pose.png'.format(i), final_image)

	print('time in 1 sample:',time.time()-st)

	# draw raw poses
	best_rectangles = grasp_pose_info['best_rectangles']
	for j,final_rect_pixel_array in enumerate(best_rectangles):
		raw_img = process_image_before_rendering(img.copy())
		draw_rectified_rect(img=raw_img, pixel_points=2*final_rect_pixel_array)
		cv2.imwrite(dump_dir+'/{0}_raw_{1}.png'.format(i,j),raw_img)

	# draw gqs_map
	gqs_map = 100*grasp_pose_info['Pred']
	gqs_map = np.where(gqs_map < 0 , 0 , gqs_map)
	draw_grasp_map(gqs_map,dump_dir+'/gqs_map/{0}_gqs_map.png'.format(i))

	filter_G = (1,1)

	# # draw top-N seed points
	top_grasp_points = 2*grasp_pose_info['top_grasp_points']
	topN_img = process_image_before_rendering(img.copy())
	draw_top_N_points(top_grasp_points,topN_img)
	cv2.imwrite(dump_dir+'/{0}_topN_points.png'.format(i),topN_img)

	# # draw sampled grasp poses
	sampled_poses_img = process_image_before_rendering(img.copy())
	rectangle_all = grasp_pose_info['rectangle_all']
	for rectangle in rectangle_all:
		draw_grasp_pose_as_a_line(sampled_poses_img, 2*rectangle)
	cv2.imwrite(dump_dir+'/{0}_sampled_poses.png'.format(i),sampled_poses_img)

	# # draw valid grasp poses
	valid_poses_img = process_image_before_rendering(img.copy())
	gdi_calculators = grasp_pose_info['selected_gdi_calculators']
	for idx in grasp_pose_info['max_idx_per_point']:
		gdi_calculators[idx].draw_refined_pose(process_image_before_rendering(img.copy()),path=dump_dir+'/sampled_poses{0}.png'.format(idx), scale=2, thickness=4)
		gdi_calculators[idx].draw_refined_pose(valid_poses_img, scale=2, thickness=4)

	cv2.imwrite(dump_dir+'/{0}_valid_poses.png'.format(i), valid_poses_img)

	# # draw final pose
	final_img = process_image_before_rendering(img.copy())
	gdi_calc = grasp_pose_info['gdi_calculator']
	gdi_calc.draw_refined_pose(final_img, scale=2, thickness=4)
	# final_img = final_img/0.8 
	# for ch in range(3):
	# 	final_img[:,:,ch] = np.where(final_img[:,:,ch]<=255 , final_img[:,:,ch], 255)
	cv2.imwrite(dump_dir+'/{0}_final.png'.format(i), final_img)

	# c = input('press 1 and enter to continue')
	if i > 0:
		avg_time += time.time()-st
		print('inference time',time.time()-st)
		print('avg_time',avg_time/i)
	gqcnn_score_list.append(gqs_score)
	
	# np.savetxt(out_path+'/score_list.txt',gqcnn_score_list,fmt='%3d')
	# cv2.imwrite(out_path+'/final_image_{0}.png'.format(idx),final_image)
	# np.savetxt(out_path+'/grasp_{0}.txt'.format(idx),final_pose_rectangle)
	# # cv2.imwrite(out_path+'/cluster_image_{0}.png'.format(idx),cluster_image)
	# cv2.imwrite(out_path+'/depth_image_{0}.png'.format(idx),dv)
	# # cv2.imwrite(out_path+'/image_cnn_pose_{0}.png'.format(idx),image_cnn_pose)
	# # c = input('ruko. analyze karo.')
	
	print('acc',np.count_nonzero(gqcnn_score_list)/scenes)
	scenes += 1
avg_time = avg_time/(i-1)
print('avg_time',avg_time)
	
	# dv[dv < 150] = 255
	# obj_region = (dv > 150) 
	# dv[dv > 100] = 255
	

	