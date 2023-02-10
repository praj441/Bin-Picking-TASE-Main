import numpy as np
import cv2
import time
import sys, os
from datetime import datetime
from math import *


sys.path.append('../commons')
from utils_gs import  Parameters
from grasp_evaluation import calculate_GDI2
from utils_gs import create_directory

#preparaing dexnet pipeline
grconv_path = '/home/prem/prem_workspace/pybullet_learning/grasping/baseline_antipodal/robotic-grasping'
sys.path.append(grconv_path)
from grasp_predictor import GQCNN_Predictor
model_name = 'GQCNN-4.0-PJ'
camera_intr_file = gqcnn_path+ '/gqcnn/data/calib/primesense/primesense.intr'
# camera_intr_file = '../gqcnn/data/calib/pybullet/pybullet_camera.intr'
gqcnn_predictor = GQCNN_Predictor(model_name,camera_intr_file)


data_path = 'test_data_mid_level_S'
out_path = data_path+'/gqcnn'
w = 320
h = 240
param = Parameters(w,h)
# self.raw_data_path = os.path.join(ROOT_DIR, 'sunrgbd/sunrgbd_trainval')
scan_names = sorted(list(set([os.path.basename(x)[0:6] \
	for x in os.listdir(data_path)])))

inputs = {'param':param}
inputs['slanted_pose_detection'] = False #False
inputs['cone_detection'] = False #False
avg_time = 0.0
gqcnn_score_list = []

# scan_names = ['000019']
scenes = 1
for idx in scan_names:
	dmap = np.loadtxt(os.path.join(data_path, idx)+'_depth_array.txt')
	img = cv2.imread(os.path.join(data_path, idx)+'_ref_image.png')
	# dt = datetime.now().strftime("%d%m%Y")
	st = time.time()
	px,py,angle,d,grasp_width_px,flag = gqcnn_predictor.predict(dmap,None,None)
	avg_time += time.time()-st
	inputs['darray'] = dmap
	dump_dir = out_path + '/' + idx
	create_directory(dump_dir)
	inputs['dump_dir'] = dump_dir
	# inputs['param'] = param

	if angle is not None:
		img,rectangle_pixels = param.draw_rect_gqcnn(img,np.array([int(px),int(py)]), angle)
		bmap,gdi,gdi_plus,gdi2, bmap_denoised ,cx,cy = calculate_GDI2(inputs,rectangle_pixels,angle-radians(180))
		if gdi is not None and gdi_plus is not None:
			gqs_score = (gdi+gdi_plus)/2
			gdi2.draw_refined_pose(img)
		else:
			if gdi_plus is not None:
				gdi2.invalid_reason = 'small contact region'
			gqs_score = 0.0
		print('gqs score',gqs_score,gdi2.FLS_score,gdi2.CRS_score)
		gqcnn_score_list.append(gqs_score)
	# bmap_vis = (bmap / bmap.max())*255
	# bmap_vis_denoised = (bmap_denoised / bmap_denoised.max())*255
	cv2.imwrite(dump_dir+'/bmap.jpg',bmap_denoised)#.astype(np.uint8))
	cv2.imwrite(dump_dir+'/bmap_ws.jpg',gdi2.bmap_ws)#.astype(np.uint8))
	with open(dump_dir+'/invalid_reason.txt', 'w') as f:
			f.write(gdi2.invalid_reason)

	cv2.imwrite(out_path+'/final_image_{0}.png'.format(idx),img)
	np.savetxt(out_path+'/score_list.txt',gqcnn_score_list,fmt='%3d')
	print('avg_time',avg_time/scenes)
	print('acc',np.count_nonzero(gqcnn_score_list)/scenes)
	scenes += 1
avg_time = avg_time/len(scan_names)

	# pc_cloud = np.loadtxt(os.path.join(data_path, idx)+'_pc.npy')
	
	# c = input('ruko. analyze karo.')

