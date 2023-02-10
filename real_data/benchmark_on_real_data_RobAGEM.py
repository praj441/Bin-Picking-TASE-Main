import numpy as np
import cv2
import time
import sys, os
from datetime import datetime
from math import *

sys.path.append('../commons')
from baseline_grasp_algo import run_grasp_algo
from utils_gs import Parameters

data_path = 'test_data_mix'
out_path = 'test_data_mix'
method = '/RoBA-GEM'
w = 640 #320
h = 480 #240
param = Parameters(w,h)
# self.raw_data_path = os.path.join(ROOT_DIR, 'sunrgbd/sunrgbd_trainval')
scan_names = sorted(list(set([os.path.basename(x)[0:6] \
	for x in os.listdir(data_path)])))
print(len(scan_names))
input()

score_list = []
median_depth_map = np.loadtxt(data_path+'/median_depth_map.txt')
median_depth_map = cv2.resize(median_depth_map,(param.w,param.h))

avg_time = 0.0
for idx in scan_names:
	dmap = np.loadtxt(os.path.join(data_path, idx)+'_depth_array.txt')
	img = cv2.imread(os.path.join(data_path, idx)+'_ref_image.png')
	
	# dt = datetime.now().strftime("%d%m%Y")
	# h,w,_ = img.shape
	# param = Parameters(w,h)

	inputs = {'image':img}
	inputs['darray'] = dmap
	inputs['depth_image'] = None
	inputs['param'] = param
	inputs['final_attempt'] = True
	inputs['dump_dir'] = out_path+'/RoMA-GEM/'+ idx
	inputs['median_depth_map'] = median_depth_map
	inputs['adaptive_clusters'] = True

	st = time.time()
	result = run_grasp_algo(inputs)
	avg_time += time.time()-st

	grasp_score = result[0][5]
	cluster_image = result[6]
	final_image = result[7]
	score_list.append(grasp_score)
	# bmap_vis = (bmap / bmap.max())*255
	# bmap_vis_denoised = (bmap_denoised / bmap_denoised.max())*255
	# cv2.imwrite(out_path+'/bmaps/bmap{0}.jpg'.format(idx),bmap_vis)#.astype(np.uint8))
	# cv2.imwrite(out_path+'/bmaps/bmap{0}_denoised.jpg'.format(idx),bmap_vis_denoised)#.astype(np.uint8))

	cv2.imwrite(out_path+method+'/final_image_{0}.png'.format(idx),final_image)
	cv2.imwrite(out_path+method+'/cluster_{0}.png'.format(idx),cluster_image)
	np.savetxt(out_path+method+'/score_list.txt',score_list,fmt='%3d')
	print('avg_time',avg_time/len(scan_names))
	print('acc',np.count_nonzero(score_list)/len(scan_names))
avg_time = avg_time/len(scan_names)
	# pc_cloud = np.loadtxt(os.path.join(data_path, idx)+'_pc.npy')
	# c = input('ruko. analyze karo.')

