import sys
import copy
import numpy as np
sys.path.append('../commons/')
from custom_grasp_planning_algorithm_dense import select_a_best_grasp_pose
from utils_gs import points_to_pixels_projection
from utils_gs import draw_clusters_into_image
from utils_gs import draw_contours_around_clusters
from utils_gs import select_top_N_grasping_points_via_top_cluster_method
from utils_gs import select_top_N_grasping_points_via_top_points_method
from utils_gs import select_top_N_grasping_points_via_distance_sampling
from utils_gs import Parameters
from utils_gs import draw_rectified_rect, draw_top_indices
from math import *
import matplotlib.pyplot as plt
import os 
import time
import cv2



class EvaluationResults:
	def __init__(self):
		print('init')
		self.gqs_top_precision = 0.0
		self.gqs_pos_precision = 0.0
		self.gqs_pos_recall = 0.0
		self.gqs_count = 0
		self.gqs_acc = 0.0

		#for training plots
		self.gtp_list = []
		self.gpp_list = []
		self.gpr_list = []
		self.acc_list = []

		self.pos_list = []
		self.neg_list = []

		self.preds_list = []
		self.idx_list = []

	def process_gqs(self,result_dict):
		# self.gqs_top_precision += result_dict['top_pos_precision']
		# self.gqs_pos_precision += result_dict['pos_precision']
		# self.gqs_pos_recall += result_dict['pos_recall'] 
		self.gqs_acc += result_dict['acc']
		self.gqs_count += 1
		
		# self.preds_list.extend(result_dict['preds'].tolist())
		# self.idx_list.extend(result_dict['idx'].tolist())
		# self.pos_list.extend(result_dict['idx'][:,0][np.where(result_dict['preds']==True)].tolist())
		# self.neg_list.extend(result_dict['idx'][:,0][np.where(result_dict['preds']==False)].tolist())

	def output_gqs_stats(self,LOG_DIR=None):
		# self.gqs_top_precision = self.gqs_top_precision/self.gqs_count
		# self.gqs_pos_precision = self.gqs_pos_precision/self.gqs_count
		# self.gqs_pos_recall = self.gqs_pos_recall/self.gqs_count
		self.gqs_acc = self.gqs_acc/self.gqs_count

		# print('gqs top pos precision',self.gqs_top_precision)
		# print('gqs pos precision',self.gqs_pos_precision)
		# print('gqs pos recall',self.gqs_pos_recall)
		print('acc',self.gqs_acc)
		# np.savetxt(os.path.join('preds.txt'),self.preds_list)
		# np.savetxt(os.path.join('idx.txt'),self.idx_list)
		# np.savetxt('pos_list.txt',np.sort(np.array(self.pos_list)),fmt='%d')
		# np.savetxt('neg_list.txt',np.sort(np.array(self.neg_list)),fmt='%d')

		if LOG_DIR is not None:
			# self.gtp_list.append(self.gqs_top_precision)
			# self.gpp_list.append(self.gqs_pos_precision)
			# self.gpr_list.append(self.gqs_pos_recall)
			self.acc_list.append(self.gqs_acc)

			# plt.plot(self.gtp_list)
			# plt.savefig(os.path.join(LOG_DIR,'gqs_top_precision.png'))
			# plt.clf()

			# plt.plot(self.gpp_list)
			# plt.savefig(os.path.join(LOG_DIR,'gqs pos precision.png'))
			# plt.clf()

			# plt.plot(self.gpr_list)
			# plt.savefig(os.path.join(LOG_DIR,'gqs_pos_recall.png'))
			# plt.clf()

			plt.plot(self.acc_list)
			plt.savefig(os.path.join(LOG_DIR,'gqs_acc.png'))
			plt.clf()


			self.gqs_top_precision = 0.0
			self.gqs_pos_precision = 0.0
			self.gqs_pos_recall = 0.0
			self.gqs_acc = 0.0
			self.gqs_count = 0




def draw_final_pose(grasp_pose_info, image, depth_image,scale=1):

	final_pose_rect_img = copy.deepcopy(image)
	# final_pose_rect_img = cv2.GaussianBlur(final_pose_rect_img, (7, 7), 0)
	depth_image_copy = copy.deepcopy(depth_image)


	best_rectangles = grasp_pose_info['best_rectangles']
	final_rect_pixel_array = grasp_pose_info['final_pose_rectangle']
	gdi2 = grasp_pose_info['gdi_calculator']
	original_idx = grasp_pose_info['selected_idx']
	gdi_old_way = grasp_pose_info['gdi_old_way']

	colors = [(255,0,0),(0,255,0),(0,0,255)]
	
	grasp_score = 0.0
	if not gdi_old_way:
		print('original_idx',original_idx)
		new_centroid, new_gripper_opening, object_width = gdi2.draw_refined_pose(final_pose_rect_img,scale=scale)
		gdi2.draw_refined_pose(depth_image_copy)
		# cv2.imwrite(dump_dir+'/depth_image.png',depth_image_copy)
		grasp_score = (gdi2.FLS_score + gdi2.CRS_score)/2
		# print('new_gripper_opening',new_gripper_opening)

	else:
		for i,final_rect_pixel_array in enumerate(best_rectangles):
			if i > 0:
				break
			draw_rectified_rect(img=final_pose_rect_img, pixel_points=final_rect_pixel_array,color=colors[i])

	return final_pose_rect_img, depth_image_copy, grasp_score



def grasp_pose_prediction(end_points,inputs_np):

	image = cv2.GaussianBlur(inputs_np['image'], (3, 3), 0)
	darray = inputs_np['darray']
	depth_image = inputs_np['depth_image']
	dump_dir = inputs_np['dump_dir']
	param = inputs_np['param']

	#clusters into image
	points = end_points['seed_xyz'].detach().cpu().numpy()[0] # (1024,3)
	indices = end_points['aggregated_vote_cluster_inds'].detach().cpu().numpy() # (1024,)
	gqs_score_predicted = end_points['gqs_score'].detach().cpu().numpy()[0]
	

	
	gqs_score_predicted = np.where(gqs_score_predicted<0,0,gqs_score_predicted)
	gqs_score_predicted = 100*gqs_score_predicted

	

	num_obj_prediction = int(100*end_points['num_obj_prediction'].detach().cpu().numpy())

	# print('total time in processing 1 sample',time.time()-st)
	P = points_to_pixels_projection(points,w=param.w,h=param.h,fx=param.f_x,fy=param.f_y)
	cluster_image, seed_image = draw_clusters_into_image(copy.deepcopy(image),P,indices,gqs_score_predicted)
	cv2.imwrite(dump_dir+'/2_graspability_scores.png',cluster_image)
	cv2.imwrite(dump_dir+'/1_seed_image.png',seed_image)
	# cluster_image = draw_contours_around_clusters(cluster_image,P,indices)

	try:
		gqs_score_gt = end_points['gqs_target'].detach().cpu().numpy()[0]
		cluster_image_gt = draw_clusters_into_image(copy.deepcopy(image),P,indices,gqs_score_gt)
	except:
		cluster_image_gt = None
		gqs_score_gt = None
	# top_indices = select_top_N_grasping_points_via_top_cluster_method(P,indices,gqs_score_predicted)
	# top_indices = select_top_N_grasping_points_via_top_points_method(P,gqs_score_predicted, topN=10)
	top_indices = select_top_N_grasping_points_via_distance_sampling(P,gqs_score_predicted, topN=10, dist_thrs = param.w/30)

	# image_cluster, centroid_pixels,label  = draw_clusters_into_image(points,indices,gqs_score_predicted,num_obj_prediction,copy.deepcopy(image),w=320,h=240,fx=307.36,fy=307.07)
	# image_cluster, centroid_pixels,label  = draw_clusters_into_image(points,indices,gqs_score_predicted,num_obj_prediction,copy.deepcopy(image),w=640,h=480,fx=614.72,fy=614.14)

	# image_cluster, centroid_pixels,label = draw_clusters_into_image(points,indices,image.copy(),w=640,h=480,fx=614.72,fy=614.14)
	# final_image = run_grasp_algo(image,darray,depth_image,dump_dir,centroid_pixels,label,final_attempt=True)
	top_grasp_points = P[top_indices]
	gqs_score = gqs_score_predicted[top_indices]
	print('gqs_score_predicted',gqs_score)
	# print('gqs_score_gt',gqs_score_gt[top_indices])
	# angles and widths
	top_indices_image = draw_top_indices(copy.deepcopy(image),top_grasp_points,param.gripper_height)
	print(dump_dir+'/top_pose_sampling.png')
	cv2.imwrite(dump_dir+'/3_top_pose_sampling.png',top_indices_image)
	# angle_score = end_points['angle_target_map'][P[top_indices][:,1].astype(np.int32),P[top_indices][:,0].astype(np.int32)]
	# angle_score_max = angle_score.max(axis=1)
	# print('angle_score_gt',angle_score)
	# angle_score_argmax = angle_score.argmax(axis=1)
	# print('angle_score_argmax',angle_score_argmax)

	# print('angle_score_predicted',end_points['ab_score'].detach().cpu().numpy()[0][top_indices])
	# # angles_predicted = end_points['angle_score'].detach().cpu().numpy()[0]
	# # angle_score = angles_predicted[top_indices]

	# widths_predicted = end_points['width_score'].detach().cpu().numpy()[0]
	# width_score = widths_predicted[top_indices]

	# import cv2
	# colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255),(255,0,255),(255,255,255),(0,0,0),(50,0,100),(155,10,100)]
	# for i,centroid in enumerate(top_grasp_points[0:10]):
	# 	image_cnn_pose = copy.deepcopy(inputs_np['image'])
	# 	angle = angle_list[angle_score_argmax[i]]
	# 	if angle_score_max[i] == 0:
	# 		print(i,'zero angle score')
	# 	param.draw_rect_cnn(image_cnn_pose, centroid, angle, width_score[i],colors[i])
	# 	cv2.imwrite('image_cnn_pose{0}.png'.format(i),image_cnn_pose)

	# input('stop')

	angles = np.array([np.random.uniform(-pi/2,pi/2) for i in range(top_grasp_points.shape[0])])
	inputs_np['top_grasp_points'] = top_grasp_points
	inputs_np['angles'] = angles
	inputs_np['final_attempt'] = True
	# inputs_np['param'] = param
	inputs_np['gqs_score'] = gqs_score
	inputs_np['num_dirs'] = 6
	# inputs_np['sim'] = True
	# final_image = generate_graspability_scores(cp.asarray(darray),dump_dir,cp.asarray(centroid_pixels),cp.asarray(label))
	# st = time.time()
	# calculate_pose_validity_by_cnn(inputs_np['pc_cloud'],points[top_indices])
	# print('time in pose validity',time.time()-st)
	grasp_pose_info = select_a_best_grasp_pose(inputs_np)
	final_image, depth_image_copy, grasp_score = draw_final_pose(grasp_pose_info,copy.deepcopy(image),inputs_np['depth_image'])
	cv2.imwrite(dump_dir+'/4_final_image.png',final_image)
	outputs = {'cluster_image':cluster_image}
	outputs['cluster_image_gt'] = cluster_image_gt
	outputs['final_image'] = final_image
	outputs['grasp_score'] = grasp_score
	outputs['image_cnn_pose'] = final_image #image_cnn_pose
	outputs['final_image'] = final_image
	outputs['grasp_pose_info'] = grasp_pose_info
	# outputs[]
	# outputs['contour_image']

	return outputs

# def calculate_pose_validity_by_cnn(point_cloud,points):
# 	pose_validity = []
# 	for each_point in points:
# 		for j in range(6):
# 			focused_points = get_an_instance(point_cloud,j,each_point)