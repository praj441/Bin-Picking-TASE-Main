#! /usr/bin/env python 
from math import *
import matplotlib.pyplot as plt
import os
from shapely.geometry import Point, Polygon
import numpy as np
import cv2
import copy
from mpl_toolkits.mplot3d import Axes3D
# from sklearn.cluster import KMeans                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
import pickle
import time
import sys
# import rospy
# from point_cloud.srv import point_cloud_service
from scipy.signal import medfilt2d
from scipy.interpolate import griddata
import warnings
warnings.filterwarnings("ignore")

manualSeed = np.random.randint(1, 10000)  # fix seed
# print("Random Seed: ", manualSeed)
np.random.seed(manualSeed)
cv2.setRNGSeed(manualSeed)

from sklearn.cluster import KMeans

from grasp_evaluation import calculate_GDI2
from utils_gs import select_best_rectangles_gdi_old_way, query_point_cloud_client, final_axis_angle
from utils_gs import keep_angle_bounds, height_difference_consideration, select_best_rectangles, draw_rectified_rect
from utils_gs import draw_rectified_rect_plain, normalize_gdi_score
from utils_gs import Parameters
from joblib import Parallel, delayed
from utils_gs import create_directory
# param = Parameters()
# score_arr = None

all_pixels_info = []
sampled_positive_list = []  
sampled_negative_list = []
sampled_gdi_plus_positive_list = [] 
sampled_gdi_plus_negative_list = []  
rectangle_list = []
pixels_corresponding_indices = []
level_label_list = [] # for debugging purpose

GDI = []
GDI_plus = []
GDI_calculator = []
GDI_calculator_all = []
rectangle_list_all = []
original_idx = []
start_time = time.time()

gpose_count = 0
directions = 4
gripper_width = 20
final_attempt=True
# if final_attempt:
#     directions = 4

def process_a_single_sample(k,each_point,inputs,all_clusters_img=None,dump_dir=None):
    # global score_arr
    darray = inputs['darray']
    param = inputs['param']
    dump_dir = inputs['dump_dir']
    st = time.time()
    # dict = param.sample_random_grasp_pose(each_point)
    # minor_points = dict["minor_axis_points"]
    # angle = dict["angle"]

    # angle = np.random.uniform(-pi/2,pi/2)

    # rectangle_pixels_list, angle_list, centroid = param.draw_rect_generic(centroid=each_point, angle=angle + radians(90), directions=directions)
    rectangle_pixels_list, angle_list, centroid = param.draw_rect_generic_fix_angles(centroid=each_point, directions=inputs['num_dirs'])
    # print(angle_list) #[0.0, 0.5235987755982988, 1.0471975511965976, 1.5707963267948966, -1.0471975511965979, -0.5235987755982987]
    # start_time = time.time()
    result = []
    if darray[int(centroid[1]),int(centroid[0])]:
        for index,rectangle_pixels in enumerate(rectangle_pixels_list):
            bmap,gdi,gdi_plus,gdi2, bmap_denoised,cx,cy  = calculate_GDI2(inputs,rectangle_pixels,angle_list[index]-radians(180))
            
            if gdi is not None and gdi_plus is not None:
                score = (gdi+gdi_plus)/2
                grasp_width = (gdi2.gripper_opening/param.gripper_height)
                result.append([cx,cy,gdi2.invalid_id,grasp_width,centroid[0],centroid[1]])
                print(k,index,'valid\n')
                gdi2.draw_refined_pose(copy.deepcopy(inputs['image']),path=dump_dir+'/poses/{0}_{1}.png'.format(k,index))
            else:
                if gdi is not None and gdi_plus is None:
                    gdi2.invalid_reason = 'small contact region'
                    gdi2.invalid_id = 4
                result.append([cx,cy,gdi2.invalid_id,0,centroid[0],centroid[1]])
                print(k,index,'\n')
                img = param.draw_rect_over_image(rectangle_pixels,copy.deepcopy(inputs['image']))
                cv2.imwrite(dump_dir+'/poses/{0}_{1}.png'.format(k,index),img)
            with open(dump_dir+'/grasp_pose_info/invalid_reason_{0}_{1}.txt'.format(k,index), 'w') as f:
                f.write(gdi2.invalid_reason)
            cv2.imwrite(dump_dir+'/bmaps/bmap{0}_{1}.jpg'.format(k,index),bmap_denoised)
            cv2.imwrite(dump_dir+'/bmaps/bmap{0}_{1}_ws.jpg'.format(k,index),gdi2.bmap_ws)
            np.savetxt(dump_dir+'/grasp_pose_info'+'/rectangle_{0}_{1}.txt'.format(k,index),rectangle_pixels)
            np.savetxt(dump_dir+'/grasp_pose_info'+'/angle_{0}_{1}.txt'.format(k,index),[angle_list[index]-radians(180)])
                # cv2.imwrite('temp/poses/{0}_{1}.png'.format(k,index),img)
            # cv2.imwrite('temp/bmaps/bmap{0}_{1}_denoised.jpg'.format(k,index),bmap_denoised)
    return np.array(result)
            
    #         img_copy = copy.deepcopy(all_clusters_img)
    #         img_path = dump_dir+'/directions/gpose{0}_{1}.jpg'.format(k,index)
    #         if final_attempt:
    #             GDI_calculator_all.append(gdi2)
    #             rectangle_list_all.append(rectangle_pixels)
    #         if gdi is not None and gdi_plus is not None: # valid grasp pose
    #             original_idx.append([k,index])
    #             GDI_calculator.append(gdi2)
    #             GDI.append(gdi)
    #             GDI_plus.append(gdi_plus)
    #             rectangle_list.append(rectangle_pixels)
                
    #             gdi2.draw_refined_pose(img_copy,path=img_path)
    #         else:
    #             # img_copy = copy.deepcopy(all_clusters_img)
    #             draw_rectified_rect(img=img_copy, pixel_points=rectangle_pixels,path=img_path)
                

    #         # gpose_count += 1
    #         # cv2.imwrite(dump_dir+'/bmaps/gpd_map{0}_{1}.jpg'.format(k,index),gpd_map)#.astype(np.uint8))
    #         # cv2.imwrite(dump_dir+'/bmaps/gpd_map{0}_{1}_denoised.jpg'.format(k,index),gpd_map_denoised)#.astype(np.uint8))
    #         cv2.imwrite(dump_dir+'/bmaps/bmap{0}_{1}.jpg'.format(k,index),bmap)#.astype(np.uint8))
    #         cv2.imwrite(dump_dir+'/bmaps/bmap{0}_{1}_laplacian.jpg'.format(k,index),gdi2.laplacian)
    #         cv2.imwrite(dump_dir+'/bmaps/bmap{0}_{1}_denoised.jpg'.format(k,index),bmap_denoised)#.astype(np.uint8))
    #         # cv2.imwrite(dump_dir+'/directions/gpose{0}_{1}.jpg'.format(k,index), img_copy)
    # print('processed',k,' in time',time.time()-st)
        # cv2.imwrite(dump_dir+'/clusters/clusters_{0}.png'.format(i),cluster_img)


def generate_graspability_scores(inputs,centroid_pixels):
    global score_arr
    darray = inputs['darray']
    dump_dir = inputs['dump_dir']
    create_directory(dump_dir+'/bmaps')
    create_directory(dump_dir+'/poses')
    create_directory(dump_dir+'/grasp_pose_info')
    N = centroid_pixels.shape[0]
    score_arr = -1*np.ones(darray.shape)
    h,w = darray.shape
    angle_arr = -1*np.ones((h,w,inputs['num_dirs'],2))
    width_arr = -1*np.ones(darray.shape)
    # score_arr[:,0:2] = centroid_pixels
    print('shape check',N)
    image = copy.deepcopy(inputs['image'])
    st = time.time()
    results = Parallel(n_jobs=10)(delayed(process_a_single_sample)(k,each_point,inputs) for k,each_point in enumerate(centroid_pixels))
    for k in range(N):
        result = results[k]
        for index in range(result.shape[0]):
            sample = result[index]
            i = int(sample[0])
            j = int(sample[1])
            invalid_id = int(sample[2])
            angle_arr[j,i,index,0] = invalid_id
            angle_arr[j,i,index,1] = k
            cv2.circle(image, (i, j), invalid_id , (0,0,0), -1)
        cv2.circle(image, (int(centroid_pixels[k][0]), int(centroid_pixels[k][1])), 1, (0,0,255), -1)
    # for k,each_point in enumerate(centroid_pixels):
    #     result = process_a_single_sample(k,each_point,inputs)
    #     i = int(centroid_pixels[k][0])
    #     j = int(centroid_pixels[k][1])
    #     score_arr[j,i] = result[2]
    # image = draw_clusters_into_image(image,centroid_pixels,score_arr)
    # cv2.save('filled_pixels.npy',)
    cv2.imwrite('temp/0_{0}.png'.format(k),image)
        # c = input('dekho!!')
    print('total time in all grasp poses',time.time()-st)
     
    return copy.deepcopy(score_arr),copy.deepcopy(angle_arr) ,copy.deepcopy(width_arr)