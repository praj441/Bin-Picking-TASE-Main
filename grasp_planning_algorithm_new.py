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
import rospy
from point_cloud.srv import point_cloud_service
import warnings

from scipy.interpolate import griddata

sys.path.append('commons')
# from filter_pixels import depth_filter
from utils_gs import Parameters, pixel_to_xyz
from baseline_grasp_algo import run_grasp_algo

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

manualSeed = np.random.randint(1, 10000)  # fix seed
# print("Random Seed: ", manualSeed)
np.random.seed(manualSeed)
cv2.setRNGSeed(manualSeed)

rospy.wait_for_service('point_cloud_access_service')
get_3d_cam_point = rospy.ServiceProxy('point_cloud_access_service', point_cloud_service)

def grasp_planning(mp):

    print('entering gp')
    path = mp.path
    while not mp.image_end or not mp.depth_end:
        time.sleep(0.01)
        continue;


    w = 320
    h = 240
    param = Parameters(w,h)
    param.THRESHOLD2 = 0.02

    #saving full resolution version
    image_org = mp.cur_image
    dmap_org = mp.cur_dmap.astype(np.float64)/1000
    pc_arr = mp.cur_pc
    # print('shapes',image.shape,dmap.shape)
    image = cv2.resize(copy.deepcopy(image_org),(w,h))
    dmap = cv2.resize(copy.deepcopy(dmap_org),(w,h)) #dmap[::2,::2]
    # print('shapes',image.shape,dmap.shape)
    dmap_vis = (dmap / dmap.max())*255
    np.savetxt(path+'/depth_array.txt',dmap)
    cv2.imwrite(path+'/ref_image.png',image)
    cv2.imwrite(path+'/depth_image.png',dmap_vis)

    np.savetxt('result_dir/depth_array.txt',dmap)
    cv2.imwrite('result_dir/ref_image.png',image)
    cv2.imwrite('result_dir/depth_image.png',dmap_vis)

    # inputs = {'image':image}
    # inputs['darray'] = dmap
    # inputs['pc_arr'] = pc_arr
    # inputs['param'] = param
    # inputs['median_depth_map'] = median_depth_map
    # inputs['seg_mask']


    counts, bins = np.histogram(dmap_vis)
    # print(counts)
    # print(bins)
    plt.hist(bins[:-1], bins, weights=counts)
    # plt.savefig(path+'/hist.png')

    dmap_vis[np.where(dmap_vis < bins[1])] = 255
    dmap_vis = ((dmap_vis-dmap_vis.min())/dmap_vis.max())*255
    cv2.imwrite(path+'/depth_image1.png',dmap_vis)
    #******************************* camera alignment correction here *************************
    inverse_cam_transform = None
    # dmap,image, inverse_cam_transform = self.virtual_camera_transformation(dmap,image)

    # grasp planning
    start_time = time.time()
    total_attempt = 3
    final_attempt = False

    median_depth_map = np.loadtxt('median_depth_map.txt')
    median_depth_map = cv2.resize(median_depth_map,(param.w,param.h))
    inputs = {'image':image.copy()}
    inputs['darray'] = dmap.copy()
    inputs['depth_image'] = dmap_vis
    inputs['param'] = param
    inputs['dump_dir'] = path
    inputs['median_depth_map'] = median_depth_map
    inputs['num_dirs'] = 6

    for attempt_num in range(total_attempt):
        if attempt_num == 2:
            final_attempt = True


        st = time.time()
        inputs['final_attempt'] = final_attempt

        results = run_grasp_algo(inputs)
        mp.action,flag,center,valid, boundary_pose, min_depth_difference, mp.fov_points, final_img = results[0:8]
        
        # cy,cx = center
        # cx = (3.2/param.mw)*cx
        # cy = (2.4/param.mh)*cy
        # z = dmap_org[cy][cx]
        # x,y,z = pixel_to_xyz(cx,cy,z,w=640,h=480,fx=614.72,fy=614.14)
        # print('FOV', x,y,z)

        # try:
        #     x,y,z = get_3d_cam_point(np.array([cx, cy])).cam_point
        # except rospy.ServiceException as e:
        #     print("Point cloud Service call failed: %s"%e)
        # mp.action[0:3] = [x,y,z]
        # print('pc service',x,y,z)

        cv2.imwrite('result_dir/final/final_{0}.jpg'.format(mp.exp_num), final_img)
        # filtered_pc_arr = depth_filter(image.copy(),dmap.copy(),path,pc_arr)
        # np.save('result_dir/filtered_point_cloud_array.npy',filtered_pc_arr)
        print('time in a loop',time.time()-st)
        if valid:
            break
    if not flag:
        print('error')
        return
    print('output',mp.action)
    # declutter action if no valid grasp found
    if not valid:
        img = cv2.imread(path+'/ref_image.png')
        darray = np.loadtxt(path+'/depth_array.txt')
        darray_empty_bin = np.loadtxt(path+'/../depth_array_empty_bin.txt')
        start_point,end_point = disperse_task(img,darray,darray_empty_bin,center,path)
        np.savetxt(path+'/start_point.txt',start_point,fmt='%d')
        np.savetxt(path+'/end_point.txt',end_point,fmt='%d')
        mp.declutter_action.actionDeclutter(start_point,end_point,darray)

    else:
        mp.gripper_opening = int(100*mp.action[4])#+5
        mp.gripper_closing = mp.gripper_grasp_value
        print('gripper_opening',mp.gripper_opening)

        # code to be optimized 
        # datum_z = 0.575 #0.640
        mp.finger_depth_down = 0.05
        # if mp.action[2] > (datum_z-0.042): #0.540:
        #     mp.finger_depth_down = (datum_z-0.042+mp.finger_depth_down) - mp.action[2]
        

        # if mp.finger_depth_down + mp.action[2] > datum_z:
        #     mp.finger_depth_down = datum_z - mp.action[2] + 0.015
        # if boundary_pose:
        #     print('boundary_pose',boundary_pose)
        #     mp.finger_depth_down += -0.004

        print('finger_depth_down:',mp.finger_depth_down,mp.action[2])
        # print('min_depth_difference:',min_depth_difference)
        
        # if finger_depth_down > min_depth_difference :
        #     finger_depth_down = min_depth_difference
        # self.gripper.run(self.gripper_homing_value) #*gripper_opening+5)        

        # if inverse_cam_transform is not None:
        #     a[0:3] = self.affine_transformation(inverse_cam_transform,a[0:3])
        
        print('time taken by the algo:{0}'.format(time.time()-start_time))                                                                                                                                                                                                                            
        print('exiting gp')


if __name__ == "__main__":
    num_obj = 10
    case = 0
    version = 0 # full method
    if len(sys.argv) > 1:
        case = sys.argv[1]
    if len(sys.argv) > 2:
        version = int(sys.argv[2])
    path = '../images_ce/{0}/{1}'.format(10,case)

    # manualSeed = random.randint(1, 10000)  # fix seed
    # print("Random Seed: ", manualSeed)
    # random.seed(manualSeed)
    # np.random.seed(manualSeed)

    sample_dirs = 1
    fix_cluster = False
    FSL_only = False
    CRS_only = False
    pose_refine = True
    center_refine = False

    if version == 1:
        pose_refine = False 
        center_refine = True     
    if version == 2:
        CRS_only = True  # w/o FSL
    if version == 3:
        FSL_only = True   # w/o CSR
    if version == 4:
        pose_refine = False
    total_attempt = 1
    final_attempt = False



    image = cv2.imread(path+'/ref_image.png')
    darray = np.loadtxt(path+'/depth_array.txt')
    darray = interpolate_noisy_2d_map(darray)
    # start_time = time.time()
    # run_grasp_algo(img,darray,case=case,final_attempt=final_attempt)
    # print('time:', time.time()-start_time)

    # action,flag,center,valid, boundary_pose, min_depth_difference, fov_points = run_grasp_algo(image.copy(),darray.copy(),path,final_attempt=final_attempt)
    # print('min_depth_difference',min_depth_difference)
