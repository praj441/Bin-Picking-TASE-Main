#!/usr/bin/env python3

import rospy, sys, os, numpy as np
import moveit_commander
from copy import deepcopy
import moveit_msgs.msg
from sensor_msgs.msg import Image
from time import sleep


from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from std_msgs.msg import Bool

import cv2, cv_bridge
from sensor_msgs.msg import Image, PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
import pcl
# from scipy.misc import imsave
from matplotlib.pyplot import imread
import matplotlib.pyplot as plt
# from find_grasp_regions import run_grasp_algo
from plyfile import PlyData, PlyElement
import time
import threading
from camera import Camera

# from filter_pixels import depth_filter
from click_empty_bin_median_depth_map import save_median_depth_empty_bin

sys.path.append('commons/')
from utils_gs import Parameters
# sys.path.append('simulation/')
# from cluster_graspability_annotation_parallel_min import generate_graspability_scores

def write_ply(points, filename,color,text=True):
    """ input: Nx3, write points to filename as PLY format. """
    points = [(points[i,0], points[i,1], points[i,2], color[i,2], color[i,1], color[i,0]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4'),('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(filename)

def write_ply_distance(points, filename,color,text=True):
    """ input: Nx3, write points to filename as PLY format. """
    points = [(points[i,0], points[i,1], points[i,2], 255-int(100*color[i])%255, 255-int(100*color[i])%255, 255-int(100*color[i])%255) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4'),('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(filename)

def write_ply_monocolor(points, filename,color=[255,255,255],text=True):
    """ input: Nx3, write points to filename as PLY format. """
    points = [(points[i,0], points[i,1], points[i,2], color[2], color[1], color[0]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4'),('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(filename)


def process_dmap_for_vis(dmap,md):
    dmap[dmap<0.4] = dmap.max()
    # md = np.loadtxt('median_depth_map.txt')
    md = cv2.resize(md,(320,240))
    obj_region = ((md - dmap) > 0.01)
    dmap = np.where(obj_region,dmap-0.4,dmap)
    dv = (dmap/dmap.max()*255)

    obj_region = obj_region 

    dv = np.where(obj_region,np.power(dv,1.5),dv)
    omax = dv[obj_region].max()
    omin = dv[obj_region].min()
    dv = np.where(obj_region,((dv-omin)/omax*255),dv)
    return dv


# def update_mask_for_current_scene(running_mask,slice_mask,obj_id):
#     running_mask[slice_mask] = obj_id



cam_path = 'temp'
cam = Camera(cam_path)
w = 640 
h = 480 
param = Parameters(w,h)

episode = sys.argv[1]
data_path = 'real_data/{0}'.format(episode)
if not os.path.exists(data_path):
    print('creating directory:',data_path)
    os.makedirs(data_path)
else:
    print('**********The path-',data_path,' already exists.***************')
    sys.exit()

def main():

    
    seq = 0
    median_dmap = save_median_depth_empty_bin(data_path,cam)
    # median_depth_map = np.loadtxt(data_path+'/median_depth_map.txt')
    # median_depth_map = cv2.resize(median_depth_map,(w,h))

    while True:
        seq += 1
        # cam.click_an_image_sample()
        # bgr = cv2.resize(cam.cur_image,(w,h))
        
        c = input('bhai jao aur objects arrange karo aur vapas aaker 1 number enter karo!')
        # st = time.time()
        cam.click_a_camera_sample()
        darray = cv2.resize(cam.cur_depth_map,(w,h))
        img = cv2.resize(cam.cur_image,(w,h))

        pc_arr_cur = cv2.resize(cam.cur_pc,(w,h))
        cond = np.isfinite(pc_arr_cur) #& np.isfinite(pc_arr_past[:,:,1]) & np.isfinite(pc_arr_past[:,:,2])
        pc_arr_cur = np.where(cond,pc_arr_cur,0.0)

        
        # dmap_vis = (darray / darray.max())*255
        # dmap_vis = process_dmap_for_vis(darray,median_depth_map)
        # running_mask_vis = (running_mask/running_mask.max())*255
        scene = seq
        # saving the data sample
        np.save(data_path+'/{0:03d}_pc.npy'.format(scene),pc_arr_cur)
        # np.save(data_path+'/{0:03d}_num_objects.npy'.format(scene),[seq])
        np.savetxt(data_path+'/{0:03d}_depth_array.txt'.format(scene),darray)
        cv2.imwrite(data_path+'/{0:03d}_ref_image.png'.format(scene),img)
        # cv2.imwrite(data_path+'/{0:03d}_depth_image.png'.format(scene),dmap_vis)

       

main()
# rospy.loginfo("hi, is this the start")
# rospy.spin()
