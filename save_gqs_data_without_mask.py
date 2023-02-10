#!/usr/bin/env python3

import  os,sys, numpy as np
from copy import deepcopy
from time import sleep



import cv2

import pcl
# from scipy.misc import imsave
from matplotlib.pyplot import imread
import matplotlib.pyplot as plt
# from find_grasp_regions import run_grasp_algo
from plyfile import PlyData, PlyElement
import time
import threading
import open3d as o3d
import random

sys.path.append('commons/')
from utils_gs import Parameters
from filter_pixels import depth_filter
from cluster_graspability_annotation_parallel_min_end_2_end import generate_graspability_scores
from utils_gs import create_directory


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


# def update_mask_for_current_scene(running_mask,slice_mask,obj_id):
#     running_mask[slice_mask] = obj_id



w = 320
h = 240
param = Parameters(w,h)
flag_generate_graspability_scores = False

data_path = 'real_data/test_data_level_1'
if not os.path.exists(data_path):
    print('path does not exists:',data_path)
    sys.exit()   

# seq = int(sys.argv[2])
def process_dmap_for_vis(dmap,md):
    dmap[dmap<0.4] = dmap.max()
    # md = np.loadtxt('median_depth_map.txt')
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

def draw_graspability_labels(gqs_arr,image):
    # print(I.shape, gqs_arr.shape)
    # input()
    # image1 = copy.deepcopy(image)
    for i in range(w):
        for j in range(h):
            if gqs_arr[j,i] > 0:
                csize = int(gqs_arr[j,i]/20) + 1
                cv2.circle(image, (i, j), csize, (255,0,0), -1)
    
    return image

def main(inp_data_path):

    scene = 0
    median_depth_map = np.loadtxt(inp_data_path+'/median_depth_map.txt')#[::2,::2]
    median_depth_map = cv2.resize(median_depth_map,(w,h))
    while True:
        scene += 1
        # cam.click_an_image_sample()
        # bgr = cv2.resize(cam.cur_image,(w,h))
        pc_arr_cur = np.load(inp_data_path+'/{0:03d}_pc.npy'.format(scene))#[::2,::2,:]
        # np.save(data_path+'/{0:03d}_num_objects.npy'.format(scene),[seq])
        darray = np.loadtxt(inp_data_path+'/{0:03d}_depth_array.txt'.format(scene))#[::2,::2]
        # running_mask = np.loadtxt(inp_data_path+'/{0:03d}_seg_mask.txt'.format(scene))
        img = cv2.imread(inp_data_path+'/{0:03d}_ref_image.png'.format(scene))#[::2,::2,:]


        seq = scene


        inputs = {'image': img}
        inputs['pc_arr'] = pc_arr_cur
        inputs['sim'] = False
        inputs['darray'] = darray
        inputs['seg_mask'] = None
        inputs['param'] = param
        inputs['median_depth_map'] = median_depth_map
        inputs['dump_dir'] = data_path#+'/{0}'.format(scene)
        inputs['num_dirs'] = 6
        create_directory(inputs['dump_dir']+'/grasp_pose_info')
        create_directory(inputs['dump_dir']+'/poses')
        create_directory(inputs['dump_dir']+'/bmaps')

        filtered_pc_arr, cluster_img, centroid_pixels, filter_mask = depth_filter(inputs,discard_prob=0.3)

        # annotating for graspability
        N = centroid_pixels.shape[0]
        M = 1024
        if M > N:
            M = N
        centroid_pixels_kp = np.array(random.sample(list(centroid_pixels), M))
        grasp_quality_score_arr, angle_arr, width_arr = generate_graspability_scores(inputs,centroid_pixels_kp)
        np.savetxt(data_path+'/{0:06d}_gqs_array.txt'.format(scene),grasp_quality_score_arr)
        # np.savetxt(data_path+'/{0:06d}_angle_array.txt'.format(scene),angle_arr)
        # np.savetxt(data_path+'/{0:06d}_width_array.txt'.format(scene),width_arr)
        # print('background',np.count_nonzero(grasp_quality_score_arr==-1))
        valids = np.count_nonzero(grasp_quality_score_arr>0)
        print('invalid',M-valids)
        print('valid',valids)

        # grasp_quality_score_arr_vis = (grasp_quality_score_arr / grasp_quality_score_arr.max())*255
        # cv2.imwrite(data_path+'/{0:06d}_gqs_image.png'.format(scene),grasp_quality_score_arr_vis)

        # angle_arr_vis = (angle_arr / angle_arr.max())*255
        # cv2.imwrite(data_path+'/{0:06d}_angle_image.png'.format(scene),angle_arr_vis)

        # width_arr_vis = (width_arr / width_arr.max())*255
        # cv2.imwrite(data_path+'/{0:06d}_width_image.png'.format(scene),width_arr_vis)
        
        gqs_draw = draw_graspability_labels(grasp_quality_score_arr,img)
        cv2.imwrite(data_path+'/{0:06d}_gqs_draw.png'.format(scene),gqs_draw)

    seq = seq + 1
# episode = int(sys.argv[1])
last_episode = int(sys.argv[1])
for episode in [last_episode]:
    inp_data_path = 'real_data/{0}'.format(episode)
    if not os.path.exists(inp_data_path):
        print('path does not exists:',inp_data_path)
        sys.exit()
    main(inp_data_path)
# rospy.loginfo("hi, is this the start")
# rospy.spin()
