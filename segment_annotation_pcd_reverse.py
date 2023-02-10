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
from cluster_graspability_annotation_parallel_min import generate_graspability_scores


def remove_outlier(points):
    mean = np.mean(points, axis=0)
    sd = np.std(points, axis=0)
    mask = (points > (mean - 2 * sd)) & (points < (mean + 2 * sd))
    mask = mask.all(axis=1)
    # print('removed ',points.shape[0]-np.count_nonzero(mask),' outliers')
    return points[mask], mask

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


w = 320
h = 240
param = Parameters(w,h)

data_path = 'real_data/test_clutter'
if not os.path.exists(data_path):
    print('path does not exists:',data_path)
    sys.exit()   

seq = int(sys.argv[2])

def main(inp_data_path):

    scene = 0
    median_depth_map = np.loadtxt(inp_data_path+'/median_depth_map.txt')
    median_depth_map = cv2.resize(median_depth_map,(w,h))
    while True:
        scene += 1
        # cam.click_an_image_sample()
        # bgr = cv2.resize(cam.cur_image,(w,h))
        pc_arr_cur = np.load(inp_data_path+'/{0:03d}_pc.npy'.format(scene))
        # np.save(data_path+'/{0:03d}_num_objects.npy'.format(scene),[seq])
        darray = np.loadtxt(inp_data_path+'/{0:03d}_depth_array.txt'.format(scene))
        running_mask = np.loadtxt(inp_data_path+'/{0:03d}_seg_mask.txt'.format(scene))
        img = cv2.imread(inp_data_path+'/{0:03d}_ref_image.png'.format(scene))

        inputs = {'image':img}
        darray = inputs['darray']
        pc_arr = inputs['pc_arr_cur']
        param = inputs['param']
        median_depth_map = inputs['median_depth_map']
        S=None

        filtered_pc_arr, cluster_img, centroid_pixels = depth_filter(inputs)
        # scene = seq
        print(filtered_pc_arr.shape)
        mask_arr = filtered_pc_arr[:,6]
        background_n_noise_filter = (mask_arr > 0) & (filtered_pc_arr[:,2] > 0)

        filtered_pc_arr = filtered_pc_arr[background_n_noise_filter]
        mask_arr = filtered_pc_arr[:,6]
        print('mask',mask_arr.shape)
        # point_cloud = filtered_pc_arr[:,0:3]
        # color_cloud = filtered_pc_arr[:,3:6]
        obj_ids = np.unique(mask_arr)
        # pcd.points = o3d.utility.Vector3dVector(point_cloud)
        print('obj ids', obj_ids)

        N = filtered_pc_arr.shape[0]

        pc_points = np.zeros((0,7))
        # pc_color = np.zeros((0,3))
        label = np.zeros((0,3))

        pcd1 = o3d.geometry.PointCloud()
        mean_points = []
        for idt in obj_ids:
            mask = (mask_arr==idt)
            points_idt = filtered_pc_arr[mask]
            # color_idt = color_cloud[mask]
            # print(idt,points_idt.shape)
            n = points_idt.shape[0]
            # print(idt,n)
            if n > 50: 
                # print(points_idt.shape)
                # points_idt = np.append(points_idt,mean_point,axis=0)
                # print(points_idt.shape)
                # points_idt[-1,:] = mean_point
                points_idt, outlier_mask = remove_outlier(points_idt)
                # color_idt = color_idt[outlier_mask]
                mean_point = np.mean(points_idt[:,0:3],axis=0)

                mean_points.append(mean_point)
                # pcd.points = o3d.utility.Vector3dVector(points_idt)
                # o3d.visualization.draw_geometries([pcd])
                offsets = mean_point - points_idt
                # label[mask][outlier_mask] = offsets
                pc_points = np.concatenate((pc_points,points_idt))
                # pc_color = np.concatenate((pc_color,color_idt))
                label = np.concatenate((label,points_idt))
                # offsets = []
                # for i in range(n):
                #   offsets.append(mean_point-points_idt[i])
                # offsets = np.array(offsets)

        mean_points = np.array(mean_points)
        print('labels',mean_points.shape)

        dmap_vis = process_dmap_for_vis(darray,median_depth_map)

        # saving the data sample
        np.save(data_path+'/{0:06d}_pc_complete.npy'.format(seq),pc_arr_cur)
        np.save(data_path+'/{0:06d}_pc.npy'.format(seq),pc_points)
        np.save(data_path+'/{0:06d}_label.npy'.format(seq),label)
        np.save(data_path+'/{0:06d}_num_objects.npy'.format(seq),[scene])
        np.savetxt(data_path+'/{0:06d}_depth_array.txt'.format(seq),darray)
        cv2.imwrite(data_path+'/{0:06d}_ref_image.png'.format(seq),img)
        cv2.imwrite(data_path+'/{0:06d}_cluster_image.png'.format(seq),cluster_img)
        cv2.imwrite(data_path+'/{0:06d}_depth_image.png'.format(scene),dmap_vis)
        # cv2.imwrite(data_path+'/{0:06d}_depth_image.png'.format(seq),dmap_vis)

        # annotating for graspability
        # M = 1024
        # if M > N:
        #     M = N
        # centroid_pixels_kp = np.array(random.sample(list(centroid_pixels), M))
        # grasp_quality_score_arr = generate_graspability_scores(darray,centroid_pixels_kp,param) #same size as darray
        # np.savetxt(data_path+'/{0:06d}_gqs_array.txt'.format(seq),grasp_quality_score_arr)
        # print('background',np.count_nonzero(grasp_quality_score_arr==-1))
        # print('invalid',np.count_nonzero(grasp_quality_score_arr==0))
        # print('valid',np.count_nonzero(grasp_quality_score_arr>0))

        # grasp_quality_score_arr_vis = (grasp_quality_score_arr / grasp_quality_score_arr.max())*255
        # cv2.imwrite(data_path+'/{0:06d}_gqs_image.png'.format(seq),grasp_quality_score_arr_vis)

        # write_ply_monocolor(mean_points, data_path + '/{0:06d}_labels.ply'.format(seq), text=True) #,color=bgr_sliced
        # bgr = np.ones(filtered_pc_arr.shape)
        # mask_arr_vis = (mask_arr/mask_arr.max())*255
        # bgr[:,0] = mask_arr_vis
        # bgr[:,1] = mask_arr_vis
        # bgr[:,2] = mask_arr_vis
        # write_ply(pc_points[:,0:3], data_path + '/pc_arr_{0:06d}.ply'.format(seq),color=pc_points[:,3:6], text=True)

        # cv2.imwrite(data_path + '/sliced_img_{0:03d}.png'.format(seq), bgr_sliced)
    
        # write_ply_distance(pc_arr, 'result_dir/sliced_distance.ply',color=pc_distance, text=True)
        # print('time in 1 sample', time.time()-st)

    seq = seq + 1
# episode = int(sys.argv[1])
last_episode = int(sys.argv[1])
for episode in range(last_episode):
    inp_data_path = 'real_data/{0}'.format(episode)
    if not os.path.exists(inp_data_path):
        print('path does not exists:',inp_data_path)
        sys.exit()
    main(inp_data_path)
# rospy.loginfo("hi, is this the start")
# rospy.spin()
