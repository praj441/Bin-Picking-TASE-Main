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

data_path = 'real_data/test_data_cas'
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

        inputs = {'image':img}
        inputs['darray'] = darray
        inputs['pc_arr'] = pc_arr_cur
        inputs['param'] = param
        inputs['median_depth_map'] = median_depth_map
        inputs['seg_mask'] = None

        filtered_pc_arr, cluster_img, centroid_pixels, filter_mask = depth_filter(inputs,discard_prob=0.3)
        # scene = seq
        print(filtered_pc_arr.shape)
        # mask_arr = filtered_pc_arr[:,6]
        background_n_noise_filter = (filtered_pc_arr[:,2] > 0)

        filtered_pc_arr = filtered_pc_arr[background_n_noise_filter]
        # mask_arr = filtered_pc_arr[:,6]
        # print('mask',mask_arr.shape)
        # # point_cloud = filtered_pc_arr[:,0:3]
        # # color_cloud = filtered_pc_arr[:,3:6]
        # obj_ids = np.unique(mask_arr)
        # # pcd.points = o3d.utility.Vector3dVector(point_cloud)
        # print('obj ids', obj_ids)

        # N = filtered_pc_arr.shape[0]

        # pc_points = np.zeros((0,7))
        # # pc_color = np.zeros((0,3))
        # label = np.zeros((0,3))

        # pcd1 = o3d.geometry.PointCloud()
        # mean_points = []
        # for idt in obj_ids:
        #     mask = (mask_arr==idt)
        #     points_idt = filtered_pc_arr[mask]
        #     # color_idt = color_cloud[mask]
        #     # print(idt,points_idt.shape)
        #     n = points_idt.shape[0]
        #     # print(idt,n)
        #     if n > 50: 
        #         # print(points_idt.shape)
        #         # points_idt = np.append(points_idt,mean_point,axis=0)
        #         # print(points_idt.shape)
        #         # points_idt[-1,:] = mean_point
        #         points_idt, outlier_mask = remove_outlier(points_idt)
        #         # color_idt = color_idt[outlier_mask]
        #         mean_point = np.mean(points_idt[:,0:3],axis=0)

        #         mean_points.append(mean_point)
        #         # pcd.points = o3d.utility.Vector3dVector(points_idt)
        #         # o3d.visualization.draw_geometries([pcd])
        #         offsets = mean_point - points_idt
        #         # label[mask][outlier_mask] = offsets
        #         pc_points = np.concatenate((pc_points,points_idt))
        #         # pc_color = np.concatenate((pc_color,color_idt))
        #         label = np.concatenate((label,points_idt))
        #         # offsets = []
        #         # for i in range(n):
        #         #   offsets.append(mean_point-points_idt[i])
        #         # offsets = np.array(offsets)

        # mean_points = np.array(mean_points)
        # print('labels',mean_points.shape)
        seq = scene
        # dmap_vis = process_dmap_for_vis(darray,median_depth_map)
        # dmap_vis = np.where(dmap_vis < (0.3*255),dmap_vis.max(),dmap_vis)
        # dmap_vis = 255*dmap_vis/(dmap_vis.max())
        dmap_vis = 255*darray/(darray.max())
        

        # saving the data sample
        np.save(data_path+'/{0:06d}_pc_complete.npy'.format(seq),pc_arr_cur)
        np.save(data_path+'/{0:06d}_pc.npy'.format(seq),filtered_pc_arr)
        # np.save(data_path+'/{0:06d}_label.npy'.format(seq),label)
        # np.save(data_path+'/{0:06d}_num_objects.npy'.format(seq),[scene])
        np.savetxt(data_path+'/{0:06d}_depth_array.txt'.format(seq),darray)

        cv2.imwrite(data_path+'/{0:06d}_ref_image.png'.format(seq),cv2.resize(img,(512,512)))
        cv2.imwrite(data_path+'/{0:06d}_cluster_image.png'.format(seq),cluster_img)
        cv2.imwrite(data_path+'/{0:06d}_depth_image.png'.format(scene),cv2.resize(dmap_vis.astype(np.uint8),(512,512)))
        # cv2.imwrite(data_path+'/{0:06d}_depth_image.png'.format(seq),dmap_vis)
        np.save(data_path+'/{0:06d}_filter_mask.npy'.format(seq),filter_mask)


        if flag_generate_graspability_scores:

            inputs = {'image': img}
            inputs['sim'] = False
            inputs['darray'] = darray
            inputs['seg_mask'] = None
            inputs['param'] = param
            inputs['median_depth_map'] = median_depth_map
            inputs['dump_dir'] = data_path+'/{0}'.format(scene)
            create_directory(inputs['dump_dir']+'/grasp_pose_info')
            create_directory(inputs['dump_dir']+'/poses')
            create_directory(inputs['dump_dir']+'/bmaps')

            # annotating for graspability
            N = centroid_pixels.shape[0]
            M = 800 #1024
            if M > N:
                M = N
            centroid_pixels_kp = np.array(random.sample(list(centroid_pixels), M))
            grasp_quality_score_arr, angle_arr, width_arr = generate_graspability_scores(inputs,centroid_pixels_kp)
            np.savetxt(data_path+'/{0:06d}_gqs_array.txt'.format(scene),grasp_quality_score_arr)
            np.savetxt(data_path+'/{0:06d}_angle_array.txt'.format(scene),angle_arr)
            np.savetxt(data_path+'/{0:06d}_width_array.txt'.format(scene),width_arr)
            # print('background',np.count_nonzero(grasp_quality_score_arr==-1))
            valids = np.count_nonzero(grasp_quality_score_arr>0)
            print('invalid',M-valids)
            print('valid',valids)

            grasp_quality_score_arr_vis = (grasp_quality_score_arr / grasp_quality_score_arr.max())*255
            cv2.imwrite(data_path+'/{0:06d}_gqs_image.png'.format(scene),grasp_quality_score_arr_vis)

            angle_arr_vis = (angle_arr / angle_arr.max())*255
            cv2.imwrite(data_path+'/{0:06d}_angle_image.png'.format(scene),angle_arr_vis)

            width_arr_vis = (width_arr / width_arr.max())*255
            cv2.imwrite(data_path+'/{0:06d}_width_image.png'.format(scene),width_arr_vis)
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
for episode in [last_episode]:
    inp_data_path = 'real_data/{0}'.format(episode)
    if not os.path.exists(inp_data_path):
        print('path does not exists:',inp_data_path)
        sys.exit()
    main(inp_data_path)
# rospy.loginfo("hi, is this the start")
# rospy.spin()
