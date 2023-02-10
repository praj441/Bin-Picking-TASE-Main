#! /usr/bin/env python 
from math import *
import numpy as np
import cv2
from mpl_toolkits.mplot3d import Axes3D
import pickle
import time
import copy
from scipy.signal import medfilt2d
from scipy.interpolate import griddata
import warnings
warnings.filterwarnings("ignore")

manualSeed = np.random.randint(1, 10000)  # fix seed
np.random.seed(manualSeed)
cv2.setRNGSeed(manualSeed)
from sklearn.cluster import KMeans

from grasp_evaluation import calculate_GDI2
from utils_gs import select_best_rectangles_gdi_old_way, query_point_cloud_client, final_axis_angle
from utils_gs import keep_angle_bounds, height_difference_consideration, select_best_rectangles, draw_rectified_rect
from utils_gs import draw_rectified_rect_plain, normalize_gdi_score
from utils_gs import Parameters


param = Parameters()

def run_grasp_algo(img,darray,depth_image,dump_dir,centroid_pixels,label_outer,final_attempt=True):
    new_img = copy.deepcopy(img)
    # clustter_img = copy.deepcopy(img)
    rectangle_img = copy.deepcopy(img)
    axis_img = copy.deepcopy(img)
    final_pose_rect_img = copy.deepcopy(img)
    depth_image_copy = copy.deepcopy(depth_image)
    img_copy = copy.deepcopy(img)
    initial_img = copy.deepcopy(img)

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
    # if final_attempt:
    #     directions = 4

    # K-Means Clustering #
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1500, 0.001)
    # span = 1
    # for level in range(num_of_clusters,num_of_clusters+span):
    print('shape check',centroid_pixels.dtype)
    ids = np.unique(label_outer)
    for i in ids:
        cluster_outer = np.array(centroid_pixels[label_outer.ravel() == i, :], np.float32)
        print('cluster shape',cluster_outer.shape)
        num_K = 1 #int(cluster_outer.shape[0]/20)+1
        # _, label, centers = cv2.kmeans(cluster_outer, K=num_K,
        #                               criteria=criteria,
        #                               attempts=6,
        #                               flags=cv2.KMEANS_RANDOM_CENTERS,bestLabels=None)
        kmeans = KMeans(n_clusters=num_K,n_init=6,max_iter=1500)
        kmeans.fit(cluster_outer)
        label = kmeans.labels_
        centers_3d = kmeans.cluster_centers_
        centers = centers_3d[:,0:2]

        cluster_img = copy.deepcopy(img)
        for j in range(len(cluster_outer)):
            # if np.random.random()>0.9:
            green_part = int((label[j]*1)%255)
            blue_part = int((label[j]*90)%255)
            red_part = int((label[j]*213)%255)
            cv2.circle(cluster_img, (int(cluster_outer[j, 0]), int(cluster_outer[j, 1])), 2, (blue_part,green_part,red_part), -1)

        for k in range(len(centers)):
            cluster = np.array(cluster_outer[label.ravel() == k, :], np.int32)
            dict = param.axis_angle(points=cluster)
            # major_points = dict["major_axis_points"]
            minor_points = dict["minor_axis_points"]
            angle = dict["angle"]
            points = cluster

            # cv2.line(axis_img, major_points[0], major_points[1], (255, 0, 0), 2)
            # cv2.line(axis_img, minor_points[0], minor_points[1], (0, 0, 255), 2)
            cv2.line(cluster_img, minor_points[0], minor_points[1], (0, 0, 255), 2)

            color = (np.random.uniform(50, 255), np.random.uniform(50, 250), np.random.uniform(50, 250))

            cv2.rectangle(rectangle_img, (int(np.min(points[:, 0])), int(np.min(points[:, 1]))),
                          (int(np.max(points[:, 0])), int(np.max(points[:, 1]))), color, 3)

            cv2.circle(cluster_img, (int(centers[k, 0]), int(centers[k, 1])), 2, (255, 0, 0), -1)

            rectangle_pixels_list, angle_list, centroid = param.draw_rect(centroid=dict["centroid"], angle=angle + radians(90), directions=directions)

            # start_time = time.time()
            if darray[int(centroid[1]),int(centroid[0])]:
                for index,rectangle_pixels in enumerate(rectangle_pixels_list):
                    bmap,gdi,gdi_plus,gdi2, bmap_denoised ,cx,cy = calculate_GDI2(rectangle_pixels,darray,angle_list[index]-radians(180),param)
                    # print(gdi,gdi_plus)
                    # print('\n\n\n')
                    # gpd_map = (gpd_map / gpd_map.max())*255
                    # gpd_map_denoised_vis = (gpd_map_denoised / gpd_map_denoised.max())*255
                    bmap_vis = (bmap / bmap.max())*255
                    bmap_vis_denoised = (bmap_denoised / bmap_denoised.max())*255
                    # cv2.imwrite(dump_dir+'/bmaps/gpd_map{0}_{1}_{2}.jpg'.format(i,k,index),gpd_map)#.astype(np.uint8))
                    # cv2.imwrite(dump_dir+'/bmaps/gpd_map{0}_{1}_{2}_denoised.jpg'.format(i,k,index),gpd_map_denoised_vis)#.astype(np.uint8))
                    cv2.imwrite(dump_dir+'/bmaps/bmap{0}_{1}_{2}.jpg'.format(i,k,index),bmap_vis)#.astype(np.uint8))
                    cv2.imwrite(dump_dir+'/bmaps/bmap{0}_{1}_{2}_denoised.jpg'.format(i,k,index),bmap_vis_denoised)#.astype(np.uint8))

                    if final_attempt:
                        GDI_calculator_all.append(gdi2)
                        rectangle_list_all.append(rectangle_pixels)
                    if gdi is not None and gdi_plus is not None: # valid grasp pose
                        original_idx.append([k,index])
                        GDI_calculator.append(gdi2)
                        GDI.append(gdi)
                        GDI_plus.append(gdi_plus)
                        rectangle_list.append(rectangle_pixels)
                        img_copy = copy.deepcopy(initial_img)
                        gdi2.draw_refined_pose(img_copy)
                        cv2.imwrite(dump_dir+'/directions/gpose{0}_{1}_{2}.jpg'.format(i,k,index), img_copy)
                    else:
                        img_copy = copy.deepcopy(initial_img)
                        draw_rectified_rect(img=img_copy, pixel_points=rectangle_pixels)
                        cv2.imwrite(dump_dir+'/directions/gpose{0}_{1}_{2}.jpg'.format(i,k,index), img_copy)

                    gpose_count += 1
        cv2.imwrite(dump_dir+'/clusters/clusters_{0}.png'.format(i),cluster_img)
        # else:
            # print('zero depth \n\n')
    
    # st = time.time()
    # cv2.imwrite(path+'/clustter_img.jpg',clustter_img)
    # cv2.imwrite(path+'/rectangle_img.jpg',rectangle_img)
    # cv2.imwrite(path+'/axis_img.jpg',axis_img)
        # print('time in saving 3 images',time.time()-st)
    # st = time.time()
    original_idx = np.array(original_idx)
    gdi_old_way = False
    if len(GDI)==0 :
        # return None,True,None,False
        if not final_attempt:
            return None,True,None,False,False, None
        else:
            best_rectangles, their_idx, GDI, GDI_plus  = select_best_rectangles_gdi_old_way(rectangle_list_all,GDI_calculator_all)
            GDI_calculator = GDI_calculator_all
            print('gdi old way')
            gdi_old_way = True
    else:
        # GDI = normalize_gdi_score(GDI)
        best_rectangles, their_idx = select_best_rectangles(rectangle_list,GDI,GDI_plus,top_rectangles_needed=3,final_attempt=final_attempt)
    
    # print(their_idx)
    best_rectangles, their_idx = height_difference_consideration(best_rectangles, their_idx,darray)
    
    colors = [(255,0,0),(0,255,0),(0,0,255)]
    poses = [(10,20),(10,35),(10,50)]
    for i in range(len(best_rectangles)):
        draw_rectified_rect(img=depth_image, pixel_points=best_rectangles[i],gdi=GDI[their_idx[i]],gdi_plus=GDI_plus[their_idx[i]],color=colors[i],pos=poses[i])
        # cv2.imwrite(path+'/top_3_recs.jpg', depth_image)
        GDI_calculator[their_idx[i]].draw_refined_pose(depth_image)
    # print(their_idx)
    # np.savetxt(path+'/rectangles.txt',np.array(rectangle_list))
    colors = [(255,0,0),(0,255,0),(0,255,255),(255,0,255),(255,255,0),(128,0,255),(0,128,255),(0,255,128),(255,0,128),(128,128,0),(128,0,128),(0,128,128),(0,128,0),(128,0,0),(0,0,128),(10,100,200),(10,200,100),(100,10,200),(100,200,10),(200,100,10)]
    # print(len(rectangle_list_all))
    for i in range(len(rectangle_list_all)):
        draw_rectified_rect_plain(depth_image_copy,rectangle_list_all[i], color=colors[int(i%20)], index = i)
    for i in range(len(rectangle_list_all)):
        draw_rectified_rect_plain(img_copy,rectangle_list_all[i], color=colors[int(i%20)])  

    # np.savetxt(path+'/gdi.txt',np.array(GDI))
    # np.savetxt(path+'/gdi_plus.txt',np.array(GDI_plus))
    # cv2.imwrite(path+'/top_3_recs.jpg', depth_image)
    # cv2.imwrite(path+'/all_poses_depth.jpg', depth_image_copy)
    # cv2.imwrite(path+'/all_poses_rgb.jpg', img_copy)

    best_rectangles = np.array(best_rectangles)
    final_rect_pixel_array = np.array(best_rectangles[0])
    # print(final_rect_pixel_cords)
    angle = final_axis_angle(final_rect_pixel_array)

    valid_flag = True
    boundary_pose = False
    min_depth_difference = 0.03
    if gdi_old_way:
        draw_rectified_rect(img=final_pose_rect_img, pixel_points=best_rectangles[0],gdi=GDI[their_idx[0]],gdi_plus=GDI_plus[their_idx[0]])
        # cv2.imwrite(path+'/final.jpg', final_pose_rect_img)
        cx = int(final_rect_pixel_array[0,0])
        cy = int(final_rect_pixel_array[0,1])
        gripper_opening = param.Max_Gripper_Opening_value
        gripper_closing = 0.1
        valid_flag = False  # set it true in case of grasping only 
    else:
        print('original_idx',original_idx[their_idx])
        draw_rectified_rect(img=final_pose_rect_img, pixel_points=best_rectangles[0])
        new_centroid, new_gripper_opening, object_width = GDI_calculator[their_idx[0]].draw_refined_pose(final_pose_rect_img)
        # cv2.imwrite(path+'/final.jpg', final_pose_rect_img)
        cx = new_centroid[0]
        cy = new_centroid[1]
        gripper_opening = (float(new_gripper_opening)/param.gripper_finger_space_max)*param.Max_Gripper_Opening_value
        if gripper_opening > 1.0:
            gripper_opening = 1.0
        gripper_closing = (float(object_width)/param.gripper_finger_space_max)*param.Max_Gripper_Opening_value
        boundary_pose = GDI_calculator[their_idx[0]].boundary_pose
        min_depth_difference = GDI_calculator[their_idx[0]].min_depth_difference
        # print('new_gripper_opening',new_gripper_opening)

    # print('time searching',time.time()-st)

    # print('time in all many things',time.time()-st)
    
    z = darray[cy][cx]
    # x = (cx - (w/2))*(z/(f_x))
    # y = (cy - (h/2))*(z/(f_y))
    x,y = param.pixel_to_xyz(cx,cy,z)
    print('FOV', x,y,z)
    fov = np.array([x,y,z])
    # st = time.time()
    # cx = (3.2/mw)*cx
    # cy = (2.4/mh)*cy
    # x1,y1,z1 = query_point_cloud_client(cx, cy)
    # # try:
    # #     x,y,z = get_3d_cam_point(np.array([cx, cy])).cam_point
    # # except rospy.ServiceException as e:
    # #     print("Point cloud Service call failed: %s"%e)
    # # print('time searching',time.time()-st)
    # print('PCD', x1,y1,z1)

    
    # target = [x,y,z,angle,gripper_opening,new_gripper_opening]
    # np.savetxt(path+'/target.txt',target,fmt='%f') 
    # np.savetxt(path+'/center.txt',np.array([cx,cy]),fmt='%d')
    # np.savetxt(path+'/manualseed.txt',np.array([manualSeed]),fmt='%d')
    
    # return target,True,np.array([cy,cx]),valid_flag,boundary_pose, min_depth_difference, fov

    return final_pose_rect_img


