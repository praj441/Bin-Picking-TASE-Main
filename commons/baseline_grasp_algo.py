import numpy as np
import sys
import cv2
import copy
import time
from utils_gs import draw_samples
from utils_gs import create_directory
from utils_gs import final_axis_angle
from utils_gs import draw_rectified_rect
from grasp_evaluation import calculate_GDI2
from custom_grasp_planning_algorithm_dense import select_a_best_grasp_pose
from utils_gs import remove_outlier

def remove_outliers_cluster_wise(points,labels):
    ids = np.unique(labels)
    output_points = []
    output_labels = []
    for i in ids:
        cluster_mask = (labels==i)
        cluster = points[cluster_mask]
        cluster_labels = labels[cluster_mask]
        cluster, filter_mask = remove_outlier(cluster)
        output_points.extend(cluster.tolist())
        output_labels.extend(cluster_labels[filter_mask].tolist())
    return np.array(output_points), np.array(output_labels)

def run_grasp_algo(inputs):
   
    

    img = inputs['image']
    darray = inputs['darray']
    depth_image = inputs['depth_image']
    param = inputs['param'] 
    final_attempt = inputs['final_attempt']
    path = inputs['dump_dir']
    median_depth_map = inputs['median_depth_map']

    adaptive_clusters = True
    try:
        if not inputs['adaptive_clusters']:
            adaptive_clusters = False
    except:
            adaptive_clusters = True

    if path is not None:
        create_directory(path+'/bmaps')
        create_directory(path+'/directions')
        create_directory(path+'/grasp_pose_info')

    new_img = copy.deepcopy(img)
    clustter_img= copy.deepcopy(img)
    final_pose_rect_img = copy.deepcopy(img)
    depth_image_copy = copy.deepcopy(depth_image)
    img_copy = copy.deepcopy(img)
    initial_img = copy.deepcopy(img)

    centroid_pixels_3D, objectness_ratio = param.median_depth_based_filtering(darray,median_depth_map,0.90)
    centroid_pixels = centroid_pixels_3D[:,0:2]
    if path is not None:
        filtered_img = draw_samples(new_img,centroid_pixels)
        cv2.imwrite(path+'/filtered_pixels.jpg',filtered_img)
    print('centroid_pixels',centroid_pixels.shape)
    # print('filtered pixels',len(filtered_pixels),float(len(filtered_pixels))/max_samples)
    # objectness_ratio = float(len(filtered_pixels))/max_samples
    print('objectness_ratio',objectness_ratio)
    # if objectness_ratio > 0.25:
    #     num_of_clusters = 20
    # elif objectness_ratio > 0.050:
    #     num_of_clusters = 15
    # # elif objectness_ratio > 0.025:
    # #     num_of_clusters = 10
    # else:
    #     num_of_clusters = 10
    if adaptive_clusters:
        num_of_clusters = int(135*np.power(objectness_ratio,1.75))
        if num_of_clusters < 10:
            num_of_clusters = 10
    else:
        num_of_clusters = 10
    print('num_of_clusters',num_of_clusters)

    # num_of_clusters = 10
    
    centroid_pixels = np.float64(centroid_pixels)
    
    # K-Means Clustering #
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1500, 0.001)

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

    span = 1
    for level in range(num_of_clusters,num_of_clusters+span):
        
        # try:
            # start_time = time.time()
            # distances_of_centers_from_datapoints, label, centers = cv2.kmeans(centroid_pixels_3D, K=level,
            #                                                               criteria=criteria,
            #                                                               attempts=6,
            #                                                               flags=cv2.KMEANS_RANDOM_CENTERS,bestLabels=None)
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=num_of_clusters,n_init=6,max_iter=1500)
        kmeans.fit(centroid_pixels_3D)
        label = kmeans.labels_
        centers_3d = kmeans.cluster_centers_

        centers = centers_3d[:,0:2]
        centroid_pixels,label = remove_outliers_cluster_wise(centroid_pixels,label)
            # print('time in clustering algo:',time.time()-start_time)
        # except Exception as e:
        #     print('Error in try block:',e)
        #     return [],False,(0,0),None, False,None,None,None
        if path is not None:
            # clustter_img = copy.deepcopy(clustter_img_orig)
            for i in range(len(centroid_pixels)):
                if np.random.random()>0.9:
                    green_part = int((label[i]*1)%255)
                    blue_part = int((label[i]*90)%255)
                    red_part = int((label[i]*213)%255)
                    cv2.circle(clustter_img, (int(centroid_pixels[i, 0]), int(centroid_pixels[i, 1])), 2, (blue_part,green_part,red_part), -1)
            ids = np.unique(label)
            for i in ids:
                mask = (label==i)
                contour = np.hstack((centroid_pixels[:,0][mask].astype(np.int32)[:, np.newaxis], centroid_pixels[:,1][mask].astype(np.int32)[:, np.newaxis]))
                convexHull = cv2.convexHull(contour)
                cv2.drawContours(clustter_img, [convexHull], -1, (255,0,0), 2)

        gpose_count = 0
        directions = 4
        gripper_width = 20


        angles = []
        for k in range(len(centers)):
            cluster = np.array(centroid_pixels[label.ravel() == k, :], np.int32)
            dict = param.axis_angle(points=cluster)
            minor_points = dict["minor_axis_points"]
            angle = dict["angle"]
            points = cluster
            angles.append(angle)
        angles = np.array(angles)
        inputs['top_grasp_points'] = centers
        inputs['angles'] = angles

    #****************** the main function *******************     
    grasp_pose_info = select_a_best_grasp_pose(inputs)

    final_rect_pixel_array = grasp_pose_info['final_pose_rectangle']
    gdi2 = grasp_pose_info['gdi_calculator']
    original_idx = grasp_pose_info['selected_idx']
    gdi_old_way = grasp_pose_info['gdi_old_way'] 

    angle = final_axis_angle(final_rect_pixel_array)
    
    grasp_score = 0.0
    valid_flag = False
    if not gdi_old_way:
        valid_flag = True
        print('original_idx',original_idx)
        final_pose_rect_img = 0.8*final_pose_rect_img
        new_centroid, new_gripper_opening, object_width = gdi2.draw_refined_pose(final_pose_rect_img, thickness=4)
        # if path is not None:
        gdi2.draw_refined_pose(depth_image_copy)
        # cv2.imwrite(dump_dir+'/depth_image.png',depth_image_copy)
        grasp_score = (gdi2.FLS_score + gdi2.CRS_score)/2

        cx = new_centroid[0]
        cy = new_centroid[1]
        gripper_opening = (float(new_gripper_opening)/param.gripper_finger_space_max)*param.Max_Gripper_Opening_value
        if gripper_opening > 1.0:
            gripper_opening = 1.0

    else:
        cx = final_rect_pixel_array[0][0]
        cy = final_rect_pixel_array[0][1]
        gripper_opening = 1.0
        # if path is not None:
        draw_rectified_rect(img=final_pose_rect_img, pixel_points=final_rect_pixel_array)

    if path is not None:
        cv2.imwrite(path+'/final.jpg', final_pose_rect_img)
        cv2.imwrite(path+'/bmap.jpg',grasp_pose_info['bmap'])
        # cv2.imwrite(path+'/bmap_ws.jpg',grasp_pose_info['bmap_ws'])
        np.savetxt(path+'/their_idx.txt',[original_idx])
        # np.savetxt(path+'/gdi.txt',np.array(gdi2.GDI))
        # np.savetxt(path+'/gdi_plus.txt',np.array(gdi2.GDI_plus))
    # cv2.imwrite(path+'/top_3_recs.jpg', depth_image)
    # cv2.imwrite(path+'/all_poses_depth.jpg', depth_image_copy)
    # cv2.imwrite(path+'/all_poses_rgb.jpg', img_copy)

    # print('cx cy', cx, cy)
    z = darray[cy][cx]
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

    
    target = [x,y,z,angle,gripper_opening,grasp_score]#,new_gripper_opening]
    grasp = [cx,cy,angle,gripper_opening]
    if path is not None:
        np.savetxt(path+'/target.txt',target,fmt='%f') 
        np.savetxt(path+'/center.txt',np.array([cx,cy]),fmt='%d')
    # np.savetxt(path+'/manualseed.txt',np.array([manualSeed]),fmt='%d')
    
    boundary_pose = False
    min_depth_difference = 0.02

    # outputs = {'grasp_score':grasp_score}

    return target,True,np.array([cy,cx]),valid_flag,boundary_pose, min_depth_difference,clustter_img, final_pose_rect_img, grasp
