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

class GDI2:
    def __init__(self,rotation_point,angle):
        self.rotation_point = rotation_point
        x = rotation_point[0]
        y = rotation_point[1]
        t = angle
        self.rotation_matrix = np.array([[cos(t), -sin(t), -x*cos(t)+y*sin(t)+x], [sin(t), cos(t), -x*sin(t)-y*cos(t)+y], [0,0,1]])
        self.tx = x - int(gripper_height/2)
        self.ty = y - int(gripper_width/2)
        self.bmap = None
        self.dmap = None
        self.new_centroid = np.array([7,35])
        self.gripper_opening = 70
        self.gdi_score_old_way = 0
        self.darray = None
        self.gripper_opening_meter = 0.1
        self.object_width = 0.05
        self.boundary_pose = False
        self.min_depth_difference = 0.03

    def rotate(self,point):
        point_homo = np.ones(3)
        point_homo[0:2] = point
        new_point = np.matmul(self.rotation_matrix,point_homo)
        return int(new_point[0]), int(new_point[1])

    def map_the_point(self,i,j):
        xp = i+self.tx
        yp = j+self.ty
        mapped_loc = self.rotate(np.array([xp,yp]))
        xo,yo = mapped_loc
        # if xo<0:
        #     xo=0
        # elif xo > 199:
        #     xo = 199
        # if yo<0:
        #     yo=0
        # elif yo > 199:
        #     yo = 199
        return xo,yo

    def calculate_gdi_score_old_way(self):
        bmap = self.bmap
        gdi_count = bmap[:,0:cy-THRESHOLD1].sum() + bmap[:,cy+THRESHOLD1:].sum()
        gdi_count_normalized = int(100*gdi_count/gdi_max)
        self.gdi_score_old_way = gdi_count_normalized
        return self.gdi_score_old_way

    def calculate_gdi_plus_score(self):
        bmap = self.bmap
        gdi_plus_count = gdi_plus_max - bmap[:,self.new_centroid[1]-THRESHOLD3:self.new_centroid[1]+THRESHOLD3].sum()
        gdi_plus_count_normalized = int(100*gdi_plus_count/gdi_plus_max)
        if gdi_plus_count_normalized < gdi_plus_cut_threshold:
            return None
        else:
            return gdi_plus_count_normalized

    def calculate_gdi_plus_score_new_way(self):
        cy = self.new_centroid[1]
        s = cy - int(self.gripper_opening/2) + 1
        e = cy + int(self.gripper_opening/2) 
        # print(cy,s,e,self.gripper_opening)
        total_score = 0
        completeness_count = 0
        for y in range(s,e):
            total_score += gripper_width - self.bmap[:,y].sum()
            if self.bmap[:,y].sum() == 0:
                completeness_count += 1

        completeness_score = completeness_count/(e-s)
        avg_score =  float(total_score)/(e-s)
        gdi_plus_count_normalized = int(50*(avg_score/gripper_width)+50*completeness_score)
        # if gdi_plus_count_normalized < 10:
        #     return None
        # else:
        return gdi_plus_count_normalized


    def calculate_pixel_meter_ratio(self,FRs,FLs):
        x,y = self.map_the_point(int((3*FLs+FRs)/4),cx)
        # x2,y2 = self.map_the_point(FRs,cx)
        px = self.rotation_point[0]
        py = self.rotation_point[1]

        # X,Y,_ = query_point_cloud_client(3.2*x, 2.4*y)
        # pX,pY,_ = query_point_cloud_client(3.2*px, 2.4*py)

        z = self.darray[y][x]
        X = (x - (w/2))*(z/(f_x))
        Y = (y - (h/2))*(z/(f_y))
        z = self.darray[py][px]
        pX = (px - (w/2))*(z/(f_x))
        pY = (py - (h/2))*(z/(f_y))
        # # print('pixels',x1,x2,y1,y2,z)
        # # print('meter',X1,X2,Y1,Y2)
        # print(sqrt((x1-x2)**2 + (y1-y2)**2))
        d = sqrt((X-pX)**2 + (Y-pY)**2)
        d_pixel = (FRs-FLs)/4

        meter_to_pixel_ratio = d/d_pixel 

        return meter_to_pixel_ratio

    def calculate_width_in_meter(self,FRs,FLs):
        x1,y1 = self.map_the_point(FLs,cx)
        x2,y2 = self.map_the_point(FRs,cx)
        px = self.rotation_point[0]
        py = self.rotation_point[1]

        # X,Y,_ = query_point_cloud_client(3.2*x, 2.4*y)
        # pX,pY,_ = query_point_cloud_client(3.2*px, 2.4*py)

        z = self.darray[py][px]
        X1 = (x1 - (w/2))*(z/(f_x))
        Y1 = (y1 - (h/2))*(z/(f_y))
        X2 = (x2 - (w/2))*(z/(f_x))
        Y2 = (y2 - (h/2))*(z/(f_y))
        # # print('pixels',x1,x2,y1,y2,z)
        # # print('meter',X1,X2,Y1,Y2)
        # print(sqrt((x1-x2)**2 + (y1-y2)**2))
        d = sqrt((X1-X2)**2 + (Y1-Y2)**2)
        return d




    def pose_refinement(self):
        bmap = self.bmap
        FLs = 0
        FLe = 0
        FRs = gripper_height-1
        FRe = gripper_height-1
        # Free space left
        for i in range(cy-2,-1,-1): # for looping backward
            if gripper_width-bmap[:,i].sum() ==0: #free space
                FLs = i
                break
        for j in range(i-1,-1,-1):
            if gripper_width-bmap[:,j].sum() > 0: #collision space
                FLe = j
                break
        # Free space right
        for i in range(cy,gripper_height): # for looping forward
            if gripper_width-bmap[:,i].sum() ==0: #free space
                FRs = i
                break
        for j in range(i+1,gripper_height):
            if gripper_width-bmap[:,j].sum() > 0: #collision space
                FRe = j
                break
        # print(FLe,FLs,FRs,FRe)
        #check validity
        valid = False
        
        xo,yo = self.map_the_point(cy,cx)
        # meter_to_pixel_ratio = self.calculate_pixel_meter_ratio(FRs,FLs)
        self.object_width = self.calculate_width_in_meter(FLs,FRs) #meter_to_pixel_ratio*(FRs-FLs)
        # print(self.object_width)
        # print('meter_to_pixel_ratio',meter_to_pixel_ratio,'object_width',object_width)
        # gripper_finger_space_max = 0.103*f_x/darray[yo,xo] # Gripper finger space is 0.103 which is in meter, we need space in pixel units
        # print(FLe,FLs,FRs,FRe)
        if (FRe-FRs) > pixel_finger_width and (FLs-FLe) > pixel_finger_width and self.object_width < gripper_finger_space_max: 
            valid = True
        if valid:
            #calculate new pose params
            cy_new = int((FLs+FRs)/2)
            self.new_centroid = np.array([cx,int(cy_new)])

            min_left = self.dmap[:,FLe+2:FLs-1].mean() - np.min([self.dmap[cx,cy],self.dmap[cx,cy_new]])
            min_right = self.dmap[:,FRs+2:FRe-1].mean() - np.min([self.dmap[cx,cy],self.dmap[cx,cy_new]])
            self.min_depth_difference = np.min([min_right,min_left])
            # print('self.min_depth_difference',self.min_depth_difference)
            if self.min_depth_difference < THRESHOLD2:
                valid = False
                return None
            # if self.boundary_pose:
            #     self.gripper_opening = int(2*min(cy_new-(FLe+3*FLs)/4, (3*FRs+FRe)/4-cy_new)) #(FRe-FLe)/gripper_height #fraction of original opening
            # else:
            self.gripper_opening = int(2*min(cy_new-(FLe+FLs)/2, (FRs+FRe)/2-cy_new))
            free_space_score = self.gripper_opening  - (FRs-FLs)
            self.gripper_opening_meter = self.calculate_width_in_meter(cy_new+int(self.gripper_opening/2),cy_new-int(self.gripper_opening/2))
            # free_space_score = int(500*free_space_score/min(FLs,2*cy-FRs))
            # if free_space_score > 500 :
            #     free_space_score = 500
            # print(self.gripper_opening,self.gripper_opening_meter)
            return free_space_score
        else:
            return None

    def draw_refined_pose(self,image):
        xmin = 0
        xmax = gripper_width-1
        ymin = self.new_centroid[1] - int(self.gripper_opening/2)
        ymax = self.new_centroid[1] + int(self.gripper_opening/2)

        point0 = np.array(self.map_the_point(self.new_centroid[1],self.new_centroid[0]))
        point1 = np.array(self.map_the_point(ymax,xmax))
        point2 = np.array(self.map_the_point(ymin,xmax))
        point3 = np.array(self.map_the_point(ymin,xmin))
        point4 = np.array(self.map_the_point(ymax,xmin))
        # print(point1)
        cv2.line(image, (point1[0], point1[1]), (point2[0], point2[1]),
             color=[0,0,0], thickness=2)
        cv2.line(image, (point2[0], point2[1]), (point3[0], point3[1]),
                 color=[0,0,0], thickness=2)
        cv2.line(image, (point3[0], point3[1]), (point4[0], point4[1]),
                 color=[0,0,0], thickness=2)
        cv2.line(image, (point4[0], point4[1]), (point1[0], point1[1]),
                 color=[0,0,0], thickness=2)
        cv2.circle(image, (point0[0], point0[1]), 3,[0,0,0], -1)
        return point0,self.gripper_opening_meter, self.object_width

def point_is_within_image(xo,yo):
    if xo<0 or xo > w-1 or yo<0 or yo > h-1:
        return False
    else:
        return True

def denoise_bmap(bmap, dmap):
    kernel = np.ones((3,3),np.uint8)
    close = cv2.morphologyEx(bmap, cv2.MORPH_CLOSE, kernel)
    close = cv2.medianBlur(bmap,5)
    bmap_denoised = close.astype(np.uint8)
    bmap_diff = bmap_denoised - bmap
    # print('bmap_diff', np.where(bmap_diff == 1))
    dmap[np.where(bmap_diff == 1)] = datum_z
    return bmap_denoised, dmap
    # cv2.imwrite('denoised_' + fn+'.jpg',plt_denoised)

def calculate_GDI2(rectangle,darray,angle):
    gdi2 = GDI2(rectangle[0],angle)
    gdi2.darray = darray
    grasp_pose_dmap_aligned = np.zeros((gripper_width,gripper_height))
    binary_map = np.zeros((gripper_width,gripper_height),np.uint8)
    centroid_depth = darray[rectangle[0][1],rectangle[0][0]]

    #later this can be replaced with paralalization (vectarization)

    boundary_pose_distance = gripper_height
    for i in range(gripper_height):
        for j in range(gripper_width):
            xo,yo = gdi2.map_the_point(i,j)
            if point_is_within_image(xo,yo):
                depth_value = darray[yo,xo]
            else:
                centroid_distance = np.sqrt((xo-rectangle[0][0])**2 + (yo-rectangle[0][1])**2)
                if centroid_distance < boundary_pose_distance:
                    boundary_pose_distance = centroid_distance
                depth_value = datum_z #0
            depth_difference = (depth_value - centroid_depth)
            contact =  depth_difference < THRESHOLD2
            

            #for visualization
            grasp_pose_dmap_aligned[j,i] = depth_value
            binary_map[j,i] = 1-int(contact)

    gdi2.bmap, gdi2.dmap = denoise_bmap(binary_map.copy(),grasp_pose_dmap_aligned.copy())
    gdi_score = gdi2.pose_refinement()
    if gdi2.gripper_opening > 2*boundary_pose_distance:
        gdi2.gripper_opening_meter = gripper_finger_space_max
    gdi_plus_score = gdi2.calculate_gdi_plus_score()
    # gdi_plus_score_new_way = gdi2.calculate_gdi_plus_score_new_way()
    # print('gdi_plus',gdi_plus_score)
    # if gdi_score:
        # gdi_plus_score = gdi2.calculate_gdi_plus_score_new_way()
    # else:
    #     gdi_plus_score = gdi2.calculate_gdi_plus_score()
    # if gdi_score is not None and gdi_plus_score is None:
        # gdi2.calculate_gdi_score_old_way() # In case no valid grasp pose found
        
    # print(gdi2.pose_refinement(binary_map))
    # print('GDI2',grasp_pose_dmap_aligned)
    return grasp_pose_dmap_aligned,binary_map,gdi_score,gdi_plus_score,gdi2, gdi2.bmap, gdi2.dmap

def select_best_rectangles_gdi_old_way(rectangle_list,GDI_calculator_all,top_rectangles_needed=3):
    rectangle_array = np.array(rectangle_list)
    GDI = []
    for gdi2 in GDI_calculator_all:
        GDI.append(gdi2.calculate_gdi_score_old_way())
    GDI_array = np.array(GDI)

    selected_idx = np.argpartition(GDI_array, -top_rectangles_needed)[-top_rectangles_needed:] # indices of top k elements
    selected_idx = selected_idx[np.argsort(GDI_array[selected_idx])[::-1]] # sorting the top k indices

    selected_rectangles = rectangle_array[selected_idx]
    GDI_plus_array = np.zeros(GDI_array.shape)
    return selected_rectangles,selected_idx,GDI_array,GDI_plus_array




mw = 1.6 #3.2
mh = 1.2 #2.4
# if len(sys.argv) > 2:
#     mw = int(sys.argv[1])/200
#     mh = int(sys.argv[2])/200
w = int(mw*200)
h = int(mh*200)
hfov = 69.4 #50 #54.3 # 55.0
vfov = 42.5 #42.69
pi = 3.14
f_x = mw*192 #w/(2*tan(hfov*pi/360))
f_y = mh*256 #h/(2*tan(vfov*pi/360))
# print('focal length',f_x,f_y)
# print('gripper width',2.0*f_x/53.0) # = 8 pixels
# print('gripper coverage in free space',1.5*f_x/53.0) # = 6 pixels
# print('total space',200*53.0/f_x) # = 6 pixels
if len(sys.argv) > 2 :
    THRESHOLD1 = int(sys.argv[1])#20
    THRESHOLD2 = 0.01*int(sys.argv[2])#0.02
    THRESHOLD3 = int(sys.argv[3])
else:
    THRESHOLD1 = int(mh*15)
    THRESHOLD2 = 0.005
    THRESHOLD3 = int(mh*7)

gripper_width = int(mh*20)
gripper_height = int(mh*70)  
gripper_max_opening_length = 0.133
gripper_finger_space_max = 0.103
# gripper_max_free_space = 35
gdi_max = 2*gripper_width*(gripper_height/2-THRESHOLD1)
gdi_plus_max = 2*gripper_width*THRESHOLD3
cx = int(gripper_width/2)
cy = int(gripper_height/2)
pixel_finger_width = mw*8 # width in pixel units.
# GDI_calculator = []
# GDI_calculator_all = []
Max_Gripper_Opening_value = 1.0
datum_z = 0.54 #0.640 # empty bin depth value
gdi_plus_cut_threshold = 20 #70


# rospy.wait_for_service('point_cloud_access_service')
# get_3d_cam_point = rospy.ServiceProxy('point_cloud_access_service', point_cloud_service)


def query_point_cloud_client(x, y):
    rospy.wait_for_service('point_cloud_access_service')
    try:
        get_3d_cam_point = rospy.ServiceProxy('point_cloud_access_service', point_cloud_service)
        resp = get_3d_cam_point(np.array([x, y]))
        return resp.cam_point
    except rospy.ServiceException as e:
        print("Point cloud Service call failed: %s"%e)

def pixel_to_xyz(px,py,darray):
    #cartesian coordinates
    if px < 0:
        px = 0
    if py < 0:
        py = 0
    if px > w-1:
        px = w-1
    if py > h-1:
        py = h-1
    z = darray[py][px]
    x = (px - (w/2))*(z/(f_x))
    y = (py - (h/2))*(z/(f_y))
    return x,y,z

def axis_angle(points):
    major_axis_length = int(mh*150)
    minor_axis_length = int(mh*60)
    cx = np.mean(points[:, 0])
    cy = np.mean(points[:, 1])
    modi_x = np.array(points[:, 0] - cx)
    modi_y = np.array(points[:, 1] - cy)
    num = np.sum(modi_x * modi_y)
    den = np.sum(modi_x ** 2 - modi_y ** 2)
    angle = 0.5 * atan2(2 * num, den)
    [x1_ma, y1_ma] = [int(cx + 0.5 * major_axis_length * cos(angle)),
                      int(cy + 0.5 * major_axis_length * sin(angle))]
    [x2_ma, y2_ma] = [int(cx - 0.5 * major_axis_length * cos(angle)),
                      int(cy - 0.5 * major_axis_length * sin(angle))]
    [x1_mi, y1_mi] = [int(cx + 0.5 * minor_axis_length * cos(angle + radians(90))),
                      int(cy + 0.5 * minor_axis_length * sin(angle + radians(90)))]
    [x2_mi, y2_mi] = [int(cx - 0.5 * minor_axis_length * cos(angle + radians(90))),
                      int(cy - 0.5 * minor_axis_length * sin(angle + radians(90)))]
    axis_dict = {
        "major_axis_points": [(x1_ma, y1_ma), (x2_ma, y2_ma)],
        "minor_axis_points": [(x1_mi, y1_mi), (x2_mi, y2_mi)],
        "angle": angle,
        "centroid": (cx, cy)}
    return axis_dict

def final_axis_angle(points):
    
    cx = np.mean(points[:, 0])
    cy = np.mean(points[:, 1])
    x34 = 0.5*(points[3,0]+points[4,0])
    y34 = 0.5*(points[3,1]+points[4,1])
    if y34-cy == 0.0:
        angle = pi/2
    else:
        angle = atan((cx-x34)/(y34-cy))
    return angle

def keep_angle_bounds(angle):
    if angle > radians(90):
        angle = angle - radians(180)
    elif angle < radians(-90):
        angle = angle + radians(180)
    assert angle >= radians(-90) and angle <= radians(90)
    return angle

def draw_rect(centroid, angle,color=(0, 0, 255),directions=1):
    angle_org = angle
    return_list = []
    angle_list = []
    for i in range(directions):
        if i == 1:
            angle = angle_org + radians(45)
        elif i == 2:
            angle = angle_org - radians(45) 
        elif i == 3:
            angle = angle_org - radians(90)
        angle = keep_angle_bounds(angle) 
        [x1, y1] = [int(centroid[0] - gripper_height * 0.5 * cos(angle) - gripper_width * 0.5 * cos(angle + radians(90))),
                    int(centroid[1] - gripper_height * 0.5 * sin(angle) - gripper_width * 0.5 * sin(angle + radians(90)))]
        [x2, y2] = [int(centroid[0] - gripper_height * 0.5 * cos(angle) + gripper_width * 0.5 * cos(angle + radians(90))),
                    int(centroid[1] - gripper_height * 0.5 * sin(angle) + gripper_width * 0.5 * sin(angle + radians(90)))]
        [x3, y3] = [int(centroid[0] + gripper_height * 0.5 * cos(angle) + gripper_width * 0.5 * cos(angle + radians(90))),
                    int(centroid[1] + gripper_height * 0.5 * sin(angle) + gripper_width * 0.5 * sin(angle + radians(90)))]
        [x4, y4] = [int(centroid[0] + gripper_height * 0.5 * cos(angle) - gripper_width * 0.5 * cos(angle + radians(90))),
                    int(centroid[1] + gripper_height * 0.5 * sin(angle) - gripper_width * 0.5 * sin(angle + radians(90)))]
        angle_list.append(angle)
        return_list.append(np.array([[int(centroid[0]), int(centroid[1])], [x1, y1], [x2, y2], [x3, y3], [x4, y4]]))
    return return_list, angle_list, centroid

def draw_samples(img,pixels):
    try:
        l,_ = pixels.shape
        for i in range(l):
            # x1 = int(pixels[i][0])
            # y1 = int(pixels[i][1])
            cx = int(pixels[i][0])
            cy = int(pixels[i][1])
            # x2 = int(pixels[i][4])
            # y2 = int(pixels[i][5])
            # print(grip_centre[0],grip_centre[1])
            # cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # cv2.circle(img, (x1, y1), 2, (0, 255, 0), -1)
            # cv2.circle(img, (x2, y2), 2, (0, 255, 0), -1)
            cv2.circle(img, (cx, cy), 2, (0, 0, 255), -1)
    except:
        print('no filtered pixels')
    return img

def pose_sample(img, grip_centre, angle):
    gripper_length = 20
    x1 = int(grip_centre[0] - gripper_length * 0.5 * cos(radians(angle)))
    y1 = int(grip_centre[1] - gripper_length * 0.5 * sin(radians(angle)))
    x2 = int(grip_centre[0] + gripper_length * 0.5 * cos(radians(angle)))
    y2 = int(grip_centre[1] + gripper_length * 0.5 * sin(radians(angle)))
    # print(grip_centre[0],grip_centre[1])
    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.circle(img, (x1, y1), 2, (0, 255, 0), -1)
    cv2.circle(img, (x2, y2), 2, (0, 255, 0), -1)
    cv2.circle(img, (int(grip_centre[0]), int(grip_centre[1])), 2, (0, 0, 255), -1)
    points = {"p1": (x1, y1), "p2": (x2, y2), "cp": (int(grip_centre[0]), int(grip_centre[1]))}
    return points

def generate_samples(num_of_samples, img, dmap):
    roi_points = Polygon([(10,10), (10,h-10), (w-10,10), (w-10,h-10)])
    [x_min, y_min, x_max, y_max] = roi_points.bounds
    # print(x_min, y_min, x_max, y_max)
    random_poses_x = np.random.uniform(x_min, x_max, [num_of_samples, ])
    random_poses_y = np.random.uniform(y_min, y_max, [num_of_samples, ])
    full_list = []
    for i in range(len(random_poses_x)):
        # if roi_points.contains(Point(random_poses_x[i], random_poses_y[i])) is True:
            # pose = Pose()
        angle = np.random.uniform(0, 360)
        corner_points = pose_sample(img=img, grip_centre=[random_poses_x[i], random_poses_y[i]], angle=angle)
        p1x = corner_points["p1"][0]
        p1y = corner_points["p1"][1]
        cpx = corner_points["cp"][0]
        cpy = corner_points["cp"][1]
        p2x = corner_points["p2"][0]
        p2y = corner_points["p2"][1]
            # pose.angle = angle
        pose = [p1x,p1y,cpx,cpy,p2x,p2y,angle]
        # if dmap[p1y][p1x] - dmap[cpy][cpx] > THRESHOLD2 and dmap[p1y][p1x] - dmap[cpy][cpx] > THRESHOLD2:
        #     full_list.append(pose)
        full_list.append(pose)

    return img, full_list

def height_difference_consideration(best_rectangles, their_idx,darray):
    if len(their_idx) < 2:
        return best_rectangles, their_idx
    new_best_rectangle = best_rectangles.copy()
    if (darray[best_rectangles[0][0][1],best_rectangles[0][0][0]] - darray[best_rectangles[1][0][1],best_rectangles[1][0][0]]) > 0.1:
        new_best_rectangle[0] = best_rectangles[1]
        new_best_rectangle[1] = best_rectangles[0]
        temp = their_idx[0]
        their_idx[0] = their_idx[1]
        their_idx[1] = their_idx[0]
    elif len(their_idx) > 2 and darray[best_rectangles[0][0][1],best_rectangles[0][0][0]] - darray[best_rectangles[2][0][1],best_rectangles[2][0][0]] > 0.1:
        new_best_rectangle[0] = best_rectangles[2]
        new_best_rectangle[2] = best_rectangles[0]
        temp = their_idx[0]
        their_idx[0] = their_idx[2]
        their_idx[2] = their_idx[0]
    # print('height_difference_consideration', their_idx)
    return new_best_rectangle, their_idx

def select_best_rectangles(rectangle_list,GDI,GDI_plus,top_rectangles_needed=3,final_attempt=False):
    if len(GDI) < top_rectangles_needed:
        top_rectangles_needed = len(GDI)
    rectangle_array = np.array(rectangle_list)
    GDI_array_org = np.array(GDI)
    GDI_plus_array = np.array(GDI_plus)
    # GDI_plus_array = np.zeros(GDI_plus_array.shape)
    GDI_array = GDI_array_org+GDI_plus_array
    # GDI_array = GDI_array_org
    selected_idx = np.argpartition(GDI_array, -top_rectangles_needed)[-top_rectangles_needed:] # indices of top k elements
    selected_idx = selected_idx[np.argsort(GDI_array[selected_idx])[::-1]] # sorting the top k indices
    selected_rectangles = rectangle_array[selected_idx]
    return selected_rectangles,selected_idx

def draw_rectified_rect(img, pixel_points,gdi=None,gdi_plus=0,color=(0, 0, 255),pos=(10,20),gdi_positives=None,gdi_negatives=None,gdi_plus_positives=None,gdi_plus_negatives=None):
    # print(pixel_points)
    pixel_points = np.array(pixel_points,dtype=np.int16)
    cv2.line(img, (pixel_points[1][0], pixel_points[1][1]), (pixel_points[2][0], pixel_points[2][1]),
             color=color, thickness=2)
    cv2.line(img, (pixel_points[2][0], pixel_points[2][1]), (pixel_points[3][0], pixel_points[3][1]),
             color=color, thickness=2)
    cv2.line(img, (pixel_points[3][0], pixel_points[3][1]), (pixel_points[4][0], pixel_points[4][1]),
             color=color, thickness=2)
    cv2.line(img, (pixel_points[4][0], pixel_points[4][1]), (pixel_points[1][0], pixel_points[1][1]),
             color=color, thickness=2)
    cv2.circle(img, (pixel_points[0][0], pixel_points[0][1]), 3,color, -1)
    # mid_2_3 = [int((pixel_points[2][0]+pixel_points[3][0])/2),int((pixel_points[2][1]+pixel_points[3][1])/2)]
    # mid_23_c = [int((pixel_points[0][0]+mid_2_3[0])/2),int((pixel_points[0][1]+mid_2_3[1])/2)]
    if gdi is not None:
        cv2.putText(img, '({0},{1})'.format(gdi,gdi_plus), pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    if gdi_positives is not None:
        for point in gdi_positives:
            cv2.circle(img, (point[0],point[1]),1,(255,255,255), -1)
        for point in gdi_negatives:
            cv2.circle(img, (point[0],point[1]),1,(0,0,0), -1)
        for point in gdi_plus_positives:
            cv2.circle(img, (point[0],point[1]),1,(255,255,0), -1)
        for point in gdi_plus_negatives:
            cv2.circle(img, (point[0],point[1]),1,(0,255,255), -1)
    # cv2.putText(img, name, (pixel_points[0][0], pixel_points[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 255), 3)

def draw_rectified_rect_plain(image_plain, pixel_points,color=(0, 0, 255), index=None):
    # print('plotting here')
    pixel_points = np.array(pixel_points,dtype=np.int16)
    cv2.line(image_plain, (pixel_points[1][0], pixel_points[1][1]), (pixel_points[2][0], pixel_points[2][1]),
             color=color, thickness=1)
    cv2.line(image_plain, (pixel_points[2][0], pixel_points[2][1]), (pixel_points[3][0], pixel_points[3][1]),
             color, thickness=1)
    cv2.line(image_plain, (pixel_points[3][0], pixel_points[3][1]), (pixel_points[4][0], pixel_points[4][1]),
             color=color, thickness=1)
    cv2.line(image_plain, (pixel_points[4][0], pixel_points[4][1]), (pixel_points[1][0], pixel_points[1][1]),
             color=color, thickness=1)
    cv2.circle(image_plain, (pixel_points[0][0], pixel_points[0][1]), 2,color, -1)
    if index is not None:
        cv2.putText(image_plain, str(index), (pixel_points[0][0]+2, pixel_points[0][1]+2), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)
    return image_plain

def send_request_for_pixel_filtering(all_pixels,darray):
    l = len(all_pixels)
    count = 0
    count1 = 0
    filtered_pixels = []
    centers_pixels = []
    for i in range(l):
        x1 = all_pixels[i][0]
        y1 = all_pixels[i][1]
        cx = all_pixels[i][2]
        cy = all_pixels[i][3]
        x2 = all_pixels[i][4]
        y2 = all_pixels[i][5]
        angle = all_pixels[i][6]
        # print(darray[int(cx),int(cy)])
        if datum_z-darray[int(cy),int(cx)] > THRESHOLD2 and darray[int(cy),int(cx)] != 0:
            # print(darray[int(cy),int(cx)])
            count1 = count1 + 1
            # print(abs(darray[int(x1),int(y1)]-darray[int(x2),int(y2)]))
            if abs(darray[int(y1),int(x1)]-darray[int(y2),int(x2)]) < 0.01:
                count = count + 1
                filtered_pixels.append(all_pixels[i])
                centers_pixels.append([cx,cy])
                # pose_sample(img, [cx,cy], angle) # to write the image with grasp regions
                # print(darray[int(cx),int(cy)])
    # print(count1,count)
    return np.array(filtered_pixels),np.array(centers_pixels)

def normalize_gdi_score(gdi):
    gdi = np.array(gdi).astype(np.float)
    gdi = (100*(gdi/gdi.max())).astype(np.int8)
    return gdi

def median_depth_based_filtering(darray,median_depth_map):
    filtered = []
    mask = ((median_depth_map - darray) > THRESHOLD2) &  (darray!=0)
    for i in range(w):
        for j in range(h):
            if mask[j][i] and np.random.random()>0.95:
                filtered.append([i,j])
    return np.array(filtered)

def run_grasp_algo(img,darray,depth_image,clustter_img,centroid_pixels,label,final_attempt=True):

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
    directions = 1
    gripper_width = 20
    if final_attempt:
        directions = 1

    # span = 1
    # for level in range(num_of_clusters,num_of_clusters+span):
    print('shape check',label.shape)
    ids = np.unique(label)
    for k in ids:
        cluster = np.array(centroid_pixels[label.ravel() == k, :], np.int32)
        dict = axis_angle(points=cluster)
        major_points = dict["major_axis_points"]
        minor_points = dict["minor_axis_points"]
        angle = dict["angle"]
        points = cluster

        # cv2.line(axis_img, major_points[0], major_points[1], (255, 0, 0), 2)
        # cv2.line(axis_img, minor_points[0], minor_points[1], (0, 0, 255), 2)
        cv2.line(clustter_img, minor_points[0], minor_points[1], (0, 0, 255), 2)

        color = (np.random.uniform(50, 255), np.random.uniform(50, 250), np.random.uniform(50, 250))

        cv2.rectangle(rectangle_img, (int(np.min(points[:, 0])), int(np.min(points[:, 1]))),
                      (int(np.max(points[:, 0])), int(np.max(points[:, 1]))), color, 3)

        # cv2.circle(clustter_img, (int(centers[k, 0]), int(centers[k, 1])), 2, (255, 0, 0), -1)

        rectangle_pixels_list, angle_list, centroid = draw_rect(centroid=dict["centroid"], angle=angle + radians(90), directions=directions)

        # start_time = time.time()
        if darray[int(centroid[1]),int(centroid[0])]:
            for index,rectangle_pixels in enumerate(rectangle_pixels_list):
                gpd_map,bmap,gdi,gdi_plus,gdi2, bmap_denoised, gpd_map_denoised  = calculate_GDI2(rectangle_pixels,darray,angle_list[index]-radians(180))
                # print(gdi,gdi_plus)
                # print('\n\n\n')
                gpd_map = (gpd_map / gpd_map.max())*255
                gpd_map_denoised_vis = (gpd_map_denoised / gpd_map_denoised.max())*255
                bmap_vis = (bmap / bmap.max())*255
                bmap_vis_denoised = (bmap_denoised / bmap_denoised.max())*255
                # cv2.imwrite(path+'/gpd_map{0}_{1}.jpg'.format(k,index),gpd_map)#.astype(np.uint8))
                # cv2.imwrite(path+'/gpd_map{0}_{1}_denoised.jpg'.format(k,index),gpd_map_denoised_vis)#.astype(np.uint8))
                # cv2.imwrite(path+'/bmap{0}_{1}.jpg'.format(k,index),bmap_vis)#.astype(np.uint8))
                # cv2.imwrite(path+'/bmap{0}_{1}_denoised.jpg'.format(k,index),bmap_vis_denoised)#.astype(np.uint8))

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
                    # cv2.imwrite(path+'/gpose{0}_{1}.jpg'.format(k,index), img_copy)
                else:
                    img_copy = copy.deepcopy(initial_img)
                    draw_rectified_rect(img=img_copy, pixel_points=rectangle_pixels)
                    # cv2.imwrite(path+'/gpose{0}_{1}.jpg'.format(k,index), img_copy)

                gpose_count += 1
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
        GDI = normalize_gdi_score(GDI)
        best_rectangles, their_idx = select_best_rectangles(rectangle_list,GDI,GDI_plus,top_rectangles_needed=3,final_attempt=final_attempt)
    
    # print(their_idx)
    # best_rectangles, their_idx = height_difference_consideration(best_rectangles, their_idx,darray)
    
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
        draw_rectified_rect_plain(depth_image_copy,rectangle_list_all[i], color=colors[int(i/4)], index = i)
    for i in range(len(rectangle_list_all)):
        draw_rectified_rect_plain(img_copy,rectangle_list_all[i], color=colors[int(i/4)])  

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
        gripper_opening = Max_Gripper_Opening_value
        gripper_closing = 0.1
        valid_flag = False  # set it true in case of grasping only 
    else:
        print('original_idx',original_idx[their_idx])
        draw_rectified_rect(img=final_pose_rect_img, pixel_points=best_rectangles[0])
        new_centroid, new_gripper_opening, object_width = GDI_calculator[their_idx[0]].draw_refined_pose(final_pose_rect_img)
        # cv2.imwrite(path+'/final.jpg', final_pose_rect_img)
        cx = new_centroid[0]
        cy = new_centroid[1]
        gripper_opening = (float(new_gripper_opening)/gripper_finger_space_max)*Max_Gripper_Opening_value
        if gripper_opening > 1.0:
            gripper_opening = 1.0
        gripper_closing = (float(object_width)/gripper_finger_space_max)*Max_Gripper_Opening_value
        boundary_pose = GDI_calculator[their_idx[0]].boundary_pose
        min_depth_difference = GDI_calculator[their_idx[0]].min_depth_difference
        # print('new_gripper_opening',new_gripper_opening)

    # print('time searching',time.time()-st)

    # print('time in all many things',time.time()-st)
    
    z = darray[cy][cx]
    x = (cx - (w/2))*(z/(f_x))
    y = (cy - (h/2))*(z/(f_y))
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

    return final_pose_rect_img, clustter_img


def interpolate_noisy_2d_map(map):
    points = np.where(map != 0)
    values = map[points]
    xi = np.where(map == 0)
    map[xi] = griddata(points, values, xi, method='nearest')
    return map

def grasp_planning(mp):

    print('entering gp')
    path = mp.path
    while not mp.image_end or not mp.depth_end:
        time.sleep(0.01)
        continue;

    #saving full resolution version
    image = mp.cur_image
    dmap = mp.cur_dmap.astype(np.float)/1000
    # print('shapes',image.shape,dmap.shape)
    image = cv2.resize(image,(w,h))
    dmap = cv2.resize(dmap,(w,h))
    # print('shapes',image.shape,dmap.shape)
    dmap_vis = (dmap / dmap.max())*255
    np.savetxt(path+'/depth_array.txt',dmap)
    cv2.imwrite(path+'/ref_image.png',image)
    cv2.imwrite(path+'/depth_image.png',dmap_vis)

    #******************************* camera alignment correction here *************************
    inverse_cam_transform = None
    # dmap,image, inverse_cam_transform = self.virtual_camera_transformation(dmap,image)

    # grasp planning
    start_time = time.time()
    total_attempt = 3
    final_attempt = False
    for attempt_num in range(total_attempt):
        if attempt_num == 2:
            final_attempt = True
        st = time.time()
        mp.action,flag,center,valid, boundary_pose, min_depth_difference, mp.fov_points = run_grasp_algo(image.copy(),dmap.copy(),path,final_attempt=final_attempt)
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
        gripper_opening = mp.action[4]
        mp.gripper_closing = mp.gripper_grasp_value

        # code to be optimized 
        # datum_z = 0.575 #0.640
        mp.finger_depth_down = 0.05
        # if mp.action[2] > (datum_z-0.042): #0.540:
        #     mp.finger_depth_down = (datum_z-0.042+mp.finger_depth_down) - mp.action[2]
        

        if mp.finger_depth_down + mp.action[2] > datum_z:
            mp.finger_depth_down = datum_z - mp.action[2]
        if boundary_pose:
            print('boundary_pose',boundary_pose)
            mp.finger_depth_down += -0.004

        print('finger_depth_down:',mp.finger_depth_down,mp.action[2])
        print('min_depth_difference:',min_depth_difference)
        
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
    final_attempt = False

    # dir_path = 'data/'
    dir_path = '../votenet/bin_data/val/'
    sample_path = dir_path+'000025'
    image = cv2.imread(sample_path+'_ref_image.png')
    darray = np.loadtxt(sample_path+'_depth_array.txt')
    depth_image = cv2.imread(sample_path+'_depth_image.png')
    num_obj = int(np.load(sample_path+'_num_objects.npy'))
    path = 'log'
    # image = cv2.imread(path+'/ref_image.png')
    # darray = np.loadtxt(path+'/depth_array.txt')
    # darray = interpolate_noisy_2d_map(darray)
    # start_time = time.time()
    # run_grasp_algo(img,darray,case=case,final_attempt=final_attempt)
    # print('time:', time.time()-start_time)

    action,flag,center,valid, boundary_pose, min_depth_difference, fov_points = run_grasp_algo(image.copy(),darray.copy(),depth_image.copy(),num_obj,path,final_attempt=final_attempt)
    print('min_depth_difference',min_depth_difference)
