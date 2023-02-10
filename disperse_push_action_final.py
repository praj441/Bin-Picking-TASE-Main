from __future__ import division
from __future__ import print_function
import future        # pip install future
import builtins      # pip install future
import past          # pip install future
import six           # pip install six

# skimage==0.14.5
# opencv-python==4.2.0.32
# pip install enum


import os
from os.path import dirname, join, abspath,split
import copy
import numpy as np
import numpy.ma as ma
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as tf
from scipy.ndimage import distance_transform_edt

from skimage.draw.draw import line

from shapely.geometry import LineString
from shapely.geometry import box

from shapely.ops import transform 
from shapely.ops import substring,split

# import pyrtools as prt

from enum import Enum
# from scene.logging.logging import Logger

u'''

'''



# class GripperPush(Enum):
DEPTH_THRES = 0.02
N_PX_BEHIND = 0

PUSH_TILL = 0.6 # Normalized

# PX_GAP_THRES = 20 #px
    
# class Table(Enum):
HEIGHT = 0.55 #0.686666


FLOOR_THRES = 0.02
FLOOR_BIN_THRES = 0.02
PX_GAP_THRES = 6 #px
CLOSE_ENOUGH = 1
DEPTH_MIS_VALUES_CLIP=0.4 


class DispersePushVector(object):

    def __init__(self,rgb,depth,start_r_c):
        
        self.disperse_rgb_img = rgb
        self.disperse_depth_img = depth
        self.start_r_c_in_px = start_r_c.astype(np.int)

        self.disperse_binary_img = None
        self.distance_transform_map = None
        self.distance_transform_img = None

        self.free_point_r_c_in_px = None
        self.end_r_c_in_px = None
        self.new_start_r_c_in_px = None

    def bin_binary_map(self,empty_bin_depth,depth,bin_z_wrt_cam):

        # Modify Depth Map : Make Bin Wall Depth in depth img equal To Floor depth
        depth_map_filled_bin_wall_floor = np.abs(np.subtract(empty_bin_depth,depth))
        depth_map_empty_bin_floor = np.abs(np.subtract(empty_bin_depth,np.full_like(empty_bin_depth,bin_z_wrt_cam)))
        masked_filled_bin_wall_floor = ma.masked_where(depth_map_filled_bin_wall_floor <= FLOOR_BIN_THRES,depth_map_filled_bin_wall_floor)
        masked_empty_bin_floor = ma.masked_where(depth_map_empty_bin_floor <= FLOOR_THRES,depth_map_empty_bin_floor)

        depth[masked_filled_bin_wall_floor.mask] = bin_z_wrt_cam

        # Used for Free point Calculation
        # disperse_binary_mask = ~np.logical_or(~masked_filled_bin_wall_floor.mask,~masked_empty_bin_floor.mask)
        disperse_binary_mask = masked_filled_bin_wall_floor.mask
        self.disperse_binary_img = disperse_binary_mask
        # self.depth = depth



    def construct_object_occupied_and_task_boundary_constrained_space(self):
        
        self.disperse_binary_img[:,0]=0
        self.disperse_binary_img[:,-1]=0
        self.disperse_binary_img[0,:]=0
        self.disperse_binary_img[-1,:]=0

        top_left_limit = np.unique(self.disperse_binary_img,return_index=True)
        top_left_limit = np.unravel_index(top_left_limit[1][1],self.disperse_depth_img.shape)
        
        bottom_right_limit = np.unique(np.fliplr(np.flipud(self.disperse_binary_img)),return_index=True)
        bottom_right_limit = np.subtract(list(self.disperse_depth_img.shape),np.unravel_index(bottom_right_limit[1][1],self.disperse_depth_img.shape))

        a,b = top_left_limit[::-1]
        c,d = bottom_right_limit[::-1]
        self.image_workspace_bounds_in_px = box(a,b,c,d)


    def compute_push_vector(self):

        self.construct_object_occupied_and_task_boundary_constrained_space() 
        self.distance_transform_map = distance_transform_edt(self.disperse_binary_img)

        index_= np.array(np.unravel_index((-self.distance_transform_map.flatten()).argsort(axis=0)[:5],self.distance_transform_map.shape),dtype=np.int).T
        # [x,y]= [c,r] = [u,v] index_

        self.free_point_r_c_in_px = index_[0] # fliped [r,c]
        self.end_r_c_in_px = self.free_point_r_c_in_px

        # debug
        self.distance_transform_img = np.repeat(self.distance_transform_map[:, :, np.newaxis], 3, axis=2)
        rt,ct = np.where(self.distance_transform_img[:,:,0]==1)
        self.distance_transform_img[rt,ct,:] = 255

        # Our Policy Start
        new_start_r_c_in_px, new_start_depth_z_in_px,push_flag,break_point_container_arr = self.get_new_start_point_in_px_along_given_vector(self.start_r_c_in_px,self.end_r_c_in_px)
        new_end_r_c_in_px, new_end_depth_z_in_px= self.end_r_c_in_px, new_start_depth_z_in_px

        self.new_start_r_c_in_px = new_start_r_c_in_px

        self.draw(self.disperse_rgb_img,break_point_container_arr)

        return new_start_r_c_in_px+new_start_depth_z_in_px  ,  new_end_r_c_in_px+new_end_depth_z_in_px , push_flag


    
    def get_new_start_point_in_px_along_given_vector(self,start,end):

        direc_vec = (np.subtract(start[:2],end[:2]))
        norm_direc_vec = direc_vec/np.linalg.norm(direc_vec)

        extreme_start = norm_direc_vec * 1000 + start[:2]
        extreme_start =  extreme_start.astype(np.int)

        tmp_direc_start_vector = LineString([start[:2][::-1],extreme_start[:2][::-1]])

        bound_clip = self.image_workspace_bounds_in_px.boundary

        pt = tmp_direc_start_vector.intersection(bound_clip)
        
        if not pt.is_empty:
            pt = pt.centroid
        else:
            pt = tmp_direc_start_vector.interpolate(0.01,normalized=True)

        r_,c_ = int(pt.y),int(pt.x)
        r_,c_ = np.clip([r_,c_],0,self.disperse_depth_img.shape[0]-2)

        # unique point (px) satisfying defined depth start threshold
        rr,cc = line(int(start[0]),int(start[1]),r_,c_)
        self.distance_transform_img[rr,cc,:] = (0,0,1)
        rr_,cc_ = line(int(start[0]),int(start[1]),end[0],end[1])
        self.distance_transform_img[rr_,cc_,:] = (1,0,0)


        depth_for_points_behind_given_start = self.disperse_depth_img[rr,cc]
        depth_difference_wrt_given_start  = depth_for_points_behind_given_start - depth_for_points_behind_given_start[0]

        bool_vector_satisfying_constrained = depth_difference_wrt_given_start > DEPTH_THRES
        
        neg_sign = np.sign(depth_difference_wrt_given_start)<0
        
        if neg_sign.all():
            print('\ndepth_thres_too_high\n')
        

        break_point_container = []
        if bool_vector_satisfying_constrained.any() and (not neg_sign.all()):

            in_ = np.where(bool_vector_satisfying_constrained==True)[0].tolist()


            f = np.subtract(in_[1:],in_[:-1])
            f = np.insert(f,0,1)

            if np.any(f):
                    
                k = np.where(np.abs(f)<=CLOSE_ENOUGH,1,0)
                g = np.subtract(k[1:],k[:-1])
                g = np.insert(g,0,1)

                # all_bp = np.where(abs(g)!=0)
                # all_indx_ = np.asarray(in_)[all_bp[0]]

                flag_ = True
                push_flag = False
                span_e = 0

                while flag_:

                    unq = np.unique(g,return_index=True)

                    if unq[0][-1]==1:
                        satis = unq[1][-1]

                    if unq[0][0]==-1:
                        breaks = unq[1][0]


                    if np.all(g==0):
                        
                        span_s = span_e
                        span_e = len(g)-1
                        tmp_idx = int((span_s+2*span_e)/3)   
                        tmp_idx= np.clip(tmp_idx,None,int(len(in_)))

                        if abs(span_e-span_s)>=PX_GAP_THRES:

                            print('free_span of length {} greater than asserted_threshold {} '.format(abs(span_e-span_s),PX_GAP_THRES))
                            push_flag = True

                            # print('true : span_length_px :',abs(span_e-span_s))
                        else:
                            # print('false : span_length_px :',abs(span_e-span_s))
                            print('span_length_px :',abs(span_e-span_s))
                            push_flag = False


                        break_point_container += [in_[span_s],in_[span_e],push_flag]
                        


                        indx_ = in_[tmp_idx]
                        flag_=False
                        break 


                    if np.all(g[span_e:][1:]==0) and g[span_e:][0]==1:
                        
                        span_s = span_e
                        span_e = len(g)-1
                        tmp_idx = int((span_s+2*span_e)/3)    #+PX_GAP_THRES+4
                        tmp_idx= np.clip(tmp_idx,None,int(len(in_)))

                        if abs(span_e-span_s)>=PX_GAP_THRES:

                            # print('true : span_length_px :',abs(span_e-span_s))
                            print('free_span of length {} greater than asserted_threshold {} '.format(abs(span_e-span_s),PX_GAP_THRES))
                            push_flag = True

                        else:
                            print('span_length_px :',abs(span_e-span_s))
                            push_flag = False


    
                        # print('span_length_px :',abs(span_e-span_s))
                       
                        break_point_container += [in_[span_s],in_[span_e],push_flag]


                        indx_ = in_[tmp_idx]
                        flag_=False
                        break

                    # breaks obtained
                    # gap = abs(satis-breaks)
                    span = [satis,breaks]
                    span_s = min(span)
                    span_e = max(span)-1
                    gap = abs(span_e-span_s)
                    print('span_length_px :',gap)
                    
                    if gap>=PX_GAP_THRES:
                    
                        print('free_span of length {} greater than asserted_threshold {} '.format(gap,PX_GAP_THRES))
                        
                        # ind_s = in_[span_s]
                        # ind_e = in_[span_e]

                        tmp_idx = int((span_s+2*span_e)/3) 
                        tmp_idx= np.clip(tmp_idx,None,int(len(in_)))
                        indx_ = in_[tmp_idx]

                        flag_=False
                        push_flag = True
                        break_point_container += [in_[span_s],in_[span_e],push_flag]


                        break

                    else:

                        g[:span_e] = 0
                        print('continue... searching for next span greater than asserted_threshold {} '.format(PX_GAP_THRES))
                        push_flag = False

                    
                    
            else:
                print('none points satisfying gap threshold so skip....push action\n ')
                
                print(' func still return random half length with failed flag as False  ')

                indx_= int(len(rr)/2)
                push_flag = False
                

                

            #......actual point .....px plane
            #     
            new_start_r_c_in_px = [rr[indx_],cc[indx_]]



        else:

            # print('none points satisfying ')
            print('none points satisfying depth threshold.........for end effector to nudge down safely behind object \n ')

            print(' func still return random half length with failed flag as False  ')

            indx_= int(len(rr)/2)
            push_flag = False


            new_start_r_c_in_px = [rr[indx_],cc[indx_]]

        

        print('point row,col : in px plane ( {} , {} ) '.format(new_start_r_c_in_px[0],new_start_r_c_in_px[1]))

        new_start_depth_z_in_px = depth_for_points_behind_given_start[0] - DEPTH_THRES

        print('point z : depth {} in meters for r,c point ( {} , {} ) '.format(new_start_depth_z_in_px,new_start_r_c_in_px[0],new_start_r_c_in_px[1]))

        print('r,c,Z,push_flag',(new_start_r_c_in_px, new_start_depth_z_in_px , push_flag) )

        if any(break_point_container):
            # break_point_container_arr = break_point_container
            
            break_point_container = np.array(break_point_container)
            if break_point_container.shape[0] == break_point_container.size:
                break_point_container = np.reshape(break_point_container,(-1,3))
                

            break_point_container_arr = [ [rr[indx_],cc[indx_]] for indx_ in  break_point_container[:,:2].flatten() ] 

            print("half_pt {} extremes {}".format(new_start_r_c_in_px,break_point_container_arr))

        else:

            break_point_container_arr = None


        return new_start_r_c_in_px, new_start_depth_z_in_px , push_flag,break_point_container_arr




    def draw(self,rgb,break_pt_arr=None):

        self.draw_rgb_img = copy.deepcopy(rgb)

        # base
        self.draw_rgb_img = np.float32(self.draw_rgb_img[:,:,::-1])

        p1,p2 = self.new_start_r_c_in_px[::-1],self.end_r_c_in_px[::-1]
        cv2.arrowedLine(self.draw_rgb_img, tuple(p1),tuple(p2) ,(255,255,255,0.5), 2) 
        # p1,p2 = self.new_start_r_c_in_px[::-1],self.start_r_c_in_px[::-1]
        # cv2.arrowedLine(self.draw_rgb_img, tuple(p1),tuple(p2),(0,0,255), 2) 
        # cv2.drawContours(self.draw_rgb_img,[arr],0,(255,0,0),3)
        c = self.start_r_c_in_px[::-1]
        cv2.circle(self.draw_rgb_img,tuple(c),5,(0,255,0),1)
        c = self.new_start_r_c_in_px[::-1]
        cv2.circle(self.draw_rgb_img,tuple(c),2,(0,0,255),-1)

        p = self.new_start_r_c_in_px[::-1]
        cv2.putText(self.draw_rgb_img,u' S',tuple(p),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),2,cv2.LINE_4)
        p = self.end_r_c_in_px[::-1]
        cv2.putText(self.draw_rgb_img,u' E',tuple(p),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),2,cv2.LINE_4)
        
        if break_pt_arr is not None:
        
            for c in break_pt_arr:
                xy_  = c[::-1] # r,c --> y,x
                cv2.circle(self.draw_rgb_img,tuple(xy_),3,(255,0,0),-1)
                    
        # base
        self.draw_rgb_img = (self.draw_rgb_img[:,:,::-1]).astype(np.uint8)


def disperse_task(rgb,depth,empty_bin_depth,start_r_c,output_path,bin_z_wrt_cam=HEIGHT):

    disperse_action = DispersePushVector(rgb,depth,start_r_c)
    disperse_action.bin_binary_map(empty_bin_depth,depth,bin_z_wrt_cam=bin_z_wrt_cam)
    start,end,push_flag = disperse_action.compute_push_vector()

    imgs = [disperse_action.disperse_depth_img,disperse_action.disperse_binary_img,disperse_action.distance_transform_map,disperse_action.distance_transform_img,disperse_action.draw_rgb_img]
    fg, axs = plt.subplots(1, 5, figsize=(16, 4))
    axs = axs.flatten()
    for img, ax in zip(imgs, axs):  
        ax.imshow(img)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
    # plt.show()
    plt.savefig(output_path+'/disperse_output.png')
    cv2.imwrite(output_path+'/disperse_push_vector.png',disperse_action.draw_rgb_img)
    # cv2.imwrite(output_path+'/disperse_depth_img.png',disperse_action.disperse_depth_img)
    # cv2.imwrite(output_path+'/disperse_binary_img.png',disperse_action.disperse_binary_img)
    cv2.imwrite(output_path+'/distance_transform_map.png',disperse_action.distance_transform_map)
    cv2.imwrite(output_path+'/distance_transform_img.png',disperse_action.distance_transform_img)
    cv2.imwrite(output_path+'/push_vector.png',disperse_action.draw_rgb_img)

    # prt.imshow([disperse_action.disperse_depth_img,disperse_action.disperse_binary_img,disperse_action.distance_transform_map,disperse_action.distance_transform_img,disperse_action.draw_rgb_img])
    # plt.show()

    print

    # start and end points in column, raw format
    print(start,end)
    return np.array([start[1], start[0]]), np.array([end[1], end[0]])


# disperse_task(np.zeros((200,200,3)),np.random.random((200,200)),np.random.random((200,200)),
#                         np.asarray([72,142]),0.55)


    
