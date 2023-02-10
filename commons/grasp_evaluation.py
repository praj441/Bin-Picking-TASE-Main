import numpy as np
from math import *
import cv2

class GDI2:
    def __init__(self,rotation_point,angle,darray,param):
        self.rotation_point = rotation_point
        x = rotation_point[0]
        y = rotation_point[1]
        t = angle
        self.rotation_matrix = np.array([[cos(t), -sin(t), -x*cos(t)+y*sin(t)+x], [sin(t), cos(t), -x*sin(t)-y*cos(t)+y], [0,0,1]])
        self.tx = x - int(param.gripper_height/2)
        self.ty = y - int(param.gripper_width/2)
        self.bmap_vis_denoised = None
        self.bmap_ws = None
        self.bmap = None
        self.dmap = None
        self.smap = None
        self.pmap = None
        self.new_centroid = np.array([7,35])
        self.gripper_opening = param.gripper_height
        self.gdi_score_old_way = 0
        self.gripper_opening_meter = 0.1
        self.object_width = 0.05
        self.boundary_pose = False
        self.min_depth_difference = 0.03
        self.laplacian = None
        self.param = param
        self.darray = darray
        self.FLS_score = None # aka gdi_score
        self.CRS_score = None # aka gdi_plus_score
        self.final_center = None
        self.surface_normal_score = None
        self.cone_detection = False
        self.invalid_reason = 'NA'
        self.invalid_id = 0
        self.final_image = None
        self.lg = None
        self.rg = None

    def rotate(self,point):
        point_homo = np.ones(3)
        point_homo[0:2] = point
        new_point = np.matmul(self.rotation_matrix,point_homo)
        return int(new_point[0]), int(new_point[1])


    def map_the_point_vectorize(self,I):
        I[0] = I[0] + self.tx
        I[1] = I[1] + self.ty

        # rotation part
        I = np.concatenate((I,np.ones((1,I.shape[1])))) # homofie 2xN -> 3xN
        O = np.matmul(self.rotation_matrix,I).astype(np.int32)[0:2,:]
        return O

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
        cy = self.param.cy
        bmap = self.bmap
        gdi_count = bmap[:,0:cy-self.param.THRESHOLD1].sum() + bmap[:,cy+self.param.THRESHOLD1:].sum()
        gdi_count_normalized = int(100*gdi_count/self.param.gdi_max)
        self.gdi_score_old_way = gdi_count_normalized
        return self.gdi_score_old_way

    def calculate_gdi_plus_score(self):
        bmap = self.bmap
        gdi_plus_count = self.param.gdi_plus_max - bmap[:,self.new_centroid[1]-self.param.THRESHOLD3:self.new_centroid[1]+self.param.THRESHOLD3].sum()
        gdi_plus_count_normalized = int(100*gdi_plus_count/self.param.gdi_plus_max)
        if gdi_plus_count_normalized < self.param.gdi_plus_cut_threshold:
            return None
        else:
            return gdi_plus_count_normalized

    def calculate_gdi_plus_score_better_way(self):
        bmap = self.bmap
        gdi_plus_count_upper = self.param.gdi_plus_max - bmap[:self.new_centroid[0],self.new_centroid[1]-self.param.THRESHOLD3:self.new_centroid[1]+self.param.THRESHOLD3].sum()
        gdi_plus_count_lower = self.param.gdi_plus_max - bmap[self.new_centroid[0]:,self.new_centroid[1]-self.param.THRESHOLD3:self.new_centroid[1]+self.param.THRESHOLD3].sum()
        gdi_plus_count = min(int(gdi_plus_count_upper),int(gdi_plus_count_lower))
        gdi_plus_count_normalized = int(100*gdi_plus_count/self.param.gdi_plus_max)
        
        if gdi_plus_count_normalized < self.param.gdi_plus_cut_threshold:
            # print('rejected',gdi_plus_count_normalized,gdi_plus_count_upper,gdi_plus_count_lower)
            return None
        else:
            # print('selected',gdi_plus_count_normalized,gdi_plus_count_upper,gdi_plus_count_lower)
            self.CRS_score = gdi_plus_count_normalized
            return gdi_plus_count_normalized


    # def calculate_gdi_plus_score_new_way(self):
    #     cy = self.new_centroid[1]
    #     s = cy - int(self.gripper_opening/2) + 1
    #     e = cy + int(self.gripper_opening/2) 
    #     # print(cy,s,e,self.gripper_opening)
    #     total_score = 0
    #     completeness_count = 04 (FLs-FLe) 20 self.object_width 0.1300949046355436

    #     for y in range(s,e):
    #         total_score += gripper_width - self.bmap[:,y].sum()
    #         if self.bmap[:,y].sum() == 0:
    #             completeness_count += 1

    #     completeness_score = completeness_count/(e-s)
    #     avg_score =  float(total_score)/(e-s)
    #     gdi_plus_count_normalized = int(50*(avg_score/gripper_width)+50*completeness_score)
    #     # if gdi_plus_count_normalized < 10:
    #     #     return None
    #     # else:
    #     return gdi_plus_count_normalized


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
        cx = self.param.cx
        
        x1,y1 = self.map_the_point(FLs,cx)
        x2,y2 = self.map_the_point(FRs,cx)
        px = self.rotation_point[0]
        py = self.rotation_point[1]

        # X,Y,_ = query_point_cloud_client(3.2*x, 2.4*y)
        # pX,pY,_ = query_point_cloud_client(3.2*px, 2.4*py)

        z = self.param.datum_z
        X1,Y1 = self.param.pixel_to_xyz(x1,y1,z)
        X2,Y2 = self.param.pixel_to_xyz(x2,y2,z)
        # # print('pixels',x1,x2,y1,y2,z)
        # # print('meter',X1,X2,Y1,Y2)
        # print(sqrt((x1-x2)**2 + (y1-y2)**2))
        d = sqrt((X1-X2)**2 + (Y1-Y2)**2)
        return d




    def pose_refinement(self):
        bmap = self.bmap
        cy = self.param.cy
        cx = self.param.cx
        FLs = 0
        FLe = 0
        FRs = self.param.gripper_height-1
        FRe = self.param.gripper_height-1
        # Free space left
        for i in range(cy-2,-1,-1): # for looping backward
            if self.param.gripper_width-bmap[:,i].sum() ==0: #free space
                FLs = i
                break
        for j in range(i-1,-1,-1):
            if self.param.gripper_width-bmap[:,j].sum() > 0: #collision space
                FLe = j
                break
        # Free space right
        for i in range(cy,self.param.gripper_height): # for looping forward
            if self.param.gripper_width-bmap[:,i].sum() ==0: #free space
                FRs = i
                break
        for j in range(i+1,self.param.gripper_height):
            if self.param.gripper_width-bmap[:,j].sum() > 0: #collision space
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
        # self.laplacian = cv2.Laplacian(bmap,cv2.CV_64F)
        # self.laplacian = (self.laplacian/self.laplacian.max())*255
        if (FRe-FRs) > self.param.pixel_finger_width and (FLs-FLe) > self.param.pixel_finger_width and self.object_width < self.param.gripper_finger_space_max: 
            valid = True

        else:
            # print(self.param.pixel_finger_width,self.param.gripper_finger_space_max,'(FRe-FRs)',(FRe-FRs),'(FLs-FLe)',(FLs-FLe),'self.object_width',self.object_width)
            self.invalid_reason = 'large object or less free space'
            self.invalid_id = 1

        cy_new = int((FLs+FRs)/2)

        # cone detection code here 
        
            # sobelx = cv2.Sobel(self.dmap,cv2.CV_64F,1,0,ksize=-1)
            # self.laplacian = cv2.Sobel(sobelx,cv2.CV_64F,1,0,ksize=-1)
        

        if self.cone_detection: #and valid:

            sobely = cv2.Sobel(bmap,cv2.CV_64F,0,1,ksize=-1)
            left_gradient = sobely[int(cx/2):int(3*cx/2),FLs:cy_new]
            right_gradient = sobely[int(cx/2):int(3*cx/2),cy_new:FRs]
            lg = left_gradient.sum()
            rg = right_gradient.sum()
            self.lg = lg
            self.rg = rg
             # surface normal score
            avg_gradients = (abs(lg) + abs(rg))/2
            score = 150 - avg_gradients
            if score < 0:
                score = 0
            elif score > 100:
                score = 100
            self.surface_normal_score = score

            if abs(lg) > self.param.cone_thrs and abs(rg) > self.param.cone_thrs and lg*rg > 0:
                print('cone shape',lg,rg)
                self.invalid_reason = 'cone-shape_{0}_{1}'.format(lg,rg)
                if self.invalid_id == 0:
                    self.invalid_id = 2
                elif self.invalid_id == 1:
                    self.invalid_id = 6
                valid = False
            else:
                print('not a cone shape',lg,rg)

           
        # else:
        #     print('cone_detection not loaded')


        # check for in-between poses
        if valid and self.smap is not None:
            gw = self.param.gripper_width
            contact_region_seg_mask = self.smap[2:gw-2,FLs+4:FRs-4]
            contact_region_seg_mask = contact_region_seg_mask[contact_region_seg_mask>5]
            if np.count_nonzero(np.unique(contact_region_seg_mask)) > 1:
                print('********* in between pose detected **************')
                print('loc',self.map_the_point(cy_new,cx))
                self.invalid_id = 4
                valid = False
        # else:
        #     valid = True 

        
                                  
    
        #calculate new pose params
        
        self.new_centroid = np.array([cx,int(cy_new)])

        # I forgot, why I added below code
        # min_left = self.dmap[:,FLe+2:FLs-1].mean() - np.min([self.dmap[cx,cy],self.dmap[cx,cy_new]])
        # min_right = self.dmap[:,FRs+2:FRe-1].mean() - np.min([self.dmap[cx,cy],self.dmap[cx,cy_new]])
        # self.min_depth_difference = np.min([min_right,min_left])
        # if self.min_depth_difference < self.param.THRESHOLD2:
        #     valid = False
        #     return None


        # if self.boundary_pose:
        #     self.gripper_opening = int(2*min(cy_new-(FLe+3*FLs)/4, (3*FRs+FRe)/4-cy_new)) #(FRe-FLe)/gripper_height #fraction of original opening
        # else:
        min_gripper_opening = int(2*min(cy_new-(FLe+FLs)/2, (FRs+FRe)/2-cy_new))
        self.gripper_opening = min_gripper_opening #int(2*min(cy_new-(FLe+self.param.pixel_finger_width/2), (FRe-self.param.pixel_finger_width/2)-cy_new))

        free_space_score = min_gripper_opening  - (FRs-FLs)
        self.gripper_opening_meter = self.calculate_width_in_meter(cy_new+int(self.gripper_opening/2),cy_new-int(self.gripper_opening/2))
        # free_space_score = int(500*free_space_score/min(FLs,2*cy-FRs))
        # if free_space_score > 500 :
        #     free_space_score = 500
        # print(self.gripper_opening,self.gripper_opening_meter)
        free_space_score_normalized = 100*(float(free_space_score)/self.param.gdi_max)
        if valid:
            self.FLS_score = free_space_score_normalized
            return free_space_score_normalized
        else: 
            return None

    def draw_refined_pose(self,image,path=None,scale=1, thickness = 2):
        xmin = 0
        xmax = self.param.gripper_width-1
        ymin = self.new_centroid[1] - int(self.gripper_opening/2)
        ymax = self.new_centroid[1] + int(self.gripper_opening/2)

        point0 = scale*np.array(self.map_the_point(self.new_centroid[1],self.new_centroid[0]))
        point1 = scale*np.array(self.map_the_point(ymax,xmax))
        point2 = scale*np.array(self.map_the_point(ymin,xmax))
        point3 = scale*np.array(self.map_the_point(ymin,xmin))
        point4 = scale*np.array(self.map_the_point(ymax,xmin))
        # refined_pose = np.concatenate(point)
        # print(point1)
        # color1 = (255,255,0) # cyan
        # # color = (0,160,255) # orange
        # color = (178, 83, 70) # Liberty
        color = (17,233,135) # green
        color = (69,24,255)
        color1 = (25,202,242)
        # color1 = (255,80,0) 
        
        cv2.line(image, (point1[0], point1[1]), (point2[0], point2[1]),
             color=color1, thickness=thickness)
        cv2.line(image, (point2[0], point2[1]), (point3[0], point3[1]),
                 color=color, thickness=thickness)
        cv2.line(image, (point3[0], point3[1]), (point4[0], point4[1]),
                 color=color1, thickness=thickness)
        cv2.line(image, (point4[0], point4[1]), (point1[0], point1[1]),
                 color=color, thickness=thickness)
        cv2.circle(image, (point0[0], point0[1]), thickness,color, -1)
        if path is not None:
            cv2.imwrite(path, image)
        self.final_center = point0/scale
        return self.final_center,self.gripper_opening_meter, self.object_width

    def point_is_within_image(self,xo,yo):
        w = self.param.w
        h = self.param.h
        if xo<0 or xo > w-1 or yo<0 or yo > h-1:
            return False
        else:
            return True

    def denoise_bmap(self,bmap, dmap):
        # kernel = np.ones((3,3),np.uint8)
        # close = cv2.morphologyEx(bmap, cv2.MORPH_CLOSE, kernel)
        close = cv2.medianBlur(bmap,5)
        bmap_denoised = close.astype(np.uint8)
        bmap_diff = bmap_denoised - bmap
        # print('bmap_diff', np.where(bmap_diff == 1))
        dmap[np.where(bmap_diff == 1)] = self.param.datum_z
        return bmap_denoised, dmap
        # cv2.imwrite('denoised_' + fn+'.jpg',plt_denoised)

    def bound_the_point(self,xc,yc):
        w = self.param.w
        h = self.param.h
        if xc<0:
            xc = 0
        if xc > w-1:
            xc = w-1
        if yc<0:
            yc = 0
        if yc > h-1:
            yc = h-1
        return xc,yc

def calculate_GDI2(inputs,rectangle,angle):
    import time
    st = time.time()
    darray = inputs['darray']
    param = inputs['param']
    try:
        seg_mask = inputs['seg_mask']
        smap = np.zeros((param.gripper_width,param.gripper_height),np.uint8)
    except:
        seg_mask = None
        smap = None
        # print('************ seg_mask not available ******************')
    try:
        pc_arr = inputs['pc_arr']
        pmap = np.zeros((param.gripper_width,param.gripper_height,3),np.float32)
    except:
        pc_arr = None
        pmap = None
        # print('************ pc_arr not available ******************')
    gdi2 = GDI2(rectangle[0],angle,darray,param)
    dmap = np.zeros((param.gripper_width,param.gripper_height))
    binary_map = np.zeros((param.gripper_width,param.gripper_height),np.uint8)
    
    compr_depth = darray[rectangle[0][1],rectangle[0][0]]

    #later this can be replaced with paralalization (vectarization)

    boundary_pose_distance = param.gripper_height
    
    gw = param.gripper_width
    gh = param.gripper_height
    cx = param.cx
    cy = param.cy

    for j in range(gw):
        for i in range(gh):
            xo,yo = gdi2.map_the_point(i,j)
            if gdi2.point_is_within_image(xo,yo):
                depth_value = darray[yo,xo]
                if seg_mask is not None:
                    smap[j,i] = seg_mask[yo,xo]
                if pc_arr is not None:
                    pmap[j,i] = pc_arr[yo,xo]
            else:
                # centroid_distance = np.sqrt((xo-rectangle[0][0])**2 + (yo-rectangle[0][1])**2)
                # if centroid_distance < boundary_pose_distance:
                #     boundary_pose_distance = centroid_distance
                depth_value = 0.0
                if pc_arr is not None:
                    pmap[j,i] = pc_arr[rectangle[0][1],rectangle[0][0]]
            dmap[j,i] = depth_value
            
    diff_map = dmap-compr_depth

    # dh_map is differentiation of diff_map
    # dh_map is for detection of continuous slanted surface
    dh_map = np.ones(diff_map.shape,np.bool)

    dh1 = diff_map[:,cy:] - diff_map[:,cy-1:gh-1]
    dh2 = diff_map[:,:cy-1] - diff_map[:,1:cy]
    dh_map[:,cy-1] = dh_map[:,cy] 
    dh1 = dh1 > 0.005 
    dh2 = dh2 > 0.005
    for i in range(1,dh1.shape[1]):
        dh1[:,i] = dh1[:,i] | dh1[:,i-1]
    for i in range(cy-3,-1,-1):
        dh2[:,i] = dh2[:,i] | dh2[:,i+1]
    dh_map[:,cy:] = dh1
    dh_map[:,:cy-1] = dh2
    dh_map[:,cy-1] = dh_map[:,cy]

    gdi2.bmap_ws = (diff_map > param.THRESHOLD2)
    try:
        if not inputs['slanted_pose_detection']:
            # print('slanted_pose_detection not loaded')
            binary_map = gdi2.bmap_ws
            gdi2.bmap_ws = gdi2.bmap_ws & dh_map
        else:
            # print('slanted_pose_detection loaded')
            binary_map = gdi2.bmap_ws & dh_map
    except:
        # print('slanted_pose_detection loaded')
        binary_map = gdi2.bmap_ws & dh_map
    binary_map = binary_map.astype(np.uint8)

    try:
        if not inputs['cone_detection']:
            gdi2.cone_detection = False
    except:
        # print('cone detection loaded')
        gdi2.cone_detection = True

    # for i in range(param.gripper_height):
    #     for j in range(param.gripper_width):
    #         if 1-binary_map[j,i] and i-2 >=0 and i+2 < param.gripper_height and j-2 >=0 and j+2 < param.gripper_width:
    #             print('contact',1-binary_map[j,i],'cendroid_depth',compr_depth)
    #             print(grasp_pose_dmap_aligned[j-2:j+2,i-2:i+2])
    # print('grasp_pose_dmap_aligned',grasp_pose_dmap_aligned)
    gdi2.bmap, gdi2.dmap = gdi2.denoise_bmap(binary_map.copy(),dmap.copy())
    gdi2.smap = smap
    gdi2.pmap = pmap
    bmap_vis = (binary_map / binary_map.max())*255
    bmap_vis_denoised = (gdi2.bmap / gdi2.bmap.max())*255
    gdi2.bmap_ws = (gdi2.bmap_ws / gdi2.bmap_ws.max())*255
    gdi2.bmap_vis_denoised = bmap_vis_denoised

    
    


    gdi_score = gdi2.pose_refinement()
    # if gdi2.gripper_opening > 2*boundary_pose_distance:
    #     gdi2.gripper_opening_meter = gripper_finger_space_max
    gdi_plus_score = gdi2.calculate_gdi_plus_score_better_way()
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
    # gpd_map = (grasp_pose_dmap_aligned / grasp_pose_dmap_aligned.max())*255
    # gpd_map_denoised_vis = (gdi2.dmap / gdi2.dmap.max())*255
    i = gdi2.new_centroid[0]
    j = gdi2.new_centroid[1]
    xc,yc = gdi2.map_the_point(j,i)
    xc,yc = gdi2.bound_the_point(xc,yc)

    print('time in processing 1 sample',time.time()-st)

    return bmap_vis,gdi_score,gdi_plus_score,gdi2, bmap_vis_denoised,xc,yc


def calculate_GDI2_Lite(inputs,rectangle,angle,vectorization=True):
    import time
    st = time.time()
    darray = inputs['darray']
    param = inputs['param']
    
    gdi2 = GDI2(rectangle[0],angle,darray,param)
    dmap = np.zeros((param.gripper_width,param.gripper_height))
    binary_map = np.zeros((param.gripper_width,param.gripper_height),np.uint8)
    
    compr_depth = darray[rectangle[0][1],rectangle[0][0]]

    #later this can be replaced with paralalization (vectarization)

    boundary_pose_distance = param.gripper_height

    gw = param.gripper_width
    gh = param.gripper_height

    if not vectorization:
        # st = time.time()
        for j in range(gw):
            for i in range(gh):
                xo,yo = gdi2.map_the_point(i,j)
                # print(i,j,xo,yo)
                if gdi2.point_is_within_image(xo,yo):
                    depth_value = darray[yo,xo]
                else:
                    depth_value = 0.0
                dmap[j,i] = depth_value
        # print('time in dmap',time.time()-st)
    else:
    
        # st = time.time()
        # vectorization
        Imap = np.mgrid[0:gh:1,0:gw:1].reshape(2,-1)
        Omap = gdi2.map_the_point_vectorize(Imap) # 2xN
        
        # filter for points within image boundaries
        w = param.w
        h = param.h

        within_points_filter = (Omap[0,:] < 0) + (Omap[0,:] > w-1) + (Omap[1,:] < 0) + (Omap[1,:] > h-1)
        Omap[0] = np.where(within_points_filter,0.0,Omap[0])
        Omap[1] = np.where(within_points_filter,0.0,Omap[1])
        dmap = np.where(within_points_filter,0.0,darray[Omap[1],Omap[0]])
        dmap = dmap.reshape(gh,gw).T
        # print('time in dmap1',time.time()-st)

    diff_map = dmap-compr_depth
    gdi2.bmap_ws = (diff_map > param.THRESHOLD2)
    binary_map = gdi2.bmap_ws
    binary_map = binary_map.astype(np.uint8)

    # visualize
    # ri = np.random.randint(1,1000)
    # v0 = 255*dmap/dmap.max()
    # v1 = 255*dmap1/dmap1.max()
    # cv2.imwrite('temp/{0}_dmap.png'.format(ri),v0)
    # cv2.imwrite('temp/{0}_dmap1.png'.format(ri),v1)

    
    
    gdi2.bmap, gdi2.dmap = gdi2.denoise_bmap(binary_map.copy(),dmap.copy())



    gdi_score = gdi2.pose_refinement()
    gdi_plus_score = gdi2.calculate_gdi_plus_score_better_way()
    

    return None,gdi_score,gdi_plus_score,gdi2, None,None,None