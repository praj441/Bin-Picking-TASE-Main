#!/usr/bin/env python3

import rospy, sys, numpy as np
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

processing = False
processing1 = False
processing2 = False

pcd_end = False
img_end = False
new_msg = False
msg = None
start = True
start_pose_set = True
end = False
error_list = []
setting_list = []
plan = None
cam_wpos = np.zeros(6,)

import time
import threading
cur_image = None


DUMMY_FIELD_PREFIX = '__'
# mappings between PointField types and numpy types
type_mappings = [(PointField.INT8, np.dtype('int8')), (PointField.UINT8, np.dtype('uint8')),
                 (PointField.INT16, np.dtype('int16')),
                 (PointField.UINT16, np.dtype('uint16')), (PointField.INT32, np.dtype('int32')),
                 (PointField.UINT32, np.dtype('uint32')),
                 (PointField.FLOAT32, np.dtype('float32')), (PointField.FLOAT64, np.dtype('float64'))]
pftype_to_nptype = dict(type_mappings)
nptype_to_pftype = dict((nptype, pftype) for pftype, nptype in type_mappings)

# sizes (in bytes) of PointField types
pftype_sizes = {PointField.INT8: 1, PointField.UINT8: 1, PointField.INT16: 2, PointField.UINT16: 2,
                PointField.INT32: 4, PointField.UINT32: 4, PointField.FLOAT32: 4, PointField.FLOAT64: 8}

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

class Camera:
    def __init__(self):
        rospy.init_node("camera", anonymous=False)
        rospy.loginfo("hi, is this the init ?: yes")
        # self.video = cv2.VideoWriter("sample.avi", cv2.cv.CV_FOURCC('M','J','P','G'), 10.0, (640,480), True)

        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        self.depth_sub = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.depth_callback)
        self.point_sub = rospy.Subscriber('/camera/depth_registered/points',PointCloud2,self.point_cloud_callback)
        #np.savetxt('source_pose.txt',pose, delimiter=',',fmt='%.5f')
        self.main()
        print('init_end')



    def main(self):
        global processing2, pcd_end, img_end

        while not pcd_end or not img_end:
            time.sleep(0.1)

        bgr = cv2.imread('result_dir/current_image.png')

        pc_arr_past = np.load('result_dir/point_cloud_2d_array.npy')
        cond = np.isfinite(pc_arr_past) #& np.isfinite(pc_arr_past[:,:,1]) & np.isfinite(pc_arr_past[:,:,2])
        pc_arr_past = np.where(cond,pc_arr_past,0.0)
        # pc_arr_past['y'] = np.where(cond,pc_arr_past['y'],0.0)
        # pc_arr_past['z'] = np.where(cond,pc_arr_past['z'],0.0)
        c = input('bhai jao aur ek object hatao aur vapas aaker 1 number enter karo!')
        processing2 = False

        pcd_end = False
        while not pcd_end:
            time.sleep(0.1)

        pc_arr_cur = np.load('result_dir/point_cloud_2d_array.npy')

        cond = np.isfinite(pc_arr_cur) #& np.isfinite(pc_arr_past[:,:,1]) & np.isfinite(pc_arr_past[:,:,2])
        pc_arr_cur = np.where(cond,pc_arr_cur,0.0)

        pc_distance = np.linalg.norm(pc_arr_past-pc_arr_cur,axis=2)
        print('pc_distance',pc_distance)
        print(np.histogram(pc_distance,bins=4))

        slice_mask = (pc_distance > 0.030) & (pc_distance < 0.1)
        no_mask = pc_distance > -0.5

        pc_arr_sliced = pc_arr_past[slice_mask]
        bgr_sliced = bgr[slice_mask]
        pc_distance = pc_distance[slice_mask]

        pc_arr = pc_arr_past[no_mask]
        bgr = bgr[no_mask]
        
        print('pc_arr_sliced',pc_arr_sliced.shape)

        write_ply_monocolor(pc_arr_sliced, 'result_dir/sliced.ply', text=True) #,color=bgr_sliced
        write_ply(pc_arr, 'result_dir/no_slice.ply',color=bgr, text=True)
        # write_ply_distance(pc_arr, 'result_dir/sliced_distance.ply',color=pc_distance, text=True)
        

    def image_callback(self,data):
        global cam_wpos, cur_image, img_end
        global processing
        if not processing:
            try:
                processing = True
                cur_image = cv_bridge.CvBridge().imgmsg_to_cv2(data, "rgb8")
                cur_image_bgr = cv_bridge.CvBridge().imgmsg_to_cv2(data, "bgr8")
                np.savetxt('result_dir/cam_wpos.txt',cam_wpos,delimiter=',',fmt='%.5f')
                cv2.imwrite('result_dir/current_image.png',cur_image_bgr)
                #cv2.imshow(cur_image,1)
                print('ref_image.png written and cam_wpos.txt written')
                # processing1 = False
            except cv_bridge.CvBridgeError as e:
                print(e)
            img_end = True

    def depth_callback(self,data):
        global cam_wpos
        global processing1
        if not processing1:
            try:
                processing1 = True
                cur_image = cv_bridge.CvBridge().imgmsg_to_cv2(data, desired_encoding="passthrough")
                # np.savetxt('result_dir/cam_wpos.txt',cam_wpos,delimiter=',',fmt='%.5f')
                cv2.imwrite('result_dir/depth_image.png',cur_image)
                np.save('result_dir/current_depth_map.npy',cur_image.astype(np.float64)/1000)
                #cv2.imshow(cur_image,1)
                print('depth_image.png written')
            except cv_bridge.CvBridgeError as e:
                print(e)

    def point_cloud_callback(self,data):
        h = data.height
        w = data.width
        global processing2, pcd_end
        # try:
        # pcl_data = self.ros_to_pcl(data)
        if not processing2:
            processing2 = True
            np_arr = self.pointcloud2_to_array(data)
            print('shape',np_arr.shape)
            np.save('result_dir/point_cloud_2d_array.npy',np_arr)
            print('*************its really working*********')

            # point_arr, color_arr = self.get_xyz_points(np_arr, remove_nans=True)
            # print('sizes',point_arr.shape, color_arr.shape)
            # np.save('result_dir/point_cloud_array.npy',point_arr)
            # np.save('result_dir/color_cloud_array.npy',color_arr)

            pcd_end = True
           


    def ros_to_pcl(self,ros_cloud):
        """ Converts a ROS PointCloud2 message to a pcl PointXYZRGB

            Args:
                ros_cloud (PointCloud2): ROS PointCloud2 message

            Returns:
                pcl.PointCloud_PointXYZRGB: PCL XYZRGB point cloud
        """
        points_list = []

        for data in pc2.read_points(ros_cloud, skip_nans=True):
            points_list.append([data[0], data[1], data[2], data[3]])

        pcl_data = pcl.PointCloud_PointXYZRGB()
        pcl_data.from_list(points_list)

        return pcl_data 


    def pointcloud2_to_dtype(self,cloud_msg):
        """Convert a list of PointFields to a numpy record datatype.
        """
        offset = 0
        np_dtype_list = []
        for f in cloud_msg.fields:
            while offset < f.offset:
                # might be extra padding between fields
                np_dtype_list.append(('%s%d' % (DUMMY_FIELD_PREFIX, offset), np.uint8))
                offset += 1
            np_dtype_list.append((f.name, pftype_to_nptype[f.datatype]))
            offset += pftype_sizes[f.datatype]

        # might be extra padding between points
        while offset < cloud_msg.point_step:
            np_dtype_list.append(('%s%d' % (DUMMY_FIELD_PREFIX, offset), np.uint8))
            offset += 1
            
        return np_dtype_list

    def pointcloud2_to_array(self,cloud_msg, split_rgb=False):
        """
        Converts a rospy PointCloud2 message to a numpy recordarray
        
        Reshapes the returned array to have shape (height, width), even if the height is 1.

        The reason for using np.fromstring rather than struct.unpack is speed... especially
        for large point clouds, this will be <much> faster.
        """
        # construct a numpy record type equivalent to the point type of this cloud
        dtype_list = self.pointcloud2_to_dtype(cloud_msg)
        # print(dtype_list)

        # parse the cloud into an array
        cloud_arr = np.fromstring(cloud_msg.data, dtype_list)

        # remove the dummy fields that were added
        cloud_arr = cloud_arr[
            [fname for fname, _type in dtype_list if not (fname[:len(DUMMY_FIELD_PREFIX)] == DUMMY_FIELD_PREFIX)]]

        
        cloud_arr =  np.reshape(cloud_arr, (cloud_msg.height, cloud_msg.width))
        np_arr = np.zeros((cloud_msg.height, cloud_msg.width,3))
        np_arr[:,:,0] = cloud_arr['x']
        np_arr[:,:,1] = cloud_arr['y']
        np_arr[:,:,2] = cloud_arr['z']

        return np_arr

    def get_xyz_points(self,cloud_array, remove_nans=True, dtype=np.float32):
        global cur_image
        """Pulls out x, y, and z columns from the cloud recordarray, and returns
        a 3xN matrix.
        """
        # remove crap points
        print('before',cloud_array.shape, cur_image.shape)
        if remove_nans:
            mask = np.isfinite(cloud_array['x']) & np.isfinite(cloud_array['y']) & np.isfinite(cloud_array['z']) #& np.isfinite(cloud_array['rgb'])
            cloud_array = cloud_array[mask]
            color_masked = cur_image[mask]
            # zero_array = np.zeros(cloud_array.shape)
            # cloud_array = np.where(mask == True, cloud_array, zero_array)
        print('after',cloud_array.shape,color_masked.shape)
        # pull out x, y, and z values
        points = np.zeros(list(cloud_array.shape) + [3], dtype=dtype)
        points[..., 0] = cloud_array['x']
        points[..., 1] = cloud_array['y']
        points[..., 2] = cloud_array['z']

        # rgb_arr = cloud_array['rgb'].copy()
        # rgb_arr.dtype = np.uint32
        # r = np.asarray((rgb_arr >> 16) & 255, dtype=np.uint8)
        # g = np.asarray((rgb_arr >> 8) & 255, dtype=np.uint8)
        # b = np.asarray(rgb_arr & 255, dtype=np.uint8)
        # print('one',r[0:5])
        # print('two',color_masked[0:5,0])

        # points[..., 3] = r
        # points[..., 4] = g
        # points[..., 5] = b

        return points, color_masked




mp=Camera()
rospy.loginfo("hi, is this the start")
rospy.spin()
