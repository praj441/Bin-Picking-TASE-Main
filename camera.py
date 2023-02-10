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

class Camera:
    def __init__(self,path):
        rospy.init_node("camera", anonymous=False)
        rospy.loginfo("hi, is this the init ?: yes")
        # self.video = cv2.VideoWriter("sample.avi", cv2.cv.CV_FOURCC('M','J','P','G'), 10.0, (640,480), True)
        self.image_start = True
        self.image_end = False
        self.depth_start = True
        self.depth_end = False
        self.pc_start = True
        self.pc_end = False

        self.cur_depth_map = None
        self.cur_image = None
        self.cur_pc = None

        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        self.depth_sub = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.depth_callback)
        self.point_sub = rospy.Subscriber('/camera/depth_registered/points',PointCloud2,self.point_cloud_callback)
        #np.savetxt('source_pose.txt',pose, delimiter=',',fmt='%.5f')
        # self.main()
        self.path = path

        print('init_end')

    # def main():

    def click_a_depth_sample(self):
        self.depth_start = False
        while not self.depth_end:
            print('waiting for depth click...')
            time.sleep(0.05)
        self.depth_end = False
        print('depth click done!!')

    def click_an_image_sample(self):
        self.image_start = False
        while not self.image_end:
            print('waiting for image click...')
            time.sleep(0.05)
        self.image_end = False
        print('image click done!!')

    def click_a_pcd_sample(self):
        self.pc_start = False
        while not self.pc_end:
            # print('waiting for pcd click...')
            time.sleep(0.05)
        self.pc_end = False
        print('pcd click done!!')

    def click_a_camera_sample(self):
        self.cur_image = None
        self.depth_start = False
        self.image_start = False
        self.pc_start = False
        while not self.pc_end or not self.depth_end or not self.image_end:
            print('waiting for a camera click...')
            time.sleep(0.05)
        self.depth_end = False
        self.pc_end = False
        self.image_end = False
        print('camera click done!!')

    def image_callback(self,data):
        global cam_wpos, cur_image
        # global image_start, image_end
        if not self.image_start:
            try:
                self.image_start = True
                self.cur_image = cv_bridge.CvBridge().imgmsg_to_cv2(data, "bgr8")
                # cur_image_bgr = cv_bridge.CvBridge().imgmsg_to_cv2(data, "bgr8")
                # np.savetxt(self.path + '/cam_wpos.txt',cam_wpos,delimiter=',',fmt='%.5f')
                # cv2.imwrite(self.path + '/current_image.png',self.cur_image)
                #cv2.imshow(cur_image,1)
                # print('image click done')
                # processing1 = False
                self.image_end = True
                print('image shape',self.cur_image.shape)
            except cv_bridge.CvBridgeError as e:
                print(e)

    def depth_callback(self,data):
        global cam_wpos
        # global depth_start, depth_end
        if not self.depth_start:
            try:
                self.depth_start = True
                self.cur_depth_map = cv_bridge.CvBridge().imgmsg_to_cv2(data, desired_encoding="passthrough")
                self.cur_depth_map = self.cur_depth_map.astype(np.float64)/1000 
                # np.savetxt('result_dir/cam_wpos.txt',cam_wpos,delimiter=',',fmt='%.5f')
                
                #cv2.imshow(cur_image,1)
                # print(self.path + '/depth_image.png written')
                # print('depth click done')
                self.depth_end = True
            except cv_bridge.CvBridgeError as e:
                print(e)

    def point_cloud_callback(self,data):
        h = data.height
        w = data.width
        # try:
        # pcl_data = self.ros_to_pcl(data)
        if not self.pc_start:
            self.pc_start = True
            np_arr = self.pointcloud2_to_array(data)
            print('shape',np_arr.shape)
            point_arr, flatten_point_cloud, color_arr = self.get_xyz_points(np_arr, remove_nans=True)
            self.cur_pc = point_arr
            # point = np_arr[0][0]
            # point = np.array([point[0], point[1], point[2]])
            # print(np_arr.shape,point)
            # np_arr_resized = np.resize((w,h,3))
            print('sizes',point_arr.shape, color_arr.shape)
            # np.save(self.path + '/point_cloud_array.npy',point_arr)
            # np.save(self.path + '/color_cloud_array.npy',color_arr)
            # np.save('result_dir/point_cloud_array_resized.npy',np_arr_resized)
            # pcl.save(pcl_data,'result_dir/scene.pcd')
            # except:
            #     print('error in point_callback')
            self.pc_end = True
            # print('pc_click done')


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

        
        return np.reshape(cloud_arr, (cloud_msg.height, cloud_msg.width))

    def get_xyz_points(self,cloud_array, remove_nans=True, dtype=np.float32):
        # while not self.image_end:
        #     time.sleep(0.05)
        """Pulls out x, y, and z columns from the cloud recordarray, and returns
        a 3xN matrix.
        """

        while self.cur_image is None:
             time.sleep(0.05)
        # remove crap points
        print('before',cloud_array.shape, self.cur_image.shape)

        h,w = cloud_array.shape
        map_points = np.zeros((h,w,3))
        map_points[:,:, 0] = cloud_array['x']
        map_points[:,:, 1] = cloud_array['y']
        map_points[:,:, 2] = cloud_array['z']


        if remove_nans:
            mask = np.isfinite(cloud_array['x']) & np.isfinite(cloud_array['y']) & np.isfinite(cloud_array['z']) #& np.isfinite(cloud_array['rgb'])
            cloud_array = cloud_array[mask]
            color_masked = self.cur_image[mask]
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

        return map_points, points, color_masked


if __name__ == "__main__":

    mp=Camera('result_dir')
    rospy.loginfo("hi, is this the start")
    rospy.spin()
