def grasp_planning(mp):
	print('entering gp')
	path = mp.path
	while not mp.image_end or not mp.depth_end:
		time.sleep(0.01)
		continue;

	#saving full resolution version
	image = mp.cur_image
	dmap = mp.cur_dmap.astype(np.float)/1000
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
		mp.action,flag,center,valid, boundary_pose, min_depth_difference, pcd = run_grasp_algo(image.copy(),dmap.copy(),path,final_attempt=final_attempt)
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
		datum_z = 0.575 #0.640
		mp.finger_depth_down = 0.03
		if mp.action[2] > (datum_z-0.042): #0.540:
			mp.finger_depth_down = (datum_z-0.042+mp.finger_depth_down) - mp.action[2]
		if boundary_pose:
			print('boundary_pose',boundary_pose)
			mp.finger_depth_down += -0.004

		print('finger_depth_down:',mp.finger_depth_down)
		print('min_depth_difference:',min_depth_difference)
		
		# if finger_depth_down > min_depth_difference :
		#     finger_depth_down = min_depth_difference
		# self.gripper.run(self.gripper_homing_value) #*gripper_opening+5)        

		# if inverse_cam_transform is not None:
		#     a[0:3] = self.affine_transformation(inverse_cam_transform,a[0:3])
		
		print('time taken by the algo:{0}'.format(time.time()-start_time))                                                                                                                                                                                                                            
		print('exiting gp')