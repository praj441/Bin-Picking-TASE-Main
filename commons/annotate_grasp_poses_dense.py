import matplotlib.pyplot as plt
import matplotlib.image as mpimg




scene = 0
data_path = '../simulation/temp'



ref_img = mpimg.imread(data_path+'/{0:06d}_ref_image.png'.format(scene))
fig = plt.figure()
plt.rcParams['figure.dpi'] = 200
plt.rcParams['savefig.dpi'] = 200
plt.ion()
fig.show()
for i in range(2048):
	for j in range(4):
		gpose = i
		index = j

		# pose_img = mpimg.imread(data_path+'/{0}/poses/{1}_{2}.png'.format(scene,gpose,index))
		# bmap = mpimg.imread(data_path+'/{0}/bmaps/bmap{1}_{2}_denoised.jpg'.format(scene,gpose,index))
		# bmap_ws = mpimg.imread(data_path+'/{0}/bmaps/bmap{1}_{2}_ws.jpg'.format(scene,gpose,index))

		
		with open(data_path+'/grasp_pose_info/invalid_reason_{0}_{1}.txt'.format(gpose,index).format(scene)) as f:
			invalid_reason = f.readlines()[0]

		print(i,j,invalid_reason)
		if 'cone-shape' not in invalid_reason:
			continue

		pose_img = mpimg.imread(data_path+'/poses/{0}_{1}.png'.format(gpose,index))
		bmap = mpimg.imread(data_path+'/bmaps/bmap{0}_{1}.jpg'.format(gpose,index))
		bmap_ws = mpimg.imread(data_path+'/bmaps/bmap{0}_{1}_ws.jpg'.format(gpose,index))

		fig.clf()
		ax = fig.add_subplot(2, 2, 1)
		imgplot = plt.imshow(ref_img)
		ax.set_title('Scene:{0}'.format(scene))
		# plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')
		ax = fig.add_subplot(2, 2, 2)
		imgplot = plt.imshow(pose_img)
		ax.set_title('grasp pose:{0},{1}'.format(gpose,index))
		# plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')
		ax = fig.add_subplot(2, 2, 3)
		imgplot = plt.imshow(bmap)
		ax.set_title('main')
		# plt.imshow(fig)
		ax = fig.add_subplot(2, 2, 4)
		imgplot = plt.imshow(bmap_ws)
		ax.set_title('other:'+invalid_reason)
		
		print(input('enter a number'))
		# plt.savefig('sample.png')