import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np



scene = 1
data_path = 'test_data_mid_level_S'
method = '/baseline'


fig = plt.figure()
plt.rcParams['figure.dpi'] = 200
plt.rcParams['savefig.dpi'] = 200
plt.ion()
fig.show()
labels = np.zeros(100).astype(np.int32)
for scene in range(1,101):
	# gpose = i
	# index = j
	ref_img = mpimg.imread(data_path+'/{0:06d}_ref_image.png'.format(scene))
	# pose_img = mpimg.imread(data_path+method+'/{0:06d}/final.jpg'.format(scene))
	pose_img = mpimg.imread(data_path+method+'/final_image_{0:06d}.png'.format(scene))
	bmap = mpimg.imread(data_path+method+'/{0:06d}/bmap.jpg'.format(scene))
	bmap_ws = mpimg.imread(data_path+method+'/{0:06d}/bmap_ws.jpg'.format(scene))

	try:
		with open(data_path+method+'/{0:06d}/invalid_reason.txt'.format(scene)) as f:
			invalid_reason = f.readlines()[0]
	except:
		invalid_reason = 'NA'

	fig.clf()
	ax = fig.add_subplot(2, 2, 1)
	imgplot = plt.imshow(ref_img)
	ax.set_title('Scene:{0}'.format(scene))
	# plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')
	ax = fig.add_subplot(2, 2, 2)
	imgplot = plt.imshow(pose_img)
	ax.set_title('grasp pose')
	# plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')
	ax = fig.add_subplot(2, 2, 3)
	imgplot = plt.imshow(bmap)
	ax.set_title('main')
	# # plt.imshow(fig)
	ax = fig.add_subplot(2, 2, 4)
	imgplot = plt.imshow(bmap_ws)
	ax.set_title('other:'+invalid_reason)
	labels[scene-1] = input('enter a number')
	np.savetxt(data_path+method+'/acc.txt',labels)

print('statistic')

print('acc',np.count_nonzero(labels==1))
print('ungraspable (geometry)',np.count_nonzero(labels==0))
print('cone shape',np.count_nonzero(labels==2))
print('slanted',np.count_nonzero(labels==3))
print('depth noises',np.count_nonzero(labels==4))
print('in between-poses',np.count_nonzero(labels==5))
print('invalid poses',np.count_nonzero(labels==6))
		# plt.savefig('sample.png')