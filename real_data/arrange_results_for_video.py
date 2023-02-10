import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

scene = 1
data_path = 'results_ros_policy'
out_path = 'test_temp'

fig = plt.figure()
my_dpi = 200
plt.rcParams['figure.dpi'] = my_dpi
plt.rcParams['savefig.dpi'] = my_dpi
# plt.figure(figsize=(400/my_dpi, 900/my_dpi), dpi=my_dpi)
plt.ion()

# fig = plt.figure(frameon=False)
fig.set_size_inches(950/my_dpi,1500/my_dpi)
fig.show()

nota = [6,18,29,44,54,59]

skips = 0

for scene in range(31):
	if scene+1 in nota:
		skips += 1
		continue
	elif scene+1 <= 30:
		continue
	# elif scene+1 == 31:
	# 	skips -= 1
	final_img = mpimg.imread(data_path+'/{0}_final.png'.format(scene))
	gqs_img = mpimg.imread(data_path+'/{0}_gqs_map.png'.format(scene))
	# seed_img = mpimg.imread(data_path+'/{0}_top_N_points.png'.format(scene))
	seed_img = mpimg.imread(out_path+'/top_N_points/{0}_topN_points.png'.format(scene))

	fig.clf()
	
	# plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')

	# plt.title('Scene:{0}'.format(scene),  fontsize=10)
	# ax = plt.Axes(fig, [0., 0., 1., 1.])
	# ax.set_axis_off()
	# fig.add_axes(ax)

	fig.suptitle('Iteration:{0}'.format(scene+1-skips),  fontsize=18)

	ax = fig.add_subplot(3, 1, 1)
	ax.set_axis_off()
	imgplot = plt.imshow(gqs_img)
	ax.set_title('Graspability Map', y=-0.15,  fontsize=12)

	ax = fig.add_subplot(3, 1, 2)
	ax.set_axis_off()
	imgplot = plt.imshow(seed_img)
	ax.set_title('Seed Points', y=-0.15,  fontsize=12)
	
	ax = fig.add_subplot(3, 1, 3)
	ax.set_axis_off()
	imgplot = plt.imshow(final_img)
	ax.set_title('Grasp Pose Output', y=-0.15, fontsize=12)

	


	plt.savefig(out_path+'/video_result/{0}.png'.format(scene+1))