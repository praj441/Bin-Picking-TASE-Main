import numpy as np 
from tqdm import tqdm
path = 'test_data_level_1'
import torch


max_pool = torch.nn.MaxPool2d(9, padding=4, stride=1,return_indices=True).cuda()

for i in tqdm(range(1,51)):
	gqs_map = torch.FloatTensor(np.loadtxt(path+'/{0:06d}_gqs_array.txt'.format(i))).cuda()
	gqs_map, indices  = max_pool(gqs_map.unsqueeze(0))

	np.save(path+'/{0:06d}_gqs_map.npy'.format(i),gqs_map.cpu().numpy())