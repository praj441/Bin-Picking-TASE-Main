# SoftCrossEntropyLoss(input, target) = KLDivLoss(LogSoftmax(input), target)

import os
import sys
import numpy as np
from datetime import datetime
import argparse
import importlib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from pytorch_utils import BNMomentumScheduler
# from tf_visualizer import Visualizer as TfVisualizer

# from cluster_net import ClusterNet
from loss_helper import get_loss_custom, computer_ab_acc, compute_gqs_acc
# from BMC_loss import BMCLoss
# from dump_helper_custom import dump_results

sys.path.append(os.path.join(ROOT_DIR, '../commons'))
from utils_cnn import EvaluationResults
Eval_ResultProcessor = EvaluationResults()

parser = argparse.ArgumentParser()
parser.add_argument('--num_point', type=int, default=10000, help='Point Number [default: 20000]')
parser.add_argument('--num_target', type=int, default=20, help='Proposal number [default: 256]')
parser.add_argument('--vote_factor', type=int, default=1, help='Vote factor [default: 1]')
parser.add_argument('--cluster_sampling', default='vote_fps', help='Sampling strategy for vote clusters: vote_fps, seed_fps, random [default: vote_fps]')
parser.add_argument('--max_epoch', type=int, default=1004, help='Epoch to run [default: 180]')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 8]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--weight_decay', type=float, default=0, help='Optimization L2 weight decay [default: 0]')
parser.add_argument('--bn_decay_step', type=int, default=20, help='Period of BN decay (in epochs) [default: 20]')
parser.add_argument('--bn_decay_rate', type=float, default=0.5, help='Decay rate for BN decay [default: 0.5]')
parser.add_argument('--lr_decay_steps', default='80,120,160', help='When to decay the learning rate (in epochs) [default: 80,120,160]')
parser.add_argument('--lr_decay_rates', default='0.1,0.1,0.1', help='Decay rates for lr decay [default: 0.1,0.1,0.1]')
parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
BN_DECAY_STEP = FLAGS.bn_decay_step
BN_DECAY_RATE = FLAGS.bn_decay_rate
LR_DECAY_STEPS = [int(x) for x in FLAGS.lr_decay_steps.split(',')]
LR_DECAY_RATES = [float(x) for x in FLAGS.lr_decay_rates.split(',')]
assert(len(LR_DECAY_STEPS)==len(LR_DECAY_RATES))

LOG_DIR = 'log'
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'a')
LOG_FOUT.write(str(FLAGS)+'\n')
def log_string(out_str):
	LOG_FOUT.write(out_str+'\n')
	LOG_FOUT.flush()
	print(out_str)

DEFAULT_CHECKPOINT_PATH = os.path.join(LOG_DIR, 'checkpoint.tar')
CHECKPOINT_PATH = FLAGS.checkpoint_path if FLAGS.checkpoint_path is not None \
	else DEFAULT_CHECKPOINT_PATH


# Init datasets and dataloaders 
def my_worker_init_fn(worker_id):
	np.random.seed(np.random.get_state()[1][0] + worker_id)

use_color = True

#***********************************************************************
sys.path.append('Stratified_Transformer')
from util_st import config
from util_st.data_util import collate_fn, collate_fn_limit
from model.stratified_transformer import Stratified
from functools import partial
import torch_points_kernels as tp
def get_parser():
	parser = argparse.ArgumentParser(description='PyTorch Point Cloud Semantic Segmentation')
	parser.add_argument('--config', type=str, default='Stratified_Transformer/config/s3dis/s3dis_stratified_transformer.yaml', help='config file')
	parser.add_argument('opts', help='see config/s3dis/s3dis_stratified_transformer.yaml for all options', default=None, nargs=argparse.REMAINDER)
	args = parser.parse_args()
	assert args.config is not None
	cfg = config.load_cfg_from_cfg_file(args.config)
	if args.opts is not None:
		cfg = config.merge_cfg_from_list(cfg, args.opts)
	return cfg

args = get_parser()

sys.path.append(os.path.join(ROOT_DIR, 'bin_data'))
from bin_dataset_transformer import BinClusterDataset, MAX_NUM_OBJ
TRAIN_DATASET = BinClusterDataset('data_high/train', num_points=NUM_POINT,
	augment=False,
	use_color=use_color, use_height=False, voxel_size = args.voxel_size)
TEST_DATASET = BinClusterDataset('data_high/val', num_points=NUM_POINT,
	augment=False,
	use_color=use_color, use_height=False, voxel_size = args.voxel_size)


TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE,
	shuffle=True, num_workers=2, worker_init_fn=my_worker_init_fn, collate_fn=collate_fn)#partial(collate_fn_limit, max_batch_points=args.max_batch_points, logger=None))
TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=BATCH_SIZE,
	shuffle=True, num_workers=2, worker_init_fn=my_worker_init_fn, collate_fn=collate_fn)
print(len(TRAIN_DATALOADER), len(TEST_DATALOADER))
#***********************************************************************

if use_color:
	num_input_channel = 4
else:
	num_input_channel = 1

#***********************************************************************
args.patch_size = args.grid_size * args.patch_size
args.window_size = [args.patch_size * args.window_size * (2**i) for i in range(args.num_layers)]
args.grid_sizes = [args.patch_size * (2**i) for i in range(args.num_layers)]
args.quant_sizes = [args.quant_size * (2**i) for i in range(args.num_layers)]

net = Stratified(args.downsample_scale, args.depths, args.channels, args.num_heads, args.window_size, \
	args.up_k, args.grid_sizes, args.quant_sizes, rel_query=args.rel_query, \
	rel_key=args.rel_key, rel_value=args.rel_value, drop_path_rate=args.drop_path_rate, concat_xyz=args.concat_xyz, num_classes=args.classes, \
	ratio=args.ratio, k=args.k, prev_grid_size=args.grid_size, sigma=1.0, num_layers=args.num_layers, stem_transformer=args.stem_transformer)

#***********************************************************************

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.device_count() > 1:
  log_string("Let's use %d GPUs!" % (torch.cuda.device_count()))
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  # net = nn.DataParallel(net)
net.to(device)
criterion = get_loss_custom
# bmc_loss = BMCLoss(init_noise_sigma=0.2)
# Load the Adam optimizer
optimizer = optim.Adam(net.parameters(), lr=BASE_LEARNING_RATE, weight_decay=FLAGS.weight_decay)
# optimizer.add_param_group({'params': bmc_loss.noise_sigma, 'lr': BASE_LEARNING_RATE, 'name': 'noise_sigma'})
# Load checkpoint if there is any
it = -1 # for the initialize value of `LambdaLR` and `BNMomentumScheduler`
start_epoch = 0

#loading pretrained network
if not use_color:
	checkpoint = torch.load('demo_files/pretrained_votenet_on_sunrgbd.tar')
	try:
		net.module.load_my_state_dict(checkpoint['model_state_dict'])
	except:
		net.load_my_state_dict(checkpoint['model_state_dict'])

if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
	checkpoint = torch.load(CHECKPOINT_PATH)
	try:
		net.module.load_state_dict(checkpoint['model_state_dict'])
	except:
		net.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	start_epoch = checkpoint['epoch']
	log_string("-> loaded checkpoint %s (epoch: %d)"%(CHECKPOINT_PATH, start_epoch))

# Decay Batchnorm momentum from 0.5 to 0.999
# note: pytorch's BN momentum (default 0.1)= 1 - tensorflow's BN momentum
BN_MOMENTUM_INIT = 0.5
BN_MOMENTUM_MAX = 0.001
bn_lbmd = lambda it: max(BN_MOMENTUM_INIT * BN_DECAY_RATE**(int(it / BN_DECAY_STEP)), BN_MOMENTUM_MAX)
bnm_scheduler = BNMomentumScheduler(net, bn_lambda=bn_lbmd, last_epoch=start_epoch-1)

def get_current_lr(epoch):
	lr = BASE_LEARNING_RATE
	for i,lr_decay_epoch in enumerate(LR_DECAY_STEPS):
		if epoch >= lr_decay_epoch:
			lr *= LR_DECAY_RATES[i]
	return lr

def adjust_learning_rate(optimizer, epoch):
	lr = get_current_lr(epoch)
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


# ------------------------------------------------------------------------- GLOBAL CONFIG END

def train_one_epoch():
	stat_dict = {} # collect statistics
	adjust_learning_rate(optimizer, EPOCH_CNT)
	bnm_scheduler.step() # decay BN momentum
	net.train() # set model to training mode
	avg_loss = 0.0
	avg_loss_vote = 0.0
	for i, (coord, feat, target, offset) in enumerate(TEST_DATALOADER):


		offset_ = offset.clone()
		offset_[1:] = offset_[1:] - offset_[:-1]
		batch = torch.cat([torch.tensor([ii]*o) for ii,o in enumerate(offset_)], 0).long()

		sigma = 1.0
		radius = 2.5 * args.grid_size * sigma
		neighbor_idx = tp.ball_query(radius, args.max_num_neighbors, coord, coord, mode="partial_dense", batch_x=batch, batch_y=batch)[0]
	
		coord, feat, target, offset = coord.cuda(non_blocking=True), feat.cuda(non_blocking=True), target.cuda(non_blocking=True), offset.cuda(non_blocking=True)
		batch = batch.cuda(non_blocking=True)
		neighbor_idx = neighbor_idx.cuda(non_blocking=True)
		assert batch.shape[0] == feat.shape[0]
		if args.concat_xyz:
			feat = torch.cat([feat, coord], 1)

		output = net(feat, coord, offset, batch, neighbor_idx)

		input('stop !')

		for key in batch_data_label:
			batch_data_label[key] = batch_data_label[key].to(device)

		# Forward pass
		optimizer.zero_grad()
		# print('size',batch_data_label['point_clouds'].shape)
		inputs = {'point_clouds': batch_data_label['point_clouds']}
		end_points = net(inputs)
		
		# Compute loss and gradients, update parameters.
		for key in batch_data_label:
			# print(key)
			assert(key not in end_points)
			end_points[key] = batch_data_label[key]
		end_points['train'] = True
		end_points = criterion(end_points)#, DATASET_CONFIG)
		print('vote_loss:',end_points['vote_loss'].item(),'gqs_loss:',end_points['gqs_loss'].item(),'angle_loss:',end_points['angle_loss'].item(),'width_loss:',end_points['width_loss'].item())
		loss = end_points['gqs_loss'] + end_points['vote_loss']  #+ end_points['width_loss'] #+ end_points['angle_loss']
		avg_loss += end_points['gqs_loss'].item()
		avg_loss_vote += end_points['vote_loss'].item()
		# print('vote_loss:',end_points['vote_loss'].item(),'rgs_loss:',end_points['rgs_loss'].item(),'gqs_loss:',end_points['gqs_loss'].item(),'angle_loss:',end_points['angle_loss'].item(),'width_loss:',end_points['width_loss'].item())
		# loss = end_points['vote_loss'] + end_points['gqs_loss'] + end_points['rgs_loss'] + end_points['angle_loss'] + end_points['width_loss']
		loss.backward()
		optimizer.step()

		# Accumulate statistics and print out
		for key in end_points:
			if 'loss' in key or 'acc' in key or 'ratio' in key:
				if key not in stat_dict: stat_dict[key] = 0
				stat_dict[key] += end_points[key].item()

		batch_interval = 10
	avg_loss = avg_loss/(batch_idx+1)
	avg_loss_vote = avg_loss_vote/(batch_idx+1)
	return avg_loss, avg_loss_vote

def evaluate_one_epoch():
	stat_dict = {} # collect statistics
	stat_dict['vote_loss'] = 0
	stat_dict['rgs_loss'] = 0
	stat_dict['gqs_loss'] = 0
	stat_dict['angle_loss'] = 0
	stat_dict['width_loss'] = 0
	# ap_calculator = APCalculator(ap_iou_thresh=FLAGS.ap_iou_thresh,
	# 	class2type_map=DATASET_CONFIG.class2type)
	net.eval() # set model to eval mode (for bn and dp)
	for i, (coord, feat, target, offset) in enumerate(TEST_DATALOADER):


		offset_ = offset.clone()
		offset_[1:] = offset_[1:] - offset_[:-1]
		batch = torch.cat([torch.tensor([ii]*o) for ii,o in enumerate(offset_)], 0).long()

		sigma = 1.0
		radius = 2.5 * args.grid_size * sigma
		neighbor_idx = tp.ball_query(radius, args.max_num_neighbors, coord, coord, mode="partial_dense", batch_x=batch, batch_y=batch)[0]
	
		coord, feat, target, offset = coord.to(device), feat.to(device), target.to(device), offset.to(device)
		batch = batch.to(device)
		neighbor_idx = neighbor_idx.to(device)

		assert batch.shape[0] == feat.shape[0]
		if args.concat_xyz:
			feat = torch.cat([feat, coord], 1)

		output = net(feat, coord, offset, batch, neighbor_idx)

		input('stop !')

		# Compute loss
		for key in batch_data_label:
			assert(key not in end_points)
			end_points[key] = batch_data_label[key]
		# loss,end_points = criterion(end_points)#, DATASET_CONFIG)
		end_points['train'] = False
		end_points = criterion(end_points)
		result_dict = compute_gqs_acc(end_points,10.0,10*BATCH_SIZE)
		Eval_ResultProcessor.process_gqs(result_dict)
		print('Eval batch: %d'%(batch_idx))
		print(end_points['vote_loss'],end_points['gqs_loss'],end_points['angle_loss'],end_points['width_loss'])
		# print('losses',end_points['vote_loss'],end_points['rgs_loss'],end_points['gqs_loss'],end_points['angle_loss'],end_points['width_loss'])

		# Accumulate statistics and print out
		# for key in end_points:
		# 	if 'loss' in key or 'acc' in key or 'ratio' in key:
		# 		if key not in stat_dict: stat_dict[key] = 0
		stat_dict['vote_loss'] += end_points['vote_loss'].item() #end_points[key].item()
		stat_dict['rgs_loss'] += end_points['rgs_loss'].item()
		stat_dict['gqs_loss'] += end_points['gqs_loss'].item()
		stat_dict['angle_loss'] += end_points['angle_loss'].item()
		stat_dict['width_loss'] += end_points['width_loss'].item()
		# batch_pred_map_cls = parse_predictions(end_points, CONFIG_DICT) 
		# batch_gt_map_cls = parse_groundtruths(end_points, CONFIG_DICT) 
		# ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)

		dump_dir = 'demo'
		if not os.path.exists(dump_dir): os.mkdir(dump_dir) 
		# dump_results(end_points, dump_dir, True)
		# print('Dumped detection results to folder %s'%(dump_dir))

		# # Dump evaluation results for visualization
		# if FLAGS.dump_results and batch_idx == 0 and EPOCH_CNT %10 == 0:
		#     MODEL.dump_results(end_points, DUMP_DIR, DATASET_CONFIG) 

	mean_loss_vote = stat_dict['vote_loss']/float(batch_idx+1)
	mean_loss_rgs = stat_dict['rgs_loss']/float(batch_idx+1)
	mean_loss_gqs = stat_dict['gqs_loss']/float(batch_idx+1)
	mean_loss_angle = stat_dict['angle_loss']/float(batch_idx+1)
	mean_loss_width = stat_dict['width_loss']/float(batch_idx+1)
	return mean_loss_vote,mean_loss_gqs,mean_loss_angle,mean_loss_width


def train(start_epoch):
	global EPOCH_CNT 
	min_loss = 1e10
	loss_list_vote = []
	loss_list_rgs = []
	loss_list_gqs = []
	loss_list_angle = []
	loss_list_width = []

	loss_list_train_vote = []
	loss_list_train_gqs = []

	for epoch in range(start_epoch, MAX_EPOCH):
		EPOCH_CNT = epoch
		log_string('**** EPOCH %03d ****' % (epoch))
		
		# Reset numpy seed.
		# REF: https://github.com/pytorch/pytorch/issues/5059
		np.random.seed()
		# evaluate_one_epoch()
		# mean_loss_vote,mean_loss_rgs,mean_loss_gqs,mean_loss_angle,mean_loss_width = evaluate_one_epoch()
		mean_loss_vote,mean_loss_gqs,mean_loss_angle,mean_loss_width = evaluate_one_epoch()
		Eval_ResultProcessor.output_gqs_stats(LOG_DIR)
		loss_list_vote.append(mean_loss_vote)
		# loss_list_rgs.append(mean_loss_rgs)
		loss_list_angle.append(mean_loss_angle)
		loss_list_gqs.append(mean_loss_gqs)
		loss_list_width.append(mean_loss_width)

		train_gqs_loss, train_vote_loss = train_one_epoch()
		loss_list_train_gqs.append(train_gqs_loss)
		loss_list_train_vote.append(train_vote_loss)
		# if EPOCH_CNT == 0 or EPOCH_CNT % 10 == 9: # Eval every 10 epochs
		
		# log_string('loss: {0}'.format(mean_loss_vote))
		log_string('Current learning rate: %f'%(get_current_lr(epoch)))
		log_string('Current BN decay momentum: %f'%(bnm_scheduler.lmbd(bnm_scheduler.last_epoch)))
		log_string(str(datetime.now()))
		# Save checkpoint
		save_dict = {'epoch': epoch+1, # after training one epoch, the start_epoch should be epoch+1
					'optimizer_state_dict': optimizer.state_dict(),
					'loss': mean_loss_gqs,
					}
		try: # with nn.DataParallel() the net is added as a submodule of DataParallel
			save_dict['model_state_dict'] = net.module.state_dict()
		except:
			save_dict['model_state_dict'] = net.state_dict()
		# ********* plots *******************
		torch.save(save_dict, os.path.join(LOG_DIR, 'checkpoint.tar'))
		np.savetxt(LOG_DIR+'/train_loss_gqs.txt',loss_list_train_gqs,fmt='%1.4f')
		np.savetxt(LOG_DIR+'/train_loss_vote.txt',loss_list_train_vote,fmt='%1.4f')
		np.savetxt(LOG_DIR+'/val_loss_gqs.txt',loss_list_gqs,fmt='%1.4f')
		np.savetxt(LOG_DIR+'/val_loss_vote.txt',loss_list_vote,fmt='%1.4f')
		# plt.plot(loss_list_vote)
		# plt.savefig(os.path.join(LOG_DIR,'loss_vote.png'))
		# plt.clf()



		plt.plot(loss_list_vote)
		plt.savefig(os.path.join(LOG_DIR,'loss_vote.png'))
		plt.clf()

		plt.plot(loss_list_gqs)
		plt.savefig(os.path.join(LOG_DIR,'loss_gqs.png'))
		plt.clf()

		plt.plot(loss_list_angle)
		plt.savefig(os.path.join(LOG_DIR,'loss_angle.png'))
		plt.clf()

		plt.plot(loss_list_width)
		plt.savefig(os.path.join(LOG_DIR,'loss_width.png'))
		plt.clf()
		#************ plots done **********************
if __name__=='__main__':
	train(start_epoch)

print('done')