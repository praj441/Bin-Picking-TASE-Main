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

from cluster_net import ClusterNet
from loss_helper import get_loss_custom, compute_gqs_acc
from dump_helper_custom import dump_results

sys.path.append(os.path.join(ROOT_DIR, '../commons'))
from utils_cnn import EvaluationResults
Eval_ResultProcessor = EvaluationResults()

parser = argparse.ArgumentParser()
parser.add_argument('--num_point', type=int, default=2000, help='Point Number [default: 20000]')
parser.add_argument('--num_target', type=int, default=20, help='Proposal number [default: 256]')
parser.add_argument('--vote_factor', type=int, default=1, help='Vote factor [default: 1]')
parser.add_argument('--cluster_sampling', default='vote_fps', help='Sampling strategy for vote clusters: vote_fps, seed_fps, random [default: vote_fps]')
parser.add_argument('--max_epoch', type=int, default=1003, help='Epoch to run [default: 180]')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 8]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--weight_decay', type=float, default=0, help='Optimization L2 weight decay [default: 0]')
parser.add_argument('--bn_decay_step', type=int, default=20, help='Period of BN decay (in epochs) [default: 20]')
parser.add_argument('--bn_decay_rate', type=float, default=0.5, help='Decay rate for BN decay [default: 0.5]')
parser.add_argument('--lr_decay_steps', default='80,120,160', help='When to decay the learning rate (in epochs) [default: 80,120,160]')
parser.add_argument('--lr_decay_rates', default='0.1,0.1,0.1', help='Decay rates for lr decay [default: 0.1,0.1,0.1]')
parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
parser.add_argument('--noise_model', action='store_true')
FLAGS = parser.parse_args()

NOISE_MODEL = FLAGS.noise_model
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


sys.path.append(os.path.join(ROOT_DIR, 'bin_data'))
from bin_cluster_dataset import BinClusterDataset, MAX_NUM_OBJ
# TRAIN_DATASET = BinClusterDataset('data/train', num_points=NUM_POINT,
# 	augment=False,
# 	use_color=False, use_height=True)
TEST_DATASET = BinClusterDataset('data/val', num_points=NUM_POINT,
	augment=False,
	use_color=False, use_height=True, noise_model=NOISE_MODEL)


# TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE,
# 	shuffle=True, num_workers=10, worker_init_fn=my_worker_init_fn)
TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=BATCH_SIZE,
	shuffle=True, num_workers=10, worker_init_fn=my_worker_init_fn)
print(len(TEST_DATALOADER))



num_input_channel = 1
net = ClusterNet(num_proposal=FLAGS.num_target,
			   input_feature_dim=num_input_channel,
			   vote_factor=FLAGS.vote_factor,
			   sampling=FLAGS.cluster_sampling)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.device_count() > 1:
  log_string("Let's use %d GPUs!" % (torch.cuda.device_count()))
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  net = nn.DataParallel(net)
net.to(device)
criterion = get_loss_custom

# Load the Adam optimizer
optimizer = optim.Adam(net.parameters(), lr=BASE_LEARNING_RATE, weight_decay=FLAGS.weight_decay)

# Load checkpoint if there is any
it = -1 # for the initialize value of `LambdaLR` and `BNMomentumScheduler`
start_epoch = 0

#loading pretrained network
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
	for batch_idx, batch_data_label in enumerate(TRAIN_DATALOADER):
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
		end_points = criterion(end_points)#, DATASET_CONFIG)
		print('rgs_loss:',end_points['rgs_loss'].item(),'gqs_loss:',end_points['gqs_loss'].item(),'angle_loss:',end_points['angle_loss'].item(),'width_loss:',end_points['width_loss'].item())
		loss = end_points['gqs_loss'] + end_points['rgs_loss'] + end_points['angle_loss'] + end_points['width_loss']
		
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

def evaluate_one_epoch():
	stat_dict = {} # collect statistics
	stat_dict['loss'] = 0
	stat_dict['rgs_loss'] = 0
	stat_dict['gqs_loss'] = 0
	stat_dict['angle_loss'] = 0
	stat_dict['width_loss'] = 0
	# ap_calculator = APCalculator(ap_iou_thresh=FLAGS.ap_iou_thresh,
	# 	class2type_map=DATASET_CONFIG.class2type)
	net.eval() # set model to eval mode (for bn and dp)
	for batch_idx, batch_data_label in enumerate(TEST_DATALOADER):
		# if batch_idx % 10 == 0:
			# print('Eval batch: %d'%(batch_idx))
		for key in batch_data_label:
			batch_data_label[key] = batch_data_label[key].to(device)
		
		# Forward pass
		inputs = {'point_clouds': batch_data_label['point_clouds']}
		with torch.no_grad():
			end_points = net(inputs)

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
		print(end_points['rgs_loss'],end_points['gqs_loss'],end_points['angle_loss'],end_points['width_loss'])
		# print('losses',end_points['vote_loss'],end_points['rgs_loss'],end_points['gqs_loss'],end_points['angle_loss'],end_points['width_loss'])

		# Accumulate statistics and print out
		# for key in end_points:
		# 	if 'loss' in key or 'acc' in key or 'ratio' in key:
		# 		if key not in stat_dict: stat_dict[key] = 0
		# stat_dict['loss'] += end_points['vote_loss'].item() #end_points[key].item()
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

	# mean_loss_vote = stat_dict['loss']/float(batch_idx+1)
	mean_loss_rgs = stat_dict['rgs_loss']/float(batch_idx+1)
	mean_loss_gqs = stat_dict['gqs_loss']/float(batch_idx+1)
	mean_loss_angle = stat_dict['angle_loss']/float(batch_idx+1)
	mean_loss_width = stat_dict['width_loss']/float(batch_idx+1)

	return mean_loss_rgs,mean_loss_gqs,mean_loss_angle,mean_loss_width


def evaluate(start_epoch):
	
	mean_loss_rgs,mean_loss_gqs,mean_loss_angle,mean_loss_width = evaluate_one_epoch()
	print('mean_loss_rgs',mean_loss_rgs)
	print('mean_loss_gqs',mean_loss_gqs)
	print('mean_loss_angle',mean_loss_angle)
	print('mean_loss_width',mean_loss_width)
	Eval_ResultProcessor.output_gqs_stats()
if __name__=='__main__':
	evaluate(start_epoch)

print('done')