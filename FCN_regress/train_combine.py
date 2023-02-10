import os
import numpy as np
import cv2
import torchvision.models.segmentation
import torch
import torchvision.transforms as tf
from dataset import Dataset
from concate_dataset import ConcatDataset
from torch.utils import data

Learning_Rate=1e-5
width=320
height=240 # image width and height
batchSize=8
max_pool = torch.nn.MaxPool2d(9, padding=4, stride=1,return_indices=False)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
num_class = 6

mse_loss = torch.nn.MSELoss()
def mse_loss_weighted(preds,targets):
	num_total = targets.shape[0]

	pos_indices = torch.where(targets>0.0)[0]
	neg_indices = torch.where(targets==0.0)[0]
	num_pos = pos_indices.shape[0]
	# print('samples gqs',num_total,num_pos)
	bw = num_pos/num_total
	maxr = 0.5
	# if bw > maxr:
	# 	bw = maxr

	targets_pos = targets[pos_indices]
	targets_neg = targets[neg_indices]

	preds_pos = preds[pos_indices]
	preds_neg = preds[neg_indices]

	pos_loss = mse_loss(preds_pos,targets_pos)
	neg_loss = mse_loss(preds_neg,targets_neg)

	return (1/bw)*pos_loss+(1/1-bw)*neg_loss

type = 'depth_only'

if type == 'rgb_depth':
	transformImg=tf.Compose([tf.Normalize((0.485, 0.456, 0.406,0.4), (0.229, 0.224, 0.225,0.2))])
else:
	transformImg=tf.Compose([tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


TrainFolder1="../../data/data_high/train"
ValFolder1="../../data/data_high/val"
TrainFolder2="../../data/data_mid/train"
ValFolder2="../../data/data_mid/val"

train_dataset1 = Dataset(TrainFolder1,transform=transformImg,type=type)
train_dataset2 = Dataset(TrainFolder2,transform=transformImg,type=type)
train_dataset = ConcatDataset(train_dataset1,train_dataset2)
train_loader = data.DataLoader(train_dataset, batch_size=batchSize, shuffle=True,
  num_workers=2)

val_dataset1 = Dataset(ValFolder1,transform=transformImg,type=type)
val_dataset2 = Dataset(ValFolder2,transform=transformImg,type=type)
val_dataset = ConcatDataset(val_dataset1,val_dataset2)
val_loader = data.DataLoader(val_dataset, batch_size=batchSize, shuffle=True,
  num_workers=2)
# ListImages=os.listdir(os.path.join(TrainFolder, "Image")) # Create list of images
#----------------------------------------------Transform image-------------------------------------------------------------------
# transformAnn=tf.Compose([tf.ToPILImage(),tf.Resize((height,width),tf.InterpolationMode.NEAREST),tf.ToTensor()])
#---------------------Read image ---------------------------------------------------------

#--------------Load and set net and optimizer-------------------------------------
Net = torchvision.models.segmentation.fcn_resnet101(pretrained=True) # Load net
Net.classifier[4] = torch.nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1)) # Change final layer to 3 classes

if type == 'rgb_depth':
	Net.backbone.conv1 = torch.nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

# feature_net = Net.backbone

if torch.cuda.device_count() > 1:
  print("Let's use %d GPUs!" % (torch.cuda.device_count()))
  Net = torch.nn.DataParallel(Net)
start_epoch = 0
loss_val_list = []
loss_train_list = []
if start_epoch > 0:  
	checkpoint = torch.load('weights/{0}.torch'.format(start_epoch-1))
	Net.load_state_dict(checkpoint)
	print('weights loaded from epoch',start_epoch-1)
	loss_val_list = list(np.loadtxt('val_loss.txt'))
	loss_train_list = list(np.loadtxt('train_loss.txt'))
Net.to(device)
optimizer=torch.optim.Adam(params=Net.parameters(),lr=Learning_Rate) # Create adam optimizer

def evaluate():
	avg_loss = 0.0
	avg_pres = np.zeros(num_class)
	avg_recall = np.zeros(num_class)
	for i,batch in enumerate(val_loader):
		images = torch.cat((batch[0][0],batch[1][0])).to(device)
		ann = torch.cat((batch[0][1],batch[1][1])).to(device)
		net_out=Net(images) # make prediction
		Pred = net_out['out']
		Net.zero_grad()
		ann = max_pool(ann)

		Pred = Pred.reshape(-1)
		ann = ann.reshape(-1)
		mask = ann >= 0
		ann = ann[mask]
		Pred = Pred[mask]

		Loss=mse_loss(Pred,torch.div(ann,100))
		avg_loss += Loss.item()

		# seg = seg.reshape(-1)
		# ann = ann.reshape(-1)
		# for k in range(num_class):
		# 	maskP = seg==k
		# 	maskR = ann==k
		# 	P = 100*torch.sum(torch.eq(seg[maskP],ann[maskP]))/torch.sum(maskP)
		# 	R =  100*torch.sum(torch.eq(seg[maskR],ann[maskR]))/torch.sum(maskR)
		# 	avg_pres[k] += P
		# 	avg_recall[k] += R
		# print('batch',i,images.shape[0])
	# return avg_acc/(i+1), avg_pres/(i+1), avg_recall/(i+1)
		# print('batch',i,images.shape[0])
	return avg_loss/(i+1)

# a dict to store the activations
activation = {}
def getActivation(name):
  # the hook signature
  def hook(model, input, output):
    activation[name] = output #.detach()
  return hook

h1 = Net.module.classifier[4].register_forward_hook(getActivation('features'))


#----------------Train--------------------------------------------------------------------------
for epoch in range(start_epoch,100): # Training loop
	avg_loss = 0.0
	for i,batch in enumerate(train_loader):
		# images=torch.autograd.Variable(images,requires_grad=False).to(device) # Load image
		# ann = torch.autograd.Variable(ann, requires_grad=False).to(device) # Load annotation
		images = torch.cat((batch[0][0],batch[1][0])).to(device)
		ann = torch.cat((batch[0][1],batch[1][1])).to(device)
		import time
		st = time.time()
		net_out=Net(images) # make prediction
		print('forward pass time',time.time()-st)
		Pred = net_out['out']
		Net.zero_grad()
		ann = max_pool(ann)

		features = activation['features']
		# indx = torch.where(ann>=0)[0]
		# ann = ann[indx]
		# Pred = Pred[indx.repeat(1,6,1,1)]
		Pred = Pred.reshape(-1)
		ann = ann.reshape(-1)

		mask = ann >= 0
		ann = ann[mask]
		Pred = Pred[mask]

		# seg = torch.argmax(Pred, 1)
		# acc = 100*torch.sum(torch.eq(seg,ann))/torch.sum(ann >=0 )
		# avg_acc += acc
		
		# Loss=mse_loss(Pred,torch.div(ann,100)) # Calculate cross entropy loss
		Loss=mse_loss(Pred,torch.div(ann,100))
		Loss.backward() # Backpropogate loss
		optimizer.step() # Apply gradient descent change to weight
		
		avg_loss += Loss.item()
		print(i,'{0:2.4f}'.format(Loss.item()))
		#.cpu().detach().numpy()  # Get  prediction classes
		# print(itr,") Loss=",Loss.data.cpu().numpy())
		if i % 100 == 0: #Save model weight once every 60k steps permenant file
			loss = evaluate()
			# print('Precision',pres)
			# print('Recall', recall)
			print('Epoch {0}'.format(epoch),'Batch {0}'.format(i),'loss train {0:2.4f}'.format(avg_loss/(i+1)),'loss val {0:2.4f}'.format(loss))
			loss_val_list.append(loss)
			loss_train_list.append(avg_loss/(i+1))
	np.savetxt('val_loss.txt',loss_val_list)
	np.savetxt('train_loss.txt',loss_train_list)
	print("Saving Model " +str(epoch) + ".torch")
	torch.save(Net.state_dict(),  "weights/" + str(epoch) + ".torch")
