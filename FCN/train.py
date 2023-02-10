import os
import numpy as np
import cv2
import torchvision.models.segmentation
import torch
import torchvision.transforms as tf
from dataset import Dataset 
from torch.utils import data

Learning_Rate=1e-4
width=320
height=240 # image width and height
batchSize=16
max_pool = torch.nn.MaxPool2d(9, padding=4, stride=1,return_indices=False)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
num_class = 6

class_weights = [0.1,2,2,2,2,2]
class_weights=torch.tensor(class_weights,dtype=torch.float).to(device)
ce_loss_weighted = torch.nn.CrossEntropyLoss(weight=class_weights,reduction='mean')


TrainFolder="../votenet/bin_data/data/train"
ValFolder="../votenet/bin_data/data/val"

transformImg=tf.Compose([tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

train_dataset = Dataset(TrainFolder,transform=transformImg)
train_loader = data.DataLoader(train_dataset, batch_size=batchSize, shuffle=True,
  num_workers=10)

val_dataset = Dataset(ValFolder,transform=transformImg)
val_loader = data.DataLoader(val_dataset, batch_size=batchSize, shuffle=True,
  num_workers=10)
# ListImages=os.listdir(os.path.join(TrainFolder, "Image")) # Create list of images
#----------------------------------------------Transform image-------------------------------------------------------------------
# transformAnn=tf.Compose([tf.ToPILImage(),tf.Resize((height,width),tf.InterpolationMode.NEAREST),tf.ToTensor()])
#---------------------Read image ---------------------------------------------------------

#--------------Load and set net and optimizer-------------------------------------
Net = torchvision.models.segmentation.fcn_resnet101(pretrained=True) # Load net
Net.classifier[4] = torch.nn.Conv2d(512, 6, kernel_size=(1, 1), stride=(1, 1)) # Change final layer to 3 classes


if torch.cuda.device_count() > 1:
  print("Let's use %d GPUs!" % (torch.cuda.device_count()))
  Net = torch.nn.DataParallel(Net)
checkpoint = torch.load('weights/5.torch')
Net.load_state_dict(checkpoint)
Net.to(device)
optimizer=torch.optim.Adam(params=Net.parameters(),lr=Learning_Rate) # Create adam optimizer

def evaluate():
	avg_acc = 0.0
	avg_pres = np.zeros(num_class)
	avg_recall = np.zeros(num_class)
	for i,batch in enumerate(val_loader):
		images = batch[0].to(device)
		ann = batch[1].to(device)
		Pred=Net(images)['out'] # make prediction
		Net.zero_grad()
		ann = max_pool(ann)

		Pred = Pred.reshape(-1,6)
		ann = ann.reshape(-1)
		mask = ann >= 0
		ann = ann[mask]
		Pred = Pred[mask]

		seg = torch.argmax(Pred, 1)
		acc = 100*torch.sum(torch.eq(seg,ann))/torch.sum(ann >=0 )
		avg_acc += acc

		seg = seg.reshape(-1)
		ann = ann.reshape(-1)
		for k in range(num_class):
			maskP = seg==k
			maskR = ann==k
			P = 100*torch.sum(torch.eq(seg[maskP],ann[maskP]))/torch.sum(maskP)
			R =  100*torch.sum(torch.eq(seg[maskR],ann[maskR]))/torch.sum(maskR)
			avg_pres[k] += P
			avg_recall[k] += R
		# print('batch',i,images.shape[0])
	return avg_acc/(i+1), avg_pres/(i+1), avg_recall/(i+1)
		# print('batch',i,images.shape[0])
	return avg_acc/(i+1)

#----------------Train--------------------------------------------------------------------------
for epoch in range(100): # Training loop
	avg_acc = 0.0
	for i,batch in enumerate(train_loader):
		# images=torch.autograd.Variable(images,requires_grad=False).to(device) # Load image
		# ann = torch.autograd.Variable(ann, requires_grad=False).to(device) # Load annotation
		images = batch[0].to(device)
		ann = batch[1].to(device)
		Pred=Net(images)['out'] # make prediction
		Net.zero_grad()
		ann = max_pool(ann)

		
		# indx = torch.where(ann>=0)[0]
		# ann = ann[indx]
		# Pred = Pred[indx.repeat(1,6,1,1)]
		Pred = Pred.reshape(-1,6)
		ann = ann.reshape(-1)

		mask = ann >= 0
		ann = ann[mask]
		Pred = Pred[mask]

		seg = torch.argmax(Pred, 1)
		acc = 100*torch.sum(torch.eq(seg,ann))/torch.sum(ann >=0 )
		avg_acc += acc

		Loss=ce_loss_weighted(Pred,ann.long()) # Calculate cross entropy loss
		Loss.backward() # Backpropogate loss
		optimizer.step() # Apply gradient descent change to weight
		
		# print(i,acc)
		#.cpu().detach().numpy()  # Get  prediction classes
		# print(itr,") Loss=",Loss.data.cpu().numpy())
		if i % 100 == 0: #Save model weight once every 60k steps permenant file
			acc,pres,recall = evaluate()
			print('Precision',pres)
			print('Recall', recall)
			print('Epoch {0}'.format(epoch),'Batch {0}'.format(i),'Loss {0}'.format(Loss.data.cpu().numpy()),'Acc. train {0:2.2f}'.format(avg_acc/(i+1)),'Acc. val {0:2.2f}'.format(acc))
	print("Saving Model " +str(epoch) + ".torch")
	torch.save(Net.state_dict(),  "weights/" + str(epoch) + ".torch")
