import numpy as np
import torch
from data_preperation import VoxbloxData
from train import Train
from PALNet import PALNet
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch import nn
import pathlib

experiment_number = '1_yes_test'
not_test = False

on_euler=True
n_classes=14 #including non-class class
lr = 0.001
batch_size = 256
n_epochs = 150
reconstruct_data = False
StepLR_period = 100
train_split = 0.8
voxel_size=0.08
use_incremental=False
loss_on_trunc=False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("GPU available? ", torch.cuda.is_available())

try: 
	print(torch.cuda.get_device_name(device=device))
	print(torch.cuda.get_device_properties(device).total_memory/1e+9)
except:pass

if on_euler:
    data_location = '/cluster/work/riner/users/PLR-2021/map-segmentation/voxel_data'
    data_save_dir = '/cluster/work/riner/users/PLR-2021/map-segmentation'
    experiment_dir = '/cluster/work/riner/users/PLR-2021/map-segmentation/experiments/'+ experiment_number
    test_dir = '/cluster/work/riner/users/PLR-2021/map-segmentation/val_voxel_data/0/300'
    try:pathlib.Path(experiment_dir).mkdir(parents=True)
    except: print('path exists')

Blocks = VoxbloxData(data_location, data_save_dir, construct_data=reconstruct_data, train_split=train_split, use_incremental=use_incremental)#, test_data_dir=test_dir, not_test=not_test)

net = PALNet(number_of_classes=n_classes)
if loss_on_trunc:boundary_scale=2
else: boundary_scale=1

train = Train(Blocks, net, experiment_dir=experiment_dir, device=device, voxel_size=voxel_size, boundary_scale=boundary_scale)
train.train_model(n_epochs, Adam, scheduler=StepLR,schedule=StepLR_period, lr=lr, batch_size=batch_size, accuracy_frequency=1)#, not_test=not_test)
train.plot_loss()
train.plot_accuracy()
net, accuracy = train.load_checkpoint(net)

text_file = open(experiment_dir + "/output.txt","w")
if not not_test:
    text_file.write('n_classes='+str(n_classes) +'\nlr='+str(lr) + '\nbatch_size='+str(batch_size)+'\nn_epochs='+str(n_epochs) + '\nStepLR_period='+ str(StepLR_period) + '\ntrain_split='+str(train_split)+'\n\nTrain score: ' + str(train.test_model(loader=train.train_loader, net_object=net))+'\nVal score: ' + str(train.test_model(loader=train.val_loader, net_object=net))+ '\nTest score: ' + str(train.test_model(loader=train.test_loader, net_object=net)) + '\nUse incremental: '+str(use_incremental) +'\nLoss on trucnc:'+str(loss_on_trunc))
    text_file.close()
else:
    text_file.write('n_classes='+str(n_classes) +'\nlr='+str(lr) + '\nbatch_size='+str(batch_size)+'\nn_epochs='+str(n_epochs) + '\nStepLR_period='+ str(StepLR_period) + '\ntrain_split='+str(train_split)  + '\nUse incremental: '+str(use_incremental) +'\nLoss on trucnc:'+str(loss_on_trunc))
    text_file.close()


#catkin build [package_name] --no-deps






