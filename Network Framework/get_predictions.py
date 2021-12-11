import torch
import sys
sys.path.insert(1,'/home/patakiz/Documents/plr-2021-map-segmentation/Network Framework/')
from PALNet import PALNet
#from data_preperation import VoxbloxData
#from train import Train
import os
from tqdm import tqdm, trange
import pickle
import ntpath
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tools.data_management import get_flat_data, reverse_get_flat_data


#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("GPU available? ", torch.cuda.is_available())

data_location = '/cluster/work/riner/users/PLR-2021/map-segmentation/voxel_data'
data_save_dir = '/cluster/work/riner/users/PLR-2021/map-segmentation'
train_split = 0.7
n_classes=14 #including non-class class


def save_list(data_save_dir, list_, file_name):
    """
    SAVES LIST TO A .PKL FILE
    :param data_save_dir: directory to which list will be saved
    :param list_: list to be saved
    :param file_name: name of list
    """
    open_file = open(data_save_dir + '/' + file_name+'.pkl', "wb")
    pickle.dump(list_, open_file)
    open_file.close()

def load_list(data_save_dir, file_name):
    """
    LOADS LIST FROM A .PKL FILE
    :param data_save_dir: directory in which list to be loaded was saved
    :param file_name: name of list
    :return: loaded list is returned
    """
    open_file = open(data_save_dir +'/'+ file_name+'.pkl', "rb")
    loaded_list = pickle.load(open_file)
    open_file.close()
    return loaded_list

def trajectory_input_to_pred(csv_data_path, csv_pred_path, net_object, device, voxel_size):
    """
    ACCESSES VOXBLOX INPUT DATA FILE, CONVERTS IT INTO A DATALOADER, MAKES PREDICTIONS FOR EACH VOXEL BLOCK, CONVERTS IT
    BACK INTO ORIGINAL DATA FORMAT WITH UPDATED CALSSES AND SAVES CSV FILE TO BE LATER ACCESSED BY CATKIN WS
    :param csv_data_path: file path where to be updated voxblox arrays are saved
    :param csv_pred_path: file to where voxblox arrays are to be saved after voxel classes are predicted
    :param net_object: neural network object
    :param device: cpu/gpu device
    :param voxel_size: size of each voxel
    """
    data = pd.read_csv(csv_data_path, header=None, index_col=None).iloc[:,:-1]
    data= data.to_numpy()
    distances, weights, dimensions = data_network_type(data, get_classes=False)
    dataset = TensorDataset(distances)
    data_loader = DataLoader(dataset, batch_size=32)

    data_stored = []
    for input in data_loader:
        pred = prediction(input_=input[0], net_object=net_object, device=device, dimensions=dimensions)
        pred_dimred = torch.argmax(pred, dim=1)
        data_stored.append(pred_dimred.cpu().numpy())
    pred_classes = np.concatenate(data_stored, axis=0)
    distances = np.swapaxes(distances.cpu().numpy()[:,0,:,:,:], axis1=3, axis2=1)
    pred_classes = np.swapaxes(pred_classes, axis1=3, axis2=1)
    distances = distances.reshape((distances.shape[0], -1))
    pred_classes = pred_classes.reshape((pred_classes.shape[0], -1))
    occupied_idx = np.nonzero(np.logical_and((np.abs(distances)<=voxel_size), (input_flat!=0))) ###############
    pred_classes[occupied_idx] = pred_classes[occupied_idx]+1
    new_data = data_with_pred_classes(data, pred_classes)
    new_data = pd.DataFrame(new_data)
    print(csv_pred_path)
    new_data.to_csv(csv_pred_path, header = False, index = False)

    return 0

def data_with_pred_classes(data, classes):
    """
    CONSTRUCTS AND RETURNS DATA IN ORIGINAL FORMAT RECEIVED FROM VOXBLOX USING PREDICTED CLASSES
    :param data: original voxblox data
    :param classes: predicted classes
    :return: returns new voxblox data where original classes are replaced with predicted classes
    """
    distances, weights, _ = np.split(data, 3, axis=1)
    return np.concatenate([distances, weights, classes], axis=1)

device = 'cpu' #aaaaaaaaaaaaaaaaa
#data_dir_test_loaded = load_list(data_save_dir, 'data_dir_test') #aaaaaaaaaaaaaaaaaa
#def trajectory_input_to_pred(csv_data_path, csv_pred_dir, net_object, device):
#print(data_dir_test_loaded[0]) #aaaaaaaaaaaaaaaaaa
net_path = '/home/patakiz/Documents/experiments/5/opt_net.pth'
data_save_path="/media/patakiz/Extreme SSD/data_incremental/voxblox_net_output.csv"
net_object = PALNet(number_of_classes=n_classes)
net_object, _ = load_checkpoint(net_object, net_path, device)
voxel_size = 0.08

#for i in trange(len(data_dir_test_loaded)):
#trajectory_input_to_pred(data_dir_test_loaded[i], None, net_object, device) #aaaaaaaaaaaaaa
_ = trajectory_input_to_pred("/media/patakiz/Extreme SSD/data_incremental/voxblox_net_input.csv", data_save_path, net_object, device, voxel_size)
print('completed')








   


