import pandas as pd
import numpy as np
import torch
import os
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm import tqdm
from tools.data_management import data_network_type


class VoxbloxData():
    def __init__(self, data_dir, data_save_dir, construct_data=False, train_split=0.8, use_incremental=False, test_data_dir=None, test_only=False, not_test=False):
        """
        AT INITIALIZATION, CLASS DATA IS EITHER LOADED OR CONSTRUCTED FOR PURPOSES SPECIFIED BY INPUT BOOLEANS
        :param data_dir: directory in which all raw voblox data is saved #TODO ONLY NEEDED IF construct_data BOOLEAN IS TRUE
        :param data_save_dir: data path to which constructed data will be saved to or from which it will be loaded
        :param construct_data: (boolean) specifies whether or not data should be constructed
        :param train_split: (float) value specifying the fraction of data used for training
        :param use_incremental: (boolean) specifies whether or not data should be used after only a certain number of views
        :param test_data_dir: path to which test data is saved
        :param test_only: (boolean) specifying whether or not the purpose of the class is to test data only
        :param not_test: (boolean) specifying whether or not the purpose of the class is to test data at all
        """
        self.dimensions = None
        self.distances, self.weights, self.classes = None, None, None
        self.distances_train, self.weights_train, self.classes_train = None, None, None
        self.distances_test, self.weights_test, self.classes_test = None, None, None
        self.distances_val, self.weights_val, self.classes_val = None, None, None
        self.distances, self.weights, self.classes = None, None, None
        self.distances, self.weights, self.classes = None, None, None
        self.train_split = train_split
        if construct_data:
            if os.path.isdir(data_dir):
                data_list = []
                for subdir, dirs, files in os.walk(data_dir):
                    if files and not use_incremental and os.path.basename(subdir) != '100' and os.path.basename(subdir) != '200' or files and use_incremental: # only if list of files in directory is not empty
                        print(subdir)
                        for file in tqdm(files):
                            data_i =  pd.read_csv(subdir + '/'+file, header=None, index_col=None,error_bad_lines=False).iloc[:, :-1]
                            data_list.append(data_i.to_numpy())

                train_length = int(self.train_split*len(data_list))
                if test_data_dir is None:
                    val_length = int((len(data_list) - train_length)/2)
                    data_train = data_list[:train_length]
                    data_val = data_list[train_length:train_length+val_length]
                    data_test = data_list[train_length+val_length:]
                    data_train, data_val, data_test = np.concatenate(data_train, axis=0), np.concatenate(data_val, axis=0), np.concatenate(data_test, axis=0)
                elif os.path.isdir(test_data_dir):
                    data_train = data_list[:train_length]
                    data_val = data_list[train_length:]
                    test_list = []
                    for subdir, dirs, files in os.walk(test_data_dir):
                        if files and not use_incremental and os.path.basename(subdir) != '100' and os.path.basename(subdir) != '200' or files and use_incremental: # only if list of files in directory is not empty
                            print(subdir)
                            for file in tqdm(files):
                                data_i =  pd.read_csv(subdir + '/'+file, header=None, index_col=None,error_bad_lines=False).iloc[:, :-1]
                                test_list.append(data_i.to_numpy())
                    data_train, data_val, data_test = np.concatenate(data_train, axis=0), np.concatenate(data_val, axis=0), np.concatenate(data_list, axis=0)
                else:
                    print("TEST DATA DIR IS WRONG")
                    assert True==False
                if use_incremental:
                    np.save(data_save_dir+ '/voxel_train_incr.npy', data_train)
                    np.save(data_save_dir+ '/voxel_val_incr.npy', data_val)
                    np.save(data_save_dir+ '/voxel_test_incr.npy', data_test)
                else:
                    np.save(data_save_dir+ '/interm/voxel_train.npy', data_train) ###############
                    np.save(data_save_dir+ '/interm/voxel_val.npy', data_val)      #################
                    np.save(data_save_dir+ '/interm/voxel_test.npy', data_test)   ############

            else:
                print("DATA DIR IS WRONG")
                assert True==False
        
        if not test_only:
            if use_incremental:
                data_train = np.load(data_save_dir+ '/voxel_train_incr.npy')
                data_val = np.load(data_save_dir+ '/voxel_val_incr.npy')
            else:
                data_train = np.load(data_save_dir+ '/interm/voxel_train.npy')
                data_val = np.load(data_save_dir+ '/interm/voxel_val.npy')
            self.distances_train, self.weights_train, self.classes_train, self.dimensions = data_network_type(data_train)
            self.distances_val, self.weights_val, self.classes_val, _ = data_network_type(data_val)
        if not not_test:
            data_test = np.load(data_save_dir+ '/interm/voxel_test.npy')                          ############################################## MUST PAY ATTENTION TO
            self.distances_test, self.weights_test, self.classes_test, self.dimensions = data_network_type(data_test)


    def get_dataloaders(self, batch_size=64, test_only=False, not_test=False, data=None):
        """
        CONVERTS CLASS DATA INTO DATALOADERS AS SPECIFIED BY INPUT BOOLEANS
        :param batch_size: dataloader batch sizes
        :param test_only: whether or not dataloaders are used to test neural network only
        :param not_test: whether or not dataloaders are used to test neural network at all
        :param data: voxblox data, only needed if do not want to use class data
        :return: dataloaders
        """
        if not test_only:
            train_dataset = TensorDataset(self.distances_train, self.classes_train, self.weights_train)
            val_dataset = TensorDataset(self.distances_val, self.classes_val, self.weights_val)
            train_loader = DataLoader(train_dataset, batch_size=batch_size)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        else:
            train_loader, val_loader = None, None
        if not not_test:
            if data is None:
                test_dataset = TensorDataset(self.distances_test, self.classes_test, self.weights_test)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            else:
                distances, weights, classes = data
                test_dataset = TensorDataset(distances, classes, weights)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        else:
            test_loader = None
        return train_loader, val_loader, test_loader




#vox = VoxbloxData('/media/patakiz/Extreme SSD/voxel_data', 14)
#print(vox.classes.nonzero().shape)



