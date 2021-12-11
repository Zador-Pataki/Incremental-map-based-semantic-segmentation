from tqdm import tqdm
import os
import numpy as np
import torch
import pandas as pd

def save_data(test_dir_csv, save_data_dir):
    """
    CONCATENATES TEST DATA FROM INDIVIDUAL CSV FILES (CONTAINING TSDF WEIGHT AND SEMANTIC LABEL INFORMATION) OF CORRESPONDING SCENES
    :param test_dir_csv: (string) directory in which csv files of data is saved
    :param save_data_dir: (string) directory to which concatenated numpy file is saved
    """
    test_list=[]
    for subdir, dirs, files in os.walk(test_dir_csv):
        if files:
            print(subdir)
            for file in tqdm(files):
                data_i =  pd.read_csv(subdir + '/'+file, header=None, index_col=None,error_bad_lines=False).iloc[:, :-1]
                test_list.append(data_i.to_numpy())
    data_test = np.concatenate(test_list, axis=0)
    np.save(save_data_dir+'/voxel_test.npy', data_test)

def load_data(npy_data_path):
    """
    LOADS NUMPY DATA AND CONVERTS IT INTO NEURAL NETWORK  DATA TYPE
    :param npy_data_path: (string) path of the numpy file is is to be loaded
    :return: numpy array loaded from corresponding path
    """
    data_test = np.load(npy_data_path)
    data = data_network_type(data_test)
    return data

if __name__ == "__main__":
    num = '280'
    csv_dir = '/cluster/work/riner/users/PLR-2021/map-segmentation/val_voxel_data/0/'+num
    #csv_dir = '/cluster/work/riner/users/PLR-2021/map-segmentation/voxel_data/3/300'
    save_dir = '/cluster/work/riner/users/PLR-2021/map-segmentation/interm_test_dir/'+num
    save_data(csv_dir, save_dir)
