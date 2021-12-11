import torch
from data_preperation import VoxbloxData
from train import Train
from PALNet import PALNet
import numpy as np
from tqdm import trange
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from tools.make_load_npy_data import load_data
import matplotlib.pyplot as plt
from tqdm import tqdm
from tools.data_management import get_flat_data
from tools.network_functions import prediction, load_checkpoint



def get_pred_and_target(loader, get_indices = False, net_object = None):
    """
    CALCULATES THE PREDICTIONS OF A GIVEN TEST LOADER AND RETURNS THAT, CORRESPONDING TARGETS AND ARRAY OF SAME SHAPE CONTAINING THE DISTANCE FROM EACH ELEMENT TO THE CENTER OF THE CORRESPONDING VOXEL BLOCK
    :param loader: test data loader
    :param get_indices: distance of voxels to the voxel block centers
    :param net_object: neural network objects
    :return: flattened predictions, corresponding flattened targets, and corresponding flattened array cotaining distances from each voxel block to corresponding centers
    """
    predictions = []
    flat_distances_list = []
    targets = []
    stop = 0
    for _, (input, target, weights) in enumerate(tqdm(loader)):
        if stop == 0:
            distances_from_center = np.zeros((input.shape[2], input.shape[3], input.shape[4]))
            centers = np.array([(input.shape[2]-1)/2, (input.shape[3]-1)/2, (input.shape[4]-1)/2])
            for i in range(input.shape[2]):
                for j in range(input.shape[3]):
                    for k in range(input.shape[4]):
                        diff = centers-np.array([i, j, k])
                        distances_from_center[i,j,k]=np.max(np.abs(diff).astype(int))
        distances_from_center_input = np.repeat(distances_from_center[np.newaxis, np.newaxis,:, :, :], input.shape[0], axis=0)

        prediction_ = prediction(input.to(train.device), weights.to(train.device), net_object)

        input_flat, flat_distances = get_flat_data(input, get_indices=get_indices, distances_from_center=distances_from_center_input)
        prediction_flat = get_flat_data(prediction_)
        target_flat = get_flat_data(target)
        weights_flat = get_flat_data(weights)

        occupied_idx = torch.nonzero(torch.logical_and((torch.abs(input_flat)<=train.voxel_size), (weights_flat!=0))) ###############
        target_flat = target_flat[occupied_idx[:,0], :]

        prediction_flat = prediction_flat[occupied_idx[:,0], :]
        flat_distances = flat_distances[occupied_idx.cpu().numpy()[:,0], :]
        prediction_flat_hard = torch.argmax(prediction_flat, dim=1)

        predictions.append(prediction_flat_hard.cpu().numpy())
        flat_distances_list.append(flat_distances[:,0])
        targets.append(target_flat.cpu().numpy()[:,0])
        stop+=1
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)

    flat_distances = np.concatenate(flat_distances_list, axis=0)
    return predictions, targets, flat_distances

def accuracy_against_distance(predictions, targets, predictions_indices):
    """
    CALCULATES THE ACCURACY OF VOXEL BLOCKS AS A FUNCTION OF HOW FAR THE VOXELS ARE FROM THE CENTER OF THE CORRESPONDING VOXEL BLOCKS
    :param predictions: batch of TSDF voxel blocks
    :param targets: batch of voxel block target semantic segmentation labels
    :param predictions_indices: corresponding array containing the distances of each element to the corresponding voxel block centers
    :return: list of different types of accuracies, with each element being the average accuracy of the corresponding distance from center
    """
    accuracies = []
    ious = []
    f1s = []
    for distance in np.unique(predictions_indices):
        index_with_distance = np.nonzero(predictions_indices==distance)
        predictions[index_with_distance]
        accuracies.append(np.sum(predictions[index_with_distance] == targets[index_with_distance])/predictions[index_with_distance].shape[0])
        ious.append(np.mean(jaccard_score(predictions[index_with_distance], targets[index_with_distance], average=None)[1:]))
        f1s.append(np.mean(f1_score(predictions[index_with_distance], targets[index_with_distance], average=None)[1:]))
    return accuracies, ious, f1s


Blocks = VoxbloxData(None, None, test_only=True, not_test=True)

#experiment_number = '8'
experiment_number = '0_yes_test'
n_classes=14
reconstruct_test = False
test_only=True

experiment_dir = '/cluster/work/riner/users/PLR-2021/map-segmentation/experiments/'+ experiment_number
#npy_data_path = '/cluster/work/riner/users/PLR-2021/map-segmentation/interm_test_dir/voxel_test.npy'
npy_data_path = '/cluster/work/riner/users/PLR-2021/map-segmentation/interm_test_dir/300/voxel_test.npy'



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("GPU available? ", torch.cuda.is_available())

data = load_data(npy_data_path=npy_data_path)
print("Data accessed")
net = PALNet(number_of_classes=n_classes)
train = Train(None, net, experiment_dir=experiment_dir, device=device, voxel_size=0.08)
_, _, test_loader=Blocks.get_dataloaders(test_only=test_only, batch_size=256, data=data)
print("Data loaders created")

net, accuracy = load_checkpoint(net, train.net_state_path)

predictions, targets, predictions_indices = get_pred_and_target(test_loader, get_indices=True, net_object=net)

print('\n\n' + 80*'-')
print("IoU")
iou_scores1 = jaccard_score(predictions, targets, average=None)[1]
iou_scores2 = jaccard_score(predictions, targets, average=None)[3:]
print("IoU socres:", iou_scores1 + iou_scores2)
print("Mean IoU scores:", np.mean(iou_scores1 + iou_scores2))
print(80*'-')

print('\n\n' + 80*'-')
print("F1")
f1_scores1 = f1_score(predictions, targets, average=None)[1]
f1_scores2 = f1_score(predictions, targets, average=None)[3:]
print("F1 socres:", f1_scores1 + f1_scores2)
print("Mean F1 scores:", np.mean(f1_scores1 + f1_scores2))
print(80*'-')

print('\n\n' + 80*'-')
print("Accuracy")
accuracy = np.sum(predictions == targets)/predictions.shape[0]
print("Accuracy:", accuracy)
print(80*'-')



accuracies, ious, f1s = accuracy_against_distance(predictions, targets, predictions_indices)


"""plt.plot(np.unique(predictions_indices), accuracies)
plt.xlabel("Distance from center of block")
plt.ylabel("Mean accuracy")
plt.savefig(experiment_dir+'/accuracy_v_distance.png')
plt.close()
plt.plot(np.unique(predictions_indices), ious)
plt.xlabel("Distance from center of block")
plt.ylabel("Mean IoU score")
plt.savefig(experiment_dir+'/iou_v_distance.png')
plt.close()
plt.plot(np.unique(predictions_indices), f1s)
plt.xlabel("Distance from center of block")
plt.ylabel("Mean F1 score")
plt.savefig(experiment_dir+'/f1_v_distance.png')
plt.close()"""


print("\n\nscores found during training")
try:

    f = open(experiment_dir + '/output.txt', 'r')
    file_contents = f.read()
    print (file_contents)
    f.close()
except: print('ERROR OCCURED DURING OPENING OUTPUT.TXT; CHECK IF PROCESS IS COMPLETE')


