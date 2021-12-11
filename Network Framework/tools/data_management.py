def get_flat_data(input_, get_indices=False, distances_from_center=None):
    """
    CONVERTS 3D ARRAY INTO 1D ARRAY IN ORDER TO USE LOSS FUNCTION
    :param input_: batch of 3D tensors
    :param get_indices: boolean indicating whether or not to return flat array containing distance from center of voxel block of each voxel
    :param distances_from_center: 3D array with each element indicating the distance from center of voxel block of each voxel
    :return: batch of flattened input tensors and, if get_indices, then corresponding tensor of same shape containing the distance of each voxel to the cener of the voxel block
    """
    if get_indices:
        input_ = torch.swapaxes(input_,1,4).contiguous().view(-1,input_.shape[1])
        input_indices_ = np.reshape(np.swapaxes(distances_from_center,1,4), (-1,distances_from_center.shape[1]))
        return input_, input_indices_
    input_ = torch.swapaxes(input_,1,4).contiguous().view(-1,input_.shape[1])
    return input_

def reverse_get_flat_data(input_flat_, dimensions=None):
    """
    CONVERTS 1D FLATTENED ARRAY BACK INTO 3D ARRAY
    :param input_flat_: batch of 1D flattened tensors
    :return: output of corresponding batch of 3D tensors
    """
    if not dimensions: dimensions = 16
    input_ = torch.swapaxes(input_flat_.view(-1, dimensions, dimensions, dimensions, input_flat_.shape[1]),1,4)
    return input_


def data_network_type(data, get_classes=True):
    """
    CONVERTS NUMPY ARRAY FROM 1D TO 3D TENSORS NEEDED FOR NEURAL NETWORKS INPUTS
    :param data: numpy data where rows contain depth, weights and class information
    :return: distance weight and clases tensors and finally the dimensions of the voxel blocks
    """
    distances, weights, classes = np.split(data, 3, axis=1)
    dimensions = np.cbrt(distances.shape[1]).astype(int)

    distances = distances.reshape((distances.shape[0], dimensions, dimensions, dimensions))
    distances = np.swapaxes(distances, axis1=1, axis2=3)

    weights = weights.reshape(distances.shape)
    weights = np.swapaxes(weights, axis1=1, axis2=3)

    distances = distances[:, np.newaxis,:,:,:]
    weights = weights[:, np.newaxis,:,:,:]
    if get_classes:
        classes = classes.reshape(distances.shape).astype(int)
        classes= np.swapaxes(classes, axis1=1, axis2=3)
        classes = classes[:, np.newaxis,:,:,:]
        return torch.tensor(distances), torch.tensor(weights), torch.tensor(classes-1), dimensions
    return torch.tensor(distances), torch.tensor(weights), dimensions



