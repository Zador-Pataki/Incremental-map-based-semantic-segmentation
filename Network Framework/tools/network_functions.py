from tools.data_management import get_flat_data, reverse_get_flat_data

def prediction(input_, net_object, weights_=None, device=None, dimensions=None):
    """
    RETURNS THE PREDICTIONS OF A GIVEN INPUT IN THE APPROPRIATE FORM
    :param input_: batch of TSDF tensors
    :param net_object: neural network objects
    :param weights_: corresponding voxel weights
    :param device: cpu/gpu object
    :param dimensions: voxel block dimensions
    :return: 3D predicted classes with unnocupied voxel calsses set to 0
    """
    if not device: device='cpu'
    if not dimensions: dimensions=16
    prediction = net_object(input_.to(device))
    input_flat = get_flat_data(input_)
    prediction_flat = get_flat_data(prediction)
    if weights_:
        weights_flat = get_flat_data(weights_)
        unoccupied_idx = torch.logical_or((torch.abs(input_flat)>=self.voxel_size), (weights_flat.to(self.device)==0))
    else: #TODO MAY NOT BE NEEDED; IN get_predictions.py WEIGHTS ARE NOT USED BUT THIS MAY BE WRONG
        unoccupied_idx = torch.logical_or((torch.abs(input_flat)>=self.voxel_size), (input_flat==0))
    pred_flat_reconstructed = torch.clone(prediction_flat)
    pred_flat_reconstructed[unoccupied_idx, :] = 0
    pred_flat_reconstructed[unoccupied_idx, 0] = 1
    pred_reconstructed = reverse_get_flat_data(pred_flat_reconstructed, dimensions)
    return pred_reconstructed

def load_checkpoint(net_object, PATH, device=None):
    """
    LOADS NEURAL NETWORK PARAMETERS FROM GIVEN SAVED CHECKPOINT
    :param net_object: neural network object
    :param PATH: path to checkpoint where neural network parameters are saved
    :param device: cpu/gpu
    :return: network object and network checkpoint accuracy
    """
    checkpoint = torch.load(PATH, map_location=device)
    net_object.load_state_dict(checkpoint['net_state_dict'])
    accuracy = checkpoint['accuracy_tracker']
    return net_object, accuracy

