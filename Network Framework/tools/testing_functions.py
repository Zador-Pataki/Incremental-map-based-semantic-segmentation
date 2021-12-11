def test_model(self, loader, batch_size=None, net_object = None):
    """
    CALCULATES THE ACCURACY OF THE OCCUPIED VOXELS IN THE TEST SET
    :param loader: test set data loader
    :param batch_size: (int) batch size
    :param net_object: neural network object
    :return: (float) accuracy of model tested on test data
    """
    if batch_size is None: batch_size = len(loader.dataset)
    if net_object is None: net_object = self.net
    metric_sum = 0
    div = 0
    for input, target, weights in loader:
        prediction_ = prediction(input.to(self.device), weights, net_object)
        input_flat = get_flat_data(input)
        prediction_flat = get_flat_data(prediction_)
        target_flat = get_flat_data(target)
        weights_flat = get_flat_data(weights)
        occupied_idx = torch.nonzero(torch.logical_and((torch.abs(input_flat)<=self.voxel_size), (weights_flat!=0))) ###############
        target_flat = target_flat[occupied_idx[:,0], :]
        prediction_flat = prediction_flat[occupied_idx[:,0], :]
        prediction_flat_hard = torch.argmax(prediction_flat, dim=1)
        metric_sum += torch.sum(prediction_flat_hard==target_flat[:,0].to(self.device)).item()
        div += occupied_idx.shape[0]
    return metric_sum/div

def plot_loss(self, PATH=None):
    """
    SAVES LOSS PROGRESSION PLOT AGAINST EPOCHS
    :param PATH: path to which plot will be saved; if None, saved to self.experiment_dir
    """
    if PATH is None: PATH = self.experiment_dir + '/training_loss.png'
    fig = plt.figure()
    plt.plot(self.loss_list)
    plt.title("Training Loss")
    fig.savefig(PATH)

def plot_accuracy(self, PATH=None):
    """
    SAVES ACCURACY SCORE PROGRESSION PLOT AGAINST EPOCHS
    :param PATH: pat to which plot will be saved; if None, saved to self.experiment_dir
    """
    if PATH is None: PATH = self.experiment_dir + '/validation_accuracy.png'
    fig = plt.figure()
    plt.plot(self.epochs_list, self.accuracy_list)
    plt.title("Validation Accuracy")
    fig.savefig(PATH)

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
