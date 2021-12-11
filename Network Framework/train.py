import torch
from torch import nn
from tqdm import trange, tqdm
from torch.optim import Adam
from matplotlib import pyplot as plt
import numpy as np
from tools.data_management import get_flat_data, reverse_get_flat_data
from tools.network_functions import prediction

class Train():
    def __init__(self, Data_Object, Network_Object, voxel_size, experiment_dir=None, device=torch.device("cpu"), boundary_scale=1):
        self.D_Object = Data_Object
        self.net = Network_Object.to(device)
        self.device = device
        self.loss_list = []
        self.accuracy_list = None
        self.epochs_list = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self.net_state_path = experiment_dir +'/opt_net.pth'
        self.experiment_dir = experiment_dir
        self.voxel_size=voxel_size
        self.boundary_scale=boundary_scale

    def train_model(self, num_epochs, optimizer, scheduler=None, schedule=100, lr=0.0001, train_split = 0.8, batch_size=64, accuracy_frequency=None, not_test=False):
        """
        VOID FUNCTION; MAIN FUNCTION OF THE TRAINING PROCESS
        :param num_epochs: (int) Maximum number of training epochs
        :param optimizer: Optimizing scheme
        :param scheduler: Learning rate scheduler of trianing scheme
        :param schedule: (int) Number of epochs after which after which learning rate is updated
        :param lr: (float) Initial learning rate
        :param train_split: (float) Fraction of training data
        :param batch_size: (int) batch size
        :param accuracy_frequency: rate at which validation accuracy is calculated
        :param not_test: can't remember what this is
        """
        if accuracy_frequency is not None:
            self.accuracy_list = []
            self.epochs_list = []
            accuracy_epoch = int(1 / accuracy_frequency)
        optimizer = optimizer(self.net.parameters(), lr=lr)
        try:
            scheduler = scheduler(optimizer, schedule)
            use_scheduler = True
        except:
            print("NO LR SCHEDULER")
            use_scheduler = False
        self.train_loader, self.val_loader, self.test_loader = self.D_Object.get_dataloaders(batch_size=batch_size, not_test=not_test)
        for epoch in trange(num_epochs):
            running_loss = 0.0

            if accuracy_frequency is not None and epoch % accuracy_epoch == 0:
                accuracy = self.test_model(self.val_loader)
                self.accuracy_list.append(accuracy)
                self.epochs_list.append(epoch)
                if accuracy>=max(self.accuracy_list): self.save_checkpoint(accuracy, self.net.state_dict(), epoch, self.net_state_path)

            for i, data in enumerate(self.train_loader, 0):
                loss = self.train_step(data,optimizer)
                running_loss += loss
            self.loss_list.append(running_loss)

            if use_scheduler:scheduler.step()

    def train_step(self,data,optimizer):
        """
        PERFORMS FORWARD AND BACKWARD PASS OF NETWORK AND RETURNS LOSS VALUE
        :param data: batch of training data
        :param optimizer: Optimizing scheme
        :return: (float) loss value as standard python number (wihtout grad information)
        """
        inputs, classes_true, weights = data
        optimizer.zero_grad()
        classes_pred = self.net(inputs.to(self.device))
        loss = self.criterion(inputs.to(self.device), classes_pred.to(self.device), classes_true.to(self.device), weights.to(self.device))
        loss.backward()
        optimizer.step()
        return loss.item()

    def criterion(self, input_, prediction_, target_, weights_):
        """
        CALCULATES THE LOSS OF THE NETWORK OUTPUT
        :param input_: TSDF inputs of data
        :param prediction_: Predicted semantic labels
        :param target_: Target semantic labels
        :param weights_: Voxel weights
        :return: Returns the loss value including grad information
        """
        crit = nn.CrossEntropyLoss()
        input_flat = get_flat_data(input_)
        prediction_flat = get_flat_data(prediction_)
        target_flat = get_flat_data(target_)
        weights_flat = get_flat_data((weights_))
        occupied_idx = torch.logical_and((torch.abs(input_flat)<=(self.voxel_size*self.boundary_scale)), (weights_flat.to(self.device)!=0)) ###############
        target_flat = target_flat[occupied_idx[:,0], :]
        prediction_flat = prediction_flat[occupied_idx[:,0], :]
        return crit(prediction_flat,target_flat.view(-1))

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

    def save_checkpoint(self, accuracy, model_state_dict, epoch, PATH):
        """
        SAVES NN PARAMETERS WITH CORRESPONDING ACCURACY AND EPOCH NUMBER
        :param accuracy: accuracy of model
        :param model_state_dict: state_dict of current model
        :param epoch: epoch at current model
        :param PATH: path to which model parameters will be saved
        """
        torch.save({'accuracy_tracker': accuracy,
                    'net_state_dict': model_state_dict,
                    'current_epoch': epoch,
                    }, PATH)
