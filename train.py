#!/usr/bin/env python
# coding: utf-8

# In[15]:


import math
import torch
import numpy as np
import random
import checkpoint # may need to write own checkpoint
from dataset import KittiDataset
from model_yen import AlexNet
from plot import Plotter # may need to write own plotter

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


def _train_epoch(train_loader, train_label, train_index, model, criterion, optimizer):
    """
    Train the model for one iteration through the train set.
    """
    for k in range(len(train_loader)):
        for i, (X, y) in enumerate(train_loader[k]):
            if i in train_index[k]:
                optimizer.zero_grad()
                
                a = dataset.train_index[k].index(i)

                output = model(X)
                loss = criterion(output.float(), torch.tensor(train_label[k][a]).reshape(1,6).float())
                print(loss)
                loss.backward()
                optimizer.step()


def evaluate(plotter, dataset, model, criterion, epoch):
    """
    Evaluates the model on the train and validation set.
    """
    stat = []
    y_true_val, y_pred_val, loss_val = evaluate_val(dataset, model, criterion)
    #total_loss_val = math.sqrt(np.sum(loss_val) / (3*len(dataset.val_index)))
    #total_loss_val_x = (loss_val[0]+loss_val[2]+loss_val[4])/(3*len(dataset.val_index))
    #total_loss_val_y = (loss_val[1]+loss_val[3]+loss_val[5])/(3*len(dataset.val_index))
    total_loss_val_x = (loss_val[0]+loss_val[2]+loss_val[4])/6
    total_loss_val_y = (loss_val[1]+loss_val[3]+loss_val[5])/6



    total_loss_train_x = 0
    total_loss_train_y = 0
    temp = 0
    for k in range(len(dataset.train_videos)):
        temp = temp + len(dataset.train_index[k])
        y_true_train, y_pred_train, loss_train = evaluate_train(dataset, model, criterion)
        #total_loss_train = total_loss_train + np.sum(loss_train)
        total_loss_train_x = total_loss_train_x + loss_train[0] + loss_train[2] + loss_train[4]
        total_loss_train_y = total_loss_train_y + loss_train[1] + loss_train[3] + loss_train[5]
    #total_loss_train = math.sqrt(total_loss_train / (3*temp))
    total_loss_train_x = total_loss_train_x / 12
    total_loss_train_y = total_loss_train_y / 12

    stat += [total_loss_val_x,total_loss_val_y,total_loss_train_x,total_loss_train_y]
    print('epoch ', epoch,', validation loss x = ', total_loss_val_x)
    print('epoch ', epoch,', validation loss y = ', total_loss_val_y)
    print('epoch ', epoch,', training loss x = ', total_loss_train_x)
    print('epoch ', epoch,', training loss y = ', total_loss_train_y)

    plotter.stats.append(stat)
    plotter.update_plot(epoch)


def evaluate_val(data, model, criterion=None):
    model.eval()
    y_true = []
    y_pred = []
    loss = []
    for i ,(X, y) in enumerate(data.val_loader):
        if i in data.val_index:
            with torch.no_grad():
                predicted = model(X)
                a = dataset.val_index.index(i)
                y_true.append(torch.tensor(data.val_label[a]).reshape(1,6))
                y_pred.append(predicted)
                if criterion is not None:
                    loss.append(criterion(predicted, torch.tensor(data.val_label[a]).reshape(1,6)).item())
    model.train()
    y_true, y_pred = torch.cat(y_true), torch.cat(y_pred)
    return y_true, y_pred, loss

def evaluate_train(data, model, criterion=None):
    model.eval()
    y_true = []
    y_pred = []
    loss = []
    for k in range(len(data.train_videos)):
        for i ,(X, y) in enumerate(data.train_loader[k]):
            if i in data.train_index[k]:
                with torch.no_grad():
                    predicted = model(X)
                    a = dataset.train_index[k].index(i)
                    y_true.append(torch.tensor(data.train_label[k][a]).reshape(1,6))
                    y_pred.append(predicted)
                    if criterion is not None:
                        loss.append(criterion(predicted, torch.tensor(data.train_label[k][a]).reshape(1,6)).item())
    model.train()
    y_true, y_pred = torch.cat(y_true), torch.cat(y_pred)
    return y_true, y_pred, loss


def train(config, dataset, model):
    # load data
    train_loader, val_loader = dataset.train_loader, dataset.val_loader

    # assign parameters
    criterion = torch.nn.L1Loss()    
    learning_rate = config['lr']
    momentum = config['momentum']
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate,momentum = momentum)

    # load trained model
    force = config['ckpt_force'] if 'ckpt_force' in config else False
    model, start_epoch, stats = checkpoint.restore_checkpoint(model, config['ckpt_path'], force=force)

    plotter = Plotter(stats, 'AlexNet')

    # Evaluate the model
    evaluate(plotter, dataset, model, criterion, start_epoch)

    # Loop over the entire dataset multiple times
    for epoch in range(start_epoch, config['epoch']):
        # Train model on training set
        _train_epoch(dataset.train_loader, dataset.train_label, dataset.train_index, model, criterion, optimizer)

        # Evaluate model on training and validation set
        evaluate(plotter, dataset, model, criterion, epoch + 1)

        # Save model parameters
        checkpoint.save_checkpoint(model, epoch + 1, config['ckpt_path'], plotter.stats)

    print('Finished Training')

    # Save figure and keep plot open
    plotter.save_plot()
    plotter.hold_plot()


if __name__ == '__main__':
    # define config parameters for training
    param = {
        'dataset_path': 'train',
        'if_resize': True,             # If resize of the image is needed 
        'ckpt_path': 'checkpoints/AlexNet',  # directory to save our model checkpoints
        'epoch': 10,                 # number of epochs for training
        'lr': 8e-6,           # learning rate
        'momentum': 0.9,                  # momentum 
    }
    # create dataset
    dataset = KittiDataset(if_resize = True)
    # create model
    model_short = AlexNet()
    #model_long = AlexNet()
    train(param, dataset, model_short)

