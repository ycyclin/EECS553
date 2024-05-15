#!/usr/bin/env python
# coding: utf-8
# This code loads our trained model and uses images in the test data as input to generate outputs
# It prints the predicted value and the true value for each image
# In[24]:

from dataset import KittiDataset
from checkpoint import load_checkpoint
from model import AlexNet
from PIL import Image
import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision
import os
import matplotlib.pyplot as plt



def get_transforms():
    transform_list = [
            transforms.Resize((150,497)),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(x_mean,x_std)
            ]
    transform = transforms.Compose(transform_list)
    return transform

if __name__ == '__main__':
    model = AlexNet()
    trained_model=load_checkpoint(model,'checkpoints')
    dataset = KittiDataset()
    x_std = dataset.x_std
    x_mean = dataset.x_mean

    test_set = torchvision.datasets.ImageFolder(os.path.join('test', 'image_02'), transform=get_transforms())
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)
    image_path = 'test/image_02/data/'
    for i, (img, _) in enumerate(test_loader):
        prediction = trained_model(img)
        for j, (img1, _) in enumerate(test_set):
            if i == j:
                print('prediction: ', prediction)
                print('label: ', dataset.val_label[dataset.val_index.index(j)])
                print('figure ', i)
                img1 = np.transpose(img1.numpy(), (1, 2, 0))
                img1 = img1 * x_std.reshape(1, 1, 3) + x_mean.reshape(1, 1, 3)  # un-normalize
                plt.imshow((img1*255).astype('uint8'))
                plt.show()




