#!/usr/bin/env python
# coding: utf-8

# In[5]:

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 96, 11, stride=4, padding=1)  # convolutional layer 1, output 123x36x96
        self.pool1 = nn.MaxPool2d(3, stride=2) # maxpool 1, output 61x17x96
        
        self.conv2 = nn.Conv2d(96, 256, 5, stride=1, padding=2) # convolutional layer 2, output 61x17x256
        self.pool2 = nn.MaxPool2d(3, stride=2) # maxpool 2, output 30x8x256
        
        self.conv3 = nn.Conv2d(256, 384, 3, stride=1, padding=1) # convolutional layer 3, output 30x8x384
        self.conv4 = nn.Conv2d(384, 384, 3, stride=1, padding=1) # convolutional layer 4, output 30x8x384
        self.conv5 = nn.Conv2d(384, 256, 3, stride = 1, padding=1) # convolutional layer 5, output 30x8x256
        self.pool3 = nn.MaxPool2d(3, stride=2) # maxpool 3, output 14x3x256
        self.fc1 = nn.Linear(14*3*256, 4096)   # fully connected layer 1
        
        self.fc2 = nn.Linear(4096, 4096) # fully connected layer 2
        self.fc3 = nn.Linear(4096, 6) # fully connected layer 3
        self.sig = nn.Sigmoid()

        self.init_weights()

    def init_weights(self):
        # initialize weights
        for conv in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]:
            init = conv.weight.size(1)
            nn.init.normal_(conv.weight,0.0,1/math.sqrt(12.5*init))
            nn.init.constant_(conv.bias,0.0)

    def forward(self, x):
        # forward pass
        z = F.relu(self.conv1(x))
        z = self.pool1(z)
        z = F.relu(self.conv2(z))
        z = self.pool2(z)
        z = F.relu(self.conv3(z))
        z = F.relu(self.conv4(z))
        z = F.relu(self.conv5(z))
        z = self.pool3(z)
        z = torch.flatten(z,1)
        z = F.relu(self.fc1(z))
        z = self.sig(self.fc2(z))
        z = self.fc3(z)
        
        return z




if __name__ == '__main__':
    net = AlexNet()
    print(net)





