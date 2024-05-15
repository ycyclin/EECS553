#!/usr/bin/env python
# coding: utf-8

# In[15]:


# dataset.py
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import os

from parse import Tracklet
from tabulate import tabulate


class KittiDataset:
    def __init__(self, batch_size = 1, if_resize=True):
        self.batch_size = batch_size
        self.train_videos = self.calculate_train_videos()
        self.if_resize = if_resize
        self.train_tracklet = []
        for i in range(len(self.train_videos)):
            self.train_tracklet.append(Tracklet('train/'+self.train_videos[i]))
        self.val_tracklet = Tracklet('val')
        self.train_index, self.val_index = self.calculate_index()
        self.x_mean, self.x_std = self.train_statistics()
        self.transform = self.get_transforms()
        self.train_loader, self.val_loader = self.get_dataloaders()
        self.train_label, self.val_label = self.get_labels()

    def calculate_train_videos(self):
        # calculates how many non-hidden subdirectories there are in 'train'
        file = []
        for f in os.listdir('train'):
            if not f.startswith('.'):
                file.append(f)
                
        return file
    
    def calculate_index(self):
        train_index = [[] for i in range(len(self.train_videos))]
        for k in range(len(self.train_videos)):
            table_data = []
            for i, car in enumerate(self.train_tracklet[k].cars):
                table_data.append([
                    f"Object {i + 1}",
                    car['objectType'],
                    car['first_frame'],
                    len(car['poses'])
                ])
                if car['objectType'] == "Car":
                    for i in range(car['first_frame'],car['first_frame']+len(car['poses'])):
                        if i in train_index[k]:
                            continue
                        else:
                            train_index[k].append(i)
                elif car['objectType'] == "Van":
                    for i in range(car['first_frame'],car['first_frame']+len(car['poses'])):
                        if i in train_index[k]:
                            continue
                        else:
                            train_index[k].append(i)

                elif car['objectType'] == "Truck":
                    for i in range(car['first_frame'],car['first_frame']+len(car['poses'])):
                        if i in train_index[k]:
                            continue
                        else:
                            train_index[k].append(i)
 
                            
            headers = ['Object', 'Object Type', 'First Frame', 'Count']
            print(tabulate(table_data, headers=headers))
            
        val_index = []
        table_data = []
        for i, car in enumerate(self.val_tracklet.cars):
            table_data.append([
                f"Object {i + 1}",
                car['objectType'],
                car['first_frame'],
                len(car['poses'])
            ])
            if car['objectType'] == "Car":
                for i in range(car['first_frame'],car['first_frame']+len(car['poses'])):
                    if i in val_index:
                        continue
                    else:
                        val_index.append(i)
            elif car['objectType'] == "Van":
                 for i in range(car['first_frame'],car['first_frame']+len(car['poses'])):
                    if i in val_index:
                        continue
                    else:
                        val_index.append(i)
    
            elif car['objectType'] == "Truck":
                 for i in range(car['first_frame'],car['first_frame']+len(car['poses'])):
                    if i in val_index:
                        continue
                    else:
                        val_index.append(i)
                
            
            headers = ['Object', 'Object Type', 'First Frame', 'Count']
            print(tabulate(table_data, headers=headers))
        return train_index, val_index

   
    def train_statistics(self):
        # Calculate mean and std of input data
        temp = 0
        for i in range(len(self.train_videos)):
            temp = temp + len(self.train_index[i])
        
        train_x = np.zeros((temp, 375, 1242, 3))
        temp = 0
        for i in range(len(self.train_videos)):
            for j, (img, _) in enumerate(torchvision.datasets.ImageFolder(os.path.join('train/'+self.train_videos[i], 'image_02'))):
                if j in self.train_index[i]:
                    #print(img)
                    train_x[temp] = img
        x_mean = np.mean(train_x/255.0 , axis=(0,1,2))
        x_std = np.std(train_x/255.0 , axis=(0,1,2))
        return x_mean, x_std
    
    def get_transforms(self):
        if self.if_resize == True:
            transform_list = [
                transforms.Resize((150,497)),
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize(self.x_mean,self.x_std)
            ]
        else:
            transform_list = [
                transforms.CenterCrop((150,497)),
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize(self.x_mean,self.x_std)
            ]
        transform = transforms.Compose(transform_list)
        return transform

    def get_dataloaders(self):
        # train set
        train_set = [[] for i in range(len(self.train_videos))]
        train_loader = [[] for i in range(len(self.train_videos))]
        for i in range(len(self.train_videos)):
            train_set[i] = torchvision.datasets.ImageFolder(os.path.join('train/'+self.train_videos[i], 'image_02'), transform=self.transform)                  
            train_loader[i] = torch.utils.data.DataLoader(train_set[i], batch_size=self.batch_size, shuffle=False)
        
        
        # validation set
        val_set = torchvision.datasets.ImageFolder(os.path.join('val', 'image_02'), transform=self.transform)                  
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=self.batch_size, shuffle=False)
        return train_loader, val_loader
    
    def get_labels(self):
        # training labels
        train_lab = [[] for i in range(len(self.train_videos))]
        for k in range(len(self.train_videos)):
            train_lab[k] = 50*np.ones([len(self.train_index[k]),6])
            for j in range(len(self.train_index[k])):
                train_lab[k][j][4] = -50
                train_lab[k][j][2] = 0
            for i, car in enumerate(self.train_tracklet[k].cars):
                if car['objectType'] == "Car":
                    tx_values = [-pose['ty'] for pose in car['poses']]
                    ty_values = [pose['tx'] for pose in car['poses']]
                    for j in range(car['first_frame'],car['first_frame']+len(car['poses'])):
                        a = self.train_index[k].index(j)
                        if (tx_values[j-car['first_frame']] > 1.6) & (tx_values[j-car['first_frame']] < 12):
                            # right area
                            if (train_lab[k][a][0] == 50) & (train_lab[k][a][1] == 50):
                                train_lab[k][a][0] = tx_values[j-car['first_frame']]
                                train_lab[k][a][1] = ty_values[j-car['first_frame']]
                            else:
                                if (tx_values[j-car['first_frame']]**2+ty_values[j-car['first_frame']]**2) < (train_lab[k][a][0]**2 + train_lab[k][a][1]**2):
                                    train_lab[k][a][2] = tx_values[j-car['first_frame']]
                                    train_lab[k][a][3] = ty_values[j-car['first_frame']]
                        elif (tx_values[j-car['first_frame']] < 1.6) & (tx_values[j-car['first_frame']] > -1.6):
                            # center area
                            if (train_lab[a][2] == 0) & (train_lab[a][3] == 50):
                                train_lab[k][a][2] = tx_values[j-car['first_frame']]
                                train_lab[k][a][3] = ty_values[j-car['first_frame']]
                            else:
                                if (tx_values[j-car['first_frame']]**2+ty_values[j-car['first_frame']]**2) < (train_lab[k][a][2]**2 + train_lab[k][a][3]**2):
                                    train_lab[k][a][2] = tx_values[j-car['first_frame']]
                                    train_lab[k][a][3] = ty_values[j-car['first_frame']]
                        elif (tx_values[j-car['first_frame']] < -1.6) & (tx_values[j-car['first_frame']] > -12):
                            # left area
                            if (train_lab[k][a][4] == -50) & (train_lab[k][a][5] == 50):
                                train_lab[k][a][4] = tx_values[j-car['first_frame']]
                                train_lab[k][a][5] = ty_values[j-car['first_frame']]
                            else:
                                if (tx_values[j-car['first_frame']]**2+ty_values[j-car['first_frame']]**2) < (train_lab[k][a][4]**2 + train_lab[k][a][5]**2):
                                    train_lab[k][a][4] = tx_values[j-car['first_frame']]
                                    train_lab[k][a][5] = ty_values[j-car['first_frame']]
                    
        # validation labels
        val_lab = 50*np.ones([len(self.val_index),6])
        for j in range(len(self.val_index)):
                val_lab[j][4] = -50
                val_lab[j][2] = 0

        for i, car in enumerate(self.val_tracklet.cars):
            if car['objectType'] == "Car":
                tx_values = [-pose['ty'] for pose in car['poses']]
                ty_values = [pose['tx'] for pose in car['poses']]
                for j in range(car['first_frame'],car['first_frame']+len(car['poses'])):
                    a = self.val_index.index(j)
                    if (tx_values[j-car['first_frame']] > 1.6) & (tx_values[j-car['first_frame']] < 12):
                        # right area
                        if (val_lab[a][0] == 50) & (val_lab[a][1] == 50):
                            val_lab[a][0] = tx_values[j-car['first_frame']]
                            val_lab[a][1] = ty_values[j-car['first_frame']]
                        else:
                            if (tx_values[j-car['first_frame']]**2+ty_values[j-car['first_frame']]**2) < (val_lab[a][0]**2 + val_lab[a][1]**2):
                                val_lab[a][2] = tx_values[j-car['first_frame']]
                                val_lab[a][3] = ty_values[j-car['first_frame']]
                    elif (tx_values[j-car['first_frame']] < 1.6) & (tx_values[j-car['first_frame']] > -1.6):
                        # center area
                        if (val_lab[a][2] == 0) & (val_lab[a][3] == 50):
                            val_lab[a][2] = tx_values[j-car['first_frame']]
                            val_lab[a][3] = ty_values[j-car['first_frame']]
                        else:
                            if (tx_values[j-car['first_frame']]**2+ty_values[j-car['first_frame']]**2) < (val_lab[a][2]**2 + val_lab[a][3]**2):
                                val_lab[a][2] = tx_values[j-car['first_frame']]
                                val_lab[a][3] = ty_values[j-car['first_frame']]
                    elif (tx_values[j-car['first_frame']] < -1.6) & (tx_values[j-car['first_frame']] > -12):
                        # left area
                        if (val_lab[a][4] == -50) & (val_lab[a][5] == 50):
                            val_lab[a][4] = tx_values[j-car['first_frame']]
                            val_lab[a][5] = ty_values[j-car['first_frame']]
                        else:
                            if (tx_values[j-car['first_frame']]**2+ty_values[j-car['first_frame']]**2) < (val_lab[a][4]**2 + val_lab[a][5]**2):
                                val_lab[a][4] = tx_values[j-car['first_frame']]
                                val_lab[a][5] = ty_values[j-car['first_frame']]
        #print(val_lab)
        return train_lab, val_lab
     
    
if __name__ == '__main__':
    dataset = KittiDataset()
    print(dataset.x_mean)
    print(dataset.x_std)
