#!/usr/bin/env python
# coding: utf-8


import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from tabulate import tabulate

class Tracklet:
    def __init__(self, dataset_path="2011_09_26_drive_0018_sync", if_resize=True):
        self.tree = ET.parse(dataset_path+"/tracklet_labels.xml")
        self.root = self.tree.getroot()
        self.cars = []
        for car in self.root.findall('.//item'):
            car_data = self.parse_car_data(car)
            if car_data is not None:
                self.cars.append(car_data)
        
    
    def parse_car_data(self,car):
        car_data = {}
        objectType = car.find('objectType')
        if objectType is None:
            return None
        car_data['objectType'] = objectType.text
        car_data['h'] = float(car.find('h').text)
        car_data['w'] = float(car.find('w').text)
        car_data['l'] = float(car.find('l').text)
        car_data['first_frame'] = int(car.find('first_frame').text)

        poses = car.find('poses')
        pose_count = int(poses.find('count').text)
        car_data['poses'] = []

        for pose in poses.findall('item'):
            pose_data = {}
            pose_data['tx'] = float(pose.find('tx').text)
            pose_data['ty'] = float(pose.find('ty').text)
            pose_data['tz'] = float(pose.find('tz').text)
            pose_data['rx'] = float(pose.find('rx').text)
            pose_data['ry'] = float(pose.find('ry').text)
            pose_data['rz'] = float(pose.find('rz').text)

            car_data['poses'].append(pose_data)

        return car_data


    def print_object_info(self):
        table_data = []
        for i, car in enumerate(self.cars):
            table_data.append([
                f"Object {i + 1}",
                car['objectType'],
                car['first_frame'],
                len(car['poses'])
            ])
        headers = ['Object', 'Object Type', 'First Frame', 'Count']
        print(tabulate(table_data, headers=headers))

    def plot_tx_ty_history(self):
        for i, car in enumerate(self.cars):
            tx_values = [pose['tx'] for pose in car['poses']]
            ty_values = [pose['ty'] for pose in car['poses']]

            plt.figure(i + 1)
            plt.plot(tx_values, ty_values, color='black', label=f"Object {i + 1} ({car['objectType']})")
            plt.scatter(tx_values[0], ty_values[0], color='black', marker='o', s=50, edgecolors='white', linewidth=1.5)
            plt.xlabel('tx')
            plt.ylabel('ty')
            #plt.legend()
            plt.axis('equal')

            plt.title(f'Object {i + 1} ({car["objectType"]}) - First Frame: {car["first_frame"]}, Count: {len(car["poses"])}')

            object_type = car['objectType']
            plt.savefig(f'object{i + 1:02d}_{object_type}.png', format='png')
            plt.close()

            # Add the data to the combined graph
            plt.figure(0)
            plt.plot(tx_values, ty_values, label=f"Object {i + 1} ({car['objectType']})")
            plt.scatter(tx_values[0], ty_values[0], marker='o', s=50, edgecolors='white', linewidth=1.5)

            '''
            # Display the text near the first data point for each line
            for i, car in enumerate(cars):
                tx = car['poses'][0]['tx']
                ty = car['poses'][0]['ty']
                plt.text(tx, ty, f" {i + 1}", fontsize=10, verticalalignment='bottom', horizontalalignment='left')
            '''

        # Set up the combined graph
        plt.figure(0)
        plt.xlabel('tx')
        plt.ylabel('ty')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.axis('equal')
        plt.title('All Objects')
        plt.savefig('objects_all.png', format='png', bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    test = Tracklet()
    test.print_object_info()
    #plot_tx_ty_history(cars)






