import json
import os
from matplotlib import pyplot as plt

import numpy as np

from Utils.raw_image_operations import get_file_image


class ImagePointSet:
    """
    Allows quick storage and retrieval of points on respective image pairs based on Json file.
    """
    def __init__(self, json_file_name=None) -> None:
        if json_file_name:
            self.read_file(json_file_name)
        else:
            self.images = ()
            self.points = () 
         
    def read_file(self, json_file_name):
        relative_data_dir = "data"
        dir = os.getcwd()
        file_path = os.path.join(dir, relative_data_dir, json_file_name)
        
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
        
        image1 = get_file_image(os.path.dirname(json_file_name) + '/' + data['im1_name'] + ".jpg")
        image2 = get_file_image(os.path.dirname(json_file_name) + '/' + data['im2_name'] + ".jpg")
            
        points1 = np.array(data['im1Points'])
        points2 = np.array(data['im2Points'])
        
        self.images = (image1, image2)
        self.points = (points1, points2)
        
    def display(self, is_show_points=False):
        _, axes = plt.subplots(1, 2)
        axes[0].imshow(self.images[0])
        axes[1].imshow(self.images[1])
        
        if is_show_points:
            axes[0].plot(self.points[0][:, 0], self.points[0][:, 1], 'o', markersize=5, markeredgecolor='red')
            axes[1].plot(self.points[1][:, 0], self.points[1][:, 1], 'o', markersize=5, markeredgecolor='red')
