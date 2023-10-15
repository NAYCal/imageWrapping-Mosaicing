import json
import os
import cv2
from matplotlib import pyplot as plt

import numpy as np

from Utils.raw_image_operations import get_file_image

DEFAULT_BATCH_SIZE = 5
# Note: The json file stores points in (column, row) format.
class ImagePointSet:
    """
    Allows quick storage and retrieval of points on respective image pairs based on Json file.
    """
    def __init__(self, json_file_name=None, images=None, points=None) -> None:
        if json_file_name:
            self.read_file(json_file_name)
        else:
            assert len(images) == 2 and len(points) == 2
            self.images = images
            self.points = points
         
    def read_file(self, json_file_name):
        relative_data_dir = "data"
        dir = os.getcwd()
        file_path = os.path.join(dir, relative_data_dir, json_file_name)
        
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
        
        image_path1 = os.path.join(os.path.dirname(json_file_name),  data['im1_name'] + ".jpg")
        image_path2 = os.path.join(os.path.dirname(json_file_name),  data['im2_name'] + ".jpg")
        
        image1 = get_file_image(image_path1)
        image2 = get_file_image(image_path2)
            
        points1 = np.array(data['im1Points'])
        points2 = np.array(data['im2Points'])
        
        self.images = (image1, image2)
        self.points = (points1, points2)
        
    def display(self, is_show_points=False, figsize=(12, 8)):
        _, axes = plt.subplots(1, 2, figsize=figsize)
        axes[0].imshow(self.images[0])
        axes[1].imshow(self.images[1])
        
        if is_show_points:
            axes[0].plot(self.points[0][:, 0], self.points[0][:, 1], 'o', markersize=5, markeredgecolor='red')
            axes[1].plot(self.points[1][:, 0], self.points[1][:, 1], 'o', markersize=5, markeredgecolor='red')
            
    def change_image(self, image, index):
        if index == 0:
            self.images = (image, self.images[1])
        elif index == 1:
            self.images = (self.images[0], image)
        else:
            raise ValueError("Invalid image input")
        
    def change_points(self, points, index):
        if index == 0:
            self.points = (points, self.points[1])
        elif index == 1:
            self.points = (self.points[0], points)
        else:
            raise ValueError("Invalid points input")
        
    def align_point(self, first_point, second_point):
        """Checks a batch of the images and find the most matching points
        """
        def sum_of_squared_differences(first_image, second_image):
            return np.sum((first_image - second_image) ** 2)
        
        def choose_target_and_base_image():
            #Grayscale images as well
            first_image, second_image = cv2.cvtColor((self.images[0] * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY), cv2.cvtColor((self.images[1] * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
            
            if first_image.shape <= second_image.shape:
                return first_image.copy(), second_image.copy(), first_point, second_point, 1
            return second_image.copy(), first_image.copy(), second_point, first_point, 0
        
        def get_centered_batch(image, point, batchsize):
            center_x, center_y = image.shape[1] // 2, image.shape[0] // 2
            dx, dy = center_x - point[0], center_y - point[1]

            return np.roll(image, (dy, dx), axis=(0, 1))[center_y - batchsize:center_y + batchsize, center_x - batchsize:center_x + batchsize]

        target_image, base_image, target_point, base_point, base_index = choose_target_and_base_image()
        base_image_batch = get_centered_batch(base_image, base_point, DEFAULT_BATCH_SIZE)
        
        best_target_point = target_point
        best_loss = float("inf")
        for row in range(-DEFAULT_BATCH_SIZE, DEFAULT_BATCH_SIZE):
            for col  in range(-DEFAULT_BATCH_SIZE, DEFAULT_BATCH_SIZE):
                target_batch_point = [target_point[0] + col, target_point[1] + row]
                target_image_batch = get_centered_batch(target_image, target_batch_point, DEFAULT_BATCH_SIZE)
                
                loss = sum_of_squared_differences(target_image_batch, base_image_batch)
                if loss < best_loss:
                    best_loss = loss
                    best_target_point = target_batch_point
        
        # print(first_point, second_point, target_batch_point, base_point, best_target_point)
        
        if base_index == 0:
            return first_point, best_target_point
        else:
            return best_target_point, second_point
        
    
    def align_all_points(self):
        new_first_point = []
        new_second_point = []
        
        for first_point, second_point in zip(self.points[0], self.points[1]):
            new_point = self.align_point(first_point, second_point)
            new_first_point.append(new_point[0])
            new_second_point.append(new_point[1])
        
        self.points = (np.array(new_first_point), np.array(new_second_point))
        
        