# Some basic functions that may be helpful
import os

import numpy as np
import skimage as sk
import skimage.io as skio

from skimage import img_as_ubyte
from scipy.signal import convolve2d


def get_file_image(fname):
        relative_data_dir = "data"
        dir = os.getcwd()
        image_path = os.path.join(dir, relative_data_dir, fname)

        image = skio.imread(image_path)
        image = sk.img_as_float(image)
        return image
    
def save_image(image, name, is_out=True):
        # Scale image data to 0-255 range for storing
        normalize_factor = 1
        if image.max() <= 1:
                normalize_factor = 255
        scaled_image = (image.copy() * normalize_factor).astype(np.uint8)
        # save the image
        file_name = 'out/' + name + '.jpg' if is_out else 'data/' + name + '.jpg'
        skio.imsave(file_name, scaled_image)

def channel_to_image(r, g, b, alpha=None):
        if alpha is None:
                return np.dstack(([r, g, b]))
        return img_as_ubyte(np.dstack([r, g, b, alpha]))

def get_image_channels(image):
        assert len(image.shape) == 3
        if image.shape[2] == 3:
                return image[:, :, 0], image[:, :, 1], image[:, :, 2], None
        elif image.shape[2] == 4:
                return image[:, :, 0], image[:, :, 1], image[:, :, 2], image[:, :, 3]
        else:
                raise ValueError("Image does not have 3 or 4 channels")
            
def apply_func_to_all_channels(image, function):
        if len(image.shape) == 2:
                return function(image)
        r, g, b, alpha = get_image_channels(image)
        
        r = function(r)
        g = function(g)
        b = function(b)
        alpha = function(alpha) if alpha is not None else alpha
        
        return np.dstack([r, g, b]) if alpha is None else np.dstack([r, g, b, alpha])
        
def rgb2gray(image, with_alpha=False):
        assert len(image.shape) == 3
        if image.shape[2] == 3:
                return image.dot([0.2989, 0.5870, 0.1140])
        elif image.shape[2] == 4:
                rgb_channels = image[:, :, :3]
                alpha_channel = image[:, :, 3]
                
                grayed_rgb = rgb_channels.dot([0.2989, 0.5870, 0.1140])
                return grayed_rgb + (1 - alpha_channel) if with_alpha else grayed_rgb
        else:
                raise ValueError("Image does not have 3 or 4 channels")
    
def normalize(image):
        min_val = np.min(image)
        max_val = np.max(image)
        normalized_image = (image - min_val) / (max_val - min_val)
        return normalized_image
    