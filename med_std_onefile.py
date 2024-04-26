# This file computes the mean and standard deviation 
# for the RGB channels of the bee pixels in the image.

import cv2
import numpy as np

# Retrieve image and mask
image = cv2.imread('project_data/train/images/1.jpg')
masque = cv2.imread('project_data/train/masks/binary_1.tif', cv2.IMREAD_GRAYSCALE)

# Isolate the part of the image corresponding to the mask 
image_isolee = cv2.bitwise_and(image, image, mask=masque)

# Get pixels corresponding to the bee 
bee_pixels = image_isolee[masque != 0]

# Compute mean and standard deviation for each channel
mean_values = np.mean(bee_pixels, axis=0)
std_values = np.std(bee_pixels, axis=0)

print("Mean RGB:", mean_values)
print("Standard Deviation RGB:", std_values)


