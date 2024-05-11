import numpy as np
import os
import cv2
import pandas as pd
import datetime

# Used to compute execution time
start_time = datetime.datetime.now()

# Feature Extraction Functions

### Feature 1 - Symmetry Index ###

def symmetry_index(image):
    gray_array = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    half_width = gray_array.shape[1] // 2
    left_half = gray_array[:, :half_width]
    right_half = gray_array[:, half_width:]
    symmetry = np.sum(np.abs(left_half - np.flip(right_half, axis=1))) / np.prod(left_half.shape)
    return symmetry

### Feature 2 - Median and Standard Deviation of the RGB values in the image within the mask ###

def median_std(image, mask):
    image_isolated = cv2.bitwise_and(image, image, mask=mask)
    bug_pixels = image_isolated[mask != 0]
    median_values = np.median(bug_pixels, axis=0)
    std_values = np.std(bug_pixels, axis=0)
    if len(median_values) != 3:
        return [None] * 6
    return list(np.concatenate((median_values, std_values)))

### Feature 3 - Area of the bug in the mask ###

def bug_area(mask):
    return cv2.countNonZero(mask)

### New Feature 4 - Ratio of Bug Pixels to Total Pixels ###
def bug_to_total_ratio(mask):
    total_pixels = mask.shape[0] * mask.shape[1]
    bug_pixels = cv2.countNonZero(mask)
    return bug_pixels / total_pixels if total_pixels > 0 else 0

### New Feature 5 - Min, Max, Mean Values for Red, Green, and Blue within the Bug Mask ###
def min_max_mean_color_bug_mask(image, mask):
    image_isolated = cv2.bitwise_and(image, image, mask=mask)
    red_values = image_isolated[:, :, 0][mask != 0]
    green_values = image_isolated[:, :, 1][mask != 0]
    blue_values = image_isolated[:, :, 2][mask != 0]
    min_values = np.array([np.min(red_values), np.min(green_values), np.min(blue_values)])
    max_values = np.array([np.max(red_values), np.max(green_values), np.max(blue_values)])
    mean_values = np.array([np.mean(red_values), np.mean(green_values), np.mean(blue_values)])
    return min_values, max_values, mean_values

# Extracting dataset from excel file 
df = pd.read_excel('train/classif.xlsx', index_col=0, engine='openpyxl')

# Parsing files 
folder_path = 'train/'

# Main loop to extract features from each image and mask in the dataset
for i in range(1, max(len(os.listdir(folder_path + 'images')), len(os.listdir(folder_path + 'masks')))):
    image_path = os.path.join(folder_path, 'images', f'{i}.jpg')
    mask_path = os.path.join(folder_path, 'masks', f'binary_{i}.tif')

    try:
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        full_image_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # For bug_to_total_ratio
    except FileNotFoundError:
        continue

    if mask is None:
        continue  # Skip this iteration if mask is None

    df.loc[df.index[i-1], 'sym_index'] = symmetry_index(image)
    median_std_values = median_std(image, mask)
    df.loc[df.index[i-1], ['Median_R', 'Median_G', 'Median_B', 'Std_R', 'Std_G', 'Std_B']] = median_std_values
    df.loc[df.index[i-1], 'Area'] = bug_area(mask)
    
    # New Features
    df.loc[df.index[i-1], 'Bug_to_Total_Ratio'] = bug_to_total_ratio(mask)
    min_values, max_values, mean_values = min_max_mean_color_bug_mask(image, mask)

    df.loc[df.index[i-1], ['Min_R_bug', 'Min_G_bug', 'Min_B_bug']] = min_values
    df.loc[df.index[i-1], ['Max_R_bug', 'Max_G_bug', 'Max_B_bug']] = max_values
    df.loc[df.index[i-1], ['Mean_R_bug', 'Mean_G_bug', 'Mean_B_bug']] = mean_values

# Exporting DataFrame to a CSV file
df.to_csv(r'dataVisualisation\result.csv', index=False)

end_time = datetime.datetime.now()
print(f"Execution time: {end_time - start_time}")
