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

### Feature 2 - Median and Standard Deviatio of the RGB values in the image within the mask ###

def median_std(image, mask):
    image_isolee = cv2.bitwise_and(image, image, mask=mask)
    bee_pixels = image_isolee[mask != 0]
    median_values = np.median(bee_pixels, axis=0)
    std_values = np.std(bee_pixels, axis=0)
    if len(median_values) != 3:
        return [None] * 6
    return list(np.concatenate((median_values, std_values)))


### (Extra) Feature 3 - Area of the bug in the mask ###

def bug_area(mask):
    return cv2.countNonZero(mask)

### Feature 4 - Ratio between the 2 longest orthogonal lines that can cross the bug ###

def ratio_longest_orthogonal_lines(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    max_ratio = 0
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        side_lengths = [np.linalg.norm(box[i] - box[(i+1) % 4]) for i in range(4)]
        side_lengths.sort(reverse=True)
        if side_lengths[3] != 0:
            ratio = side_lengths[2] / side_lengths[3]
            if ratio > max_ratio:
                max_ratio = ratio
    return float(max_ratio)

### (Extra) Feature 5 - Mean Intensity ###

def mean_intensity(image, mask):
    bug_pixels = cv2.bitwise_and(image, image, mask=mask)
    mean_intensity = np.mean(bug_pixels)
    
    return mean_intensity

### Feature 6 - Ratio of Bug Pixels to Total Pixels ###
def bug_to_total_ratio(mask):
    if mask is None:
        return 0
    total_pixels = mask.shape[0] * mask.shape[1]
    bug_pixels = cv2.countNonZero(mask)
    return bug_pixels / total_pixels if total_pixels > 0 else 0

### Feature 7 - Min, Max, Mean Values for Red, Green, and Blue within the Bug Mask ###
def min_max_mean_color_bug_mask(image, mask):
    if mask is None:
        return np.zeros(3), np.zeros(3), np.zeros(3)    
    image_isolated = cv2.bitwise_and(image, image, mask=mask)
    red_values = image_isolated[:, :, 0][mask != 0]
    green_values = image_isolated[:, :, 1][mask != 0]
    blue_values = image_isolated[:, :, 2][mask != 0]
    min_values = np.array([np.min(red_values), np.min(green_values), np.min(blue_values)])
    max_values = np.array([np.max(red_values), np.max(green_values), np.max(blue_values)])
    mean_values = np.array([np.mean(red_values), np.mean(green_values), np.mean(blue_values)])
    return min_values, max_values, mean_values


# Extracting dataset from excel file 
df = pd.read_excel('project_data/train/classif.xlsx', index_col=0, engine='openpyxl')

# Parsing files 
folder_path = 'project_data/train/'

# Main loop to extract features from each image and mask in the dataset
for i in range(1, max(len(os.listdir(folder_path + 'images')), len(os.listdir(folder_path + 'masks')))):
    image_path = os.path.join(folder_path, 'images', f'{i}.jpg')
    mask_path = os.path.join(folder_path, 'masks', f'binary_{i}.tif')

    try:
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    except FileNotFoundError:
        continue

    df.loc[df.index[i-1], 'sym_index'] = symmetry_index(image)
    df.loc[df.index[i-1], ['Median_R', 'Median_G', 'Median_B', 'Std_R', 'Std_G', 'Std_B']] = median_std(image, mask)
    df.loc[df.index[i-1], 'Area'] = bug_area(mask)
    df.loc[df.index[i-1], 'Orthogonal_Lines'] = ratio_longest_orthogonal_lines(mask)
    df.loc[df.index[i-1], 'Mean_Intensity'] = mean_intensity(image, mask)
    df.loc[df.index[i-1], 'Bug_to_Total_Ratio'] = bug_to_total_ratio(mask)
    min_values, max_values, mean_values = min_max_mean_color_bug_mask(image, mask)
    df.loc[df.index[i-1], ['Min_R_bug', 'Min_G_bug', 'Min_B_bug']] = min_values
    df.loc[df.index[i-1], ['Max_R_bug', 'Max_G_bug', 'Max_B_bug']] = max_values
    df.loc[df.index[i-1], ['Mean_R_bug', 'Mean_G_bug', 'Mean_B_bug']] = mean_values

# Exporting DataFrame to a CSV file
df.to_csv(r'project_data\dataVisualization\result.csv', index=False)

end_time = datetime.datetime.now()
print(f"Execution time: {end_time - start_time}")