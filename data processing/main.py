### Bastien LE GONIDEC, Rawad NAHLE, Léonard GENDREL, Arthur TERISSE

### III.1 Feature Extraction

# Imports 

import numpy as np
import os
import cv2
import pandas as pd
import datetime

# Used to compute execution time
start_time = datetime.datetime.now()

# Extracting dataset from excel file 
df = pd.read_excel('project_data/train/classif.xlsx', index_col=0, engine='openpyxl')

# Parsing files 
folder_path = 'project_data/train/'

### Feature 1 - Symmetry index function

def symmetry_index(image):

    # Convertir l'image en niveaux de gris
    gray_array = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculer l'indice de symétrie
    half_width = gray_array.shape[1] // 2
    left_half = gray_array[:, :half_width]
    right_half = gray_array[:, half_width:]
    symmetry = np.sum(np.abs(left_half - np.flip(right_half, axis=1))) / np.prod(left_half.shape)

    return symmetry

### Feature 2 - The median and standard deviation for the Red, Green and Blue within the bug mask

def median_std(image, mask):
        
    # Isolate the part of the image corresponding to the mask 
    image_isolee = cv2.bitwise_and(image, image, mask=mask)

    # Get pixels corresponding to the bee 
    bee_pixels = image_isolee[mask != 0]

    median_values = np.median(bee_pixels, axis=0)
    std_values = np.std(bee_pixels, axis=0)

    # Ensure that median_values has exactly three values / case where mask does not exist
    if len(median_values) != 3:
        return None, None, None, None, None, None

    R_median, G_median, B_median = median_values
    R_std, G_std, B_std = std_values

    return R_median, G_median, B_median, R_std, G_std, B_std

### (Added) Feature 3 - Area of the bug

def bug_area(mask):
    return cv2.countNonZero(mask)

### Feature 4 - Ratio between the 2 longest orthogonal lines that can cross the bug (smallest divided by longuest) [IG.2411]

### Feature 5 - Ratio of the number of pixels of bug divided by the number of pixels of the full image

### Feature 6 - Min, max and mean values for Red, Green and Blue within the bug mask.

### (Added) Feature 7 - ...

# Main loop to extract features from each image and mask in the dataset
for i in range(1, max(len(os.listdir(folder_path + 'images')), len(os.listdir(folder_path + 'masks')))):
        
    image_path = os.path.join(folder_path, 'images', f'{i}.jpg')
    mask_path = os.path.join(folder_path, 'masks', f'binary_{i}.tif')

    try :
        image = cv2.imread(image_path)
        masque = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
    except FileNotFoundError:
        continue
        
    df.loc[df.index[i-1], 'sym_index'] = symmetry_index(image)

    R_median, G_median, B_median, R_std, G_std, B_std = median_std(image, masque)

    # Add median values to dataframe
    df.loc[df.index[i-1], 'Median_R'] = R_median
    df.loc[df.index[i-1], 'Median_G'] = G_median
    df.loc[df.index[i-1], 'Median_B'] = B_median

    # Add standard deviation values to dataframe
    df.loc[df.index[i-1], 'Std_R'] = R_std
    df.loc[df.index[i-1], 'Std_G'] = G_std
    df.loc[df.index[i-1], 'Std_B'] = B_std

    # Add area to dataframe
    df.loc[df.index[i-1], 'Area'] = bug_area(masque)
    
# Exporting DataFrame to a CSV file
df.to_csv('project_data\data visualization\result.csv', index=False)

end_time = datetime.datetime.now()
print(f"Execution time: {end_time - start_time}")
