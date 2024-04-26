import cv2
import numpy as np
import pandas as pd
import os
import datetime

# Used to compute execution time
start_time = datetime.datetime.now()

# Extracting dataset from excel file 
df = pd.read_excel('project_data/train/classif.xlsx', index_col=0)

# Computation of the mean and standard deviation 
# for the RGB channels of the bee pixels in the image.

folder_path = 'project_data/train/'

for i in range(1, 251):

    # Skipping image 154 as it does not have a mask
    if i == 154:
        continue

    image_path = os.path.join(folder_path, 'images', f'{i}.jpg')
    mask_path = os.path.join(folder_path, 'masks', f'binary_{i}.tif')

    image = cv2.imread(image_path)
    masque = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Isolate the part of the image corresponding to the mask 
    image_isolee = cv2.bitwise_and(image, image, mask=masque)

    # Get pixels corresponding to the bee 
    bee_pixels = image_isolee[masque != 0]

    median_values = np.median(bee_pixels, axis=0)
    std_values = np.std(bee_pixels, axis=0)

    R_median, G_median, B_median = median_values
    R_std, G_std, B_std = std_values

    # Add median values to dataframe
    df.loc[df.index[i-1], 'Median_R'] = R_median
    df.loc[df.index[i-1], 'Median_G'] = G_median
    df.loc[df.index[i-1], 'Median_B'] = B_median

    # Add standard deviation values to dataframe
    df.loc[df.index[i-1], 'Std_R'] = R_std
    df.loc[df.index[i-1], 'Std_G'] = G_std
    df.loc[df.index[i-1], 'Std_B'] = B_std

    # Computing area of each bug
    mask_area = cv2.countNonZero(masque)

    # Add area to dataframe
    df.loc[df.index[i-1], 'Area'] = mask_area


# Exporting DataFrame to a CSV file
df.to_csv('result.csv', index=False)

end_time = datetime.datetime.now()
print(f"Execution time: {end_time - start_time}")

