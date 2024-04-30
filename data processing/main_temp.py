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

# Boucle sur chaque fichier dans le répertoire
for i in range(1, max(len(os.listdir(folder_path + 'images')), len(os.listdir(folder_path + 'masks')))):
        
    image_path = os.path.join(folder_path, 'images', f'{i}.jpg')
    mask_path = os.path.join(folder_path, 'masks', f'binary_{i}.tif')

    try :
        image = cv2.imread(image_path)
        masque = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
    except FileNotFoundError:
        continue
        
    df.loc[df.index[i-1], 'sym_index'] = symmetry_index(image)    
    


###• The ratio between the 2 longest orthogonal lines that can cross the bug (smallest divided by longuest) [IG.2411]

###• The ratio of the number of pixels of bug divided by the number of pixels of the full image

###• The min, max and mean values for Red, Green and Blue within the bug mask.

###• The median and standard deviation for the Red, Green and Blue within the bug mask [II.2413]

###• A least two other features of your choosing [II.2413], or at least four other features [IG.2411]. You may use features extracted inside or outside of the bug mask.