### Bastien LE GONIDEC, Rawad NAHLE, Léonard GENDREL, Arthur TERISSE
### data processing



### III.1 Feature Extraction
###• Symmetry index [IG.2411]
import cv2
import numpy as np
import os

# Fonction pour calculer le symétrie index pour une seule image
def compute_symmetry_index(image_path):
    # Charger l'image en niveaux de gris
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Trouver les dimensions de l'image
    height, width = image.shape
    
    # Déterminer l'axe de symétrie (par exemple, axe vertical au centre de l'image)
    axis = width // 2
    
    # Diviser l'image en deux parties égales le long de l'axe
    left_half = image[:, :axis]
    right_half = image[:, axis:]
    
    # Inverser la moitié droite pour correspondre à la moitié gauche
    right_half_flipped = cv2.flip(right_half, 1)
    
    # Calculer la similarité entre les deux moitiés (vous pouvez utiliser différentes mesures)
    correlation = cv2.matchTemplate(left_half, right_half_flipped, cv2.TM_CCOEFF_NORMED)[0][0]
    
    # Calculer le symétrie index
    symmetry_index = correlation / 2 + 0.5  # Normalize la corrélation dans la plage [0, 1]
    
    return symmetry_index

# Dossier contenant les images
image_folder = "chemin/vers/votre/dossier/images/"

# Liste des chemins d'accès à toutes les images
image_files = [os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.endswith('.jpg')]

# Calculer le symétrie index pour toutes les images
symmetry_indices = []
for image_file in image_files:
    symmetry_index = compute_symmetry_index(image_file)
    symmetry_indices.append(symmetry_index)

# Afficher les symétrie index
for i, symmetry_index in enumerate(symmetry_indices):
    print(f"Symmetry index for image {i+1}: {symmetry_index}")


###• The ratio between the 2 longest orthogonal lines that can cross the bug (smallest divided by longuest) [IG.2411]

###• The ratio of the number of pixels of bug divided by the number of pixels of the full image

###• The min, max and mean values for Red, Green and Blue within the bug mask.

###• The median and standard deviation for the Red, Green and Blue within the bug mask [II.2413]

###• A least two other features of your choosing [II.2413], or at least four other features [IG.2411]. You may use features extracted inside or outside of the bug mask.