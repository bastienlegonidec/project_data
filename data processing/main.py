### Bastien LE GONIDEC, Rawad NAHLE, Léonard GENDREL, Arthur TERISSE
### data processing



### III.1 Feature Extraction
###• Symmetry index [IG.2411]
from PIL import Image
import numpy as np
import os

def symmetry_index(image):
    # Convertir l'image en niveaux de gris
    gray = image.convert('L')
    gray_array = np.array(gray)

    # Calculer l'indice de symétrie
    half_width = gray_array.shape[1] // 2
    left_half = gray_array[:, :half_width]
    right_half = gray_array[:, half_width:]
    symmetry = np.sum(np.abs(left_half - np.flip(right_half, axis=1))) / np.prod(left_half.shape)

    return symmetry

# Chemin du répertoire contenant les images
directory = "/workspaces/projects/images"

# Liste des fichiers dans le répertoire
files = os.listdir(directory)

# Dictionnaire pour stocker les résultats
symmetry_results = {}

# Boucle sur chaque fichier dans le répertoire
for file in files:
    # Vérifier si le fichier est une image
    if file.endswith(".JPG") or file.endswith(".png") or file.endswith(".jpeg"):
        # Chemin complet de l'image
        image_path = os.path.join(directory, file)
        try:
            # Charger l'image
            with Image.open(image_path) as img:
                # Calculer l'indice de symétrie
                symmetry = symmetry_index(img)
                # Ajouter le résultat au dictionnaire
                symmetry_results[file] = symmetry
        except Exception as e:
            print(f"Erreur lors du traitement de l'image {image_path}: {e}")

# Afficher le dictionnaire des résultats
print("Résultats de l'indice de symétrie pour chaque image :")
for image, symmetry in symmetry_results.items():
    print(f"{image}: {symmetry}")




###• The ratio between the 2 longest orthogonal lines that can cross the bug (smallest divided by longuest) [IG.2411]

###• The ratio of the number of pixels of bug divided by the number of pixels of the full image

###• The min, max and mean values for Red, Green and Blue within the bug mask.

###• The median and standard deviation for the Red, Green and Blue within the bug mask [II.2413]

###• A least two other features of your choosing [II.2413], or at least four other features [IG.2411]. You may use features extracted inside or outside of the bug mask.