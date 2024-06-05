import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Chargement des données
df = pd.read_csv('project_data/dataVisualization/result.csv')

# Suppression des valeurs 'Bee & Bumblebee'
df = df[df['bug type'] != 'Bee & Bumblebee']
df['bug type'] = df['bug type'].replace(['Butterfly', 'Dragonfly', 'Hover fly', 'Wasp'], 'Other')

# Sélection des caractéristiques et de la cible
X = df.drop(columns=['bug type', 'species'])
y = df['bug type']

# Encodage des étiquettes de la cible
y_encoded, y_labels = pd.factorize(y)

# Imputation des valeurs manquantes
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Normalisation des données
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Définir les meilleurs paramètres pour l'auto-encodeur et l'encodeur
autoencoder_params = {
    'hidden_layer_sizes': (128, 64, 128),
    'max_iter': 1000,
    'alpha': 0.0001
}

encoder_params = {
    'hidden_layer_sizes': (64, 32),
    'max_iter': 1000,
    'alpha': 0.0001
}

# Entraîner l'auto-encodeur
autoencoder = MLPRegressor(random_state=42, **autoencoder_params)
autoencoder.fit(X_train, X_train)

# Entraîner l'encodeur
encoder = MLPRegressor(random_state=42, **encoder_params)
encoder.fit(X_train, X_train)

# Extraction des caractéristiques encodées
X_train_encoded = encoder.predict(X_train)
X_test_encoded = encoder.predict(X_test)

# Modèle de régression logistique
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_encoded, y_train)

# Prédiction et évaluation
y_pred = clf.predict(X_test_encoded)

# Map predicted labels to original species
y_test_labels = y_labels[y_test]
y_pred_labels = y_labels[y_pred]

# Création du DataFrame avec les colonnes requises
results_df = pd.DataFrame({
    'bug type': y_test_labels,
    'Predicted_Bug_Type': y_pred_labels,
    'Recognition': y_test_labels == y_pred_labels
})

# Convertir les booléens en chaînes "True" ou "False"
results_df['Recognition'] = results_df['Recognition'].map({True: 'True', False: 'False'})

# Enregistrement du DataFrame dans un fichier CSV
results_df.to_csv('predicted_results.csv', index=False)

# Affichage d'un message de confirmation
print("Le fichier CSV a été enregistré sous le nom 'predicted_results.csv'.")

# Évaluation du modèle
accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {accuracy}')
print('\nClassification Report:')
print(classification_report(y_test, y_pred, target_names=y_labels[np.unique(y_test)]))

# Return the results DataFrame for display
results_df.head()