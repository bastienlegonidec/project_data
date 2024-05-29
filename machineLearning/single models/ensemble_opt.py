# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Chargement des données
df = pd.read_csv('project_data/dataVisualization/result.csv')

# Suppression des valeurs 'Bee & Bumblebee'
df = df[df['bug type'] != 'Bee & Bumblebee']

# Sélection des caractéristiques et de la cible
X = df.drop(columns=['bug type', 'species'])
y = df['bug type']

# Encodage des étiquettes de la cible
y_encoded, y_labels = pd.factorize(y)

# Imputation des valeurs manquantes
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Normalisation des données
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Définir les meilleurs paramètres trouvés
best_params = {
    'n_estimators': 200,
    'max_depth': 10,
    'criterion': 'gini',
    'bootstrap': True,
    'oob_score': False,
    'warm_start': False,
    'class_weight': None
}

# Création du modèle Random Forest avec les meilleurs paramètres
rf = RandomForestClassifier(random_state=42, **best_params)

# Entraînement du modèle
rf.fit(X_train, y_train)

# Prédiction sur les données de test
y_pred = rf.predict(X_test)

# Évaluation du modèle
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, target_names=y_labels[np.unique(y_test)])

# Création du DataFrame avec les meilleurs résultats
results_df = pd.DataFrame({
    'bug type': y_labels[y_test],
    'Predicted_Bug_Type': y_labels[y_pred],
    'Recognition': y_labels[y_test] == y_labels[y_pred]
})

# Convertir les booléens en chaînes "True" ou "False"
results_df['Recognition'] = results_df['Recognition'].map({True: 'True', False: 'False'})

# Enregistrement du DataFrame dans un fichier CSV
results_df.to_csv('rf_best_predicted_results.csv', index=False)

# Affichage d'un message de confirmation
print("Le fichier CSV a été enregistré sous le nom 'rf_best_predicted_results.csv'.")

# Affichage des meilleurs résultats
print(f'Best Test Accuracy: {accuracy}')
print('Best Parameters:')
print(f'n_estimators: {best_params["n_estimators"]}')
print(f'max_depth: {best_params["max_depth"]}')
print('\nBest Classification Report:')
print(classification_rep)