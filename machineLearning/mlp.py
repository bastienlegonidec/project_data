import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# Charger les données
df = pd.read_csv("project_data/dataVisualization/result.csv")
test_df = pd.read_csv("project_data/machineLearning/test_results.csv")

# Supprimer les valeurs manquantes
df.dropna(inplace=True)

# Supprimer les valeurs 'Bee & Bumblebee'
df = df[df['bug type'] != 'Bee & Bumblebee']

# Sélectionner les colonnes de caractéristiques et la cible
train_predictor_columns = df.columns.difference(['bug type', 'species'])
target_labels = df['bug type']
train_feats = df[train_predictor_columns]

# Encodage des étiquettes de la cible
y_encoded, y_labels = pd.factorize(target_labels)

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(train_feats, y_encoded, test_size=0.2, random_state=42)

# Normalisation des données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Définir la grille des paramètres
param_grid = {
    'hidden_layer_sizes': [(64, 32, 16), (128, 64, 32), (32, 16, 8)],
    'alpha': [0.001, 0.0001, 0.01],
    'max_iter': [2000]
}

# Instancier GridSearchCV
grid = GridSearchCV(MLPClassifier(solver='adam', random_state=0, tol=1e-9), param_grid, refit=True, verbose=2, n_jobs=-1)

# Entraîner le modèle
grid.fit(X_train_scaled, y_train)

# Meilleurs paramètres
print("Best parameters found: ", grid.best_params_)

# Prédiction avec le meilleur modèle
y_pred = grid.best_estimator_.predict(X_test_scaled)

# Évaluation de la performance
classification_rep = classification_report(y_test, y_pred, target_names=y_labels[np.unique(y_test)])
print("Classification Report:")
print(classification_rep)

# Isoler les lignes de result.csv qui ont été utilisées pour tester le modèle
test_df = df.loc[X_test.index]
test_df.drop(train_predictor_columns, axis=1, inplace=True)
test_df.drop(['species'], axis=1, inplace=True)

# Création du DataFrame avec les meilleurs résultats
results_df = pd.DataFrame({
    'bug type': y_labels[y_test],
    'Predicted_Bug_Type': y_labels[y_pred],
    'Recognition': pd.Series(y_labels[y_test]) == pd.Series(y_labels[y_pred])
})

# Convertir les booléens en chaînes "True" ou "False"
test_df['MLP_Recognition'] = results_df['Recognition'].map({True: 'True', False: 'False'})
test_df.to_csv("project_data/machineLearning/test_results.csv", index=False)


