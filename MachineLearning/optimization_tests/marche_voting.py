import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

# Charger les données
df = pd.read_csv("project_data/dataVisualization/result.csv")

# Supprimer les valeurs manquantes
df = df.dropna()

# Sélectionner les colonnes de caractéristiques et la cible
train_predictor_columns = ["Median_R", "Median_G", "Median_B", "Std_R", "Std_G", "Std_B", "Area"]
target_labels = df["bug type"]
train_feats = df[train_predictor_columns]

# Division des données en ensembles d'entraînement et de test
x_train, x_test, y_train, y_test = train_test_split(train_feats, target_labels, test_size=0.2, random_state=0)

# Scaling the features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Appliquer SVM avec les meilleurs paramètres trouvés
svm = SVC(C=100, kernel='linear', degree=3, gamma='scale', coef0=0, probability=True)
svm.fit(x_train_scaled, y_train)

# Sélectionner les colonnes de caractéristiques et la cible pour MLP
train_predictor_columns_mlp = df.columns.difference(['bug type', 'species'])
target_labels_mlp = df['bug type']
train_feats_mlp = df[train_predictor_columns_mlp]

# Encodage des étiquettes de la cible
y_encoded, y_labels = pd.factorize(target_labels_mlp)

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train_mlp, y_test_mlp = train_test_split(train_feats_mlp, y_encoded, test_size=0.2, random_state=42)

# Normalisation des données
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Définir les meilleurs paramètres trouvés pour MLP
best_params = {
    'hidden_layer_sizes': (32, 16, 8),
    'alpha': 0.001,
    'max_iter': 2000
}

# Instancier le modèle MLP avec les meilleurs paramètres
mlp = MLPClassifier(solver='adam', random_state=0, tol=1e-9, **best_params)
mlp.fit(X_train_scaled, y_train_mlp)

# Combine SVM and MLP using a Voting Classifier
voting_clf = VotingClassifier(estimators=[('svm', svm), ('mlp', mlp)], voting='soft')
voting_clf.fit(x_train_scaled, y_train)

# Predict using the voting classifier
y_pred = voting_clf.predict(x_test_scaled)

# Isoler les lignes de result.csv qui ont été utilisées pour tester le modèle
test_df = df.loc[x_test.index]
test_df.drop(train_predictor_columns, axis=1, inplace=True)
test_df.drop(['sym_index', 'species'], axis=1, inplace=True)

# Évaluation de la performance
classification_report = metrics.classification_report(y_test, y_pred)
print("Classification Report:")
print(classification_report)

# Export des résultats
test_df['Voting'] = y_pred
test_df['Voting_Recognition'] = test_df['bug type'] == test_df['Voting']
test_df.to_csv("project_data/machineLearning/optimized_algorithm/voting_results.csv", index=False)
