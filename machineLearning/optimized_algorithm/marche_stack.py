import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

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

# Define the SVM model with the best parameters found
svm = SVC(C=100, kernel='linear', degree=3, gamma='scale', coef0=0, probability=True)

# Define the MLP model with the best parameters found
best_params = {
    'hidden_layer_sizes': (32, 16, 8),
    'alpha': 0.001,
    'max_iter': 2000
}
mlp = MLPClassifier(solver='adam', random_state=0, tol=1e-9, **best_params)

# Create a stacking classifier
estimators = [
    ('svm', svm),
    ('mlp', mlp)
]
stacking_clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

# Fit the stacking classifier
stacking_clf.fit(x_train_scaled, y_train)

# Predict using the stacking classifier
y_pred = stacking_clf.predict(x_test_scaled)

# Isoler les lignes de result.csv qui ont été utilisées pour tester le modèle
test_df = df.loc[x_test.index]
test_df.drop(train_predictor_columns, axis=1, inplace=True)
test_df.drop(['sym_index', 'species'], axis=1, inplace=True)

# Évaluation de la performance
classification_report = metrics.classification_report(y_test, y_pred)
print("Classification Report:")
print(classification_report)

# Export des résultats
test_df['Stacking'] = y_pred
test_df['Stacking_Recognition'] = test_df['bug type'] == test_df['Stacking']
test_df.to_csv("project_data/machineLearning/optimized_algorithm/stacking_results.csv", index=False)
