import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Charger les données
df = pd.read_csv("project_data/dataVisualization/result.csv")
test_df = pd.read_csv("project_data/machineLearning/test_results.csv")

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

# Define the parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
    'degree': [3, 4, 5],  # Only relevant for 'poly' kernel
    'coef0': [0, 0.1, 0.5, 1]  # Only relevant for 'poly' and 'sigmoid' kernels
}

# Instantiate GridSearchCV
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, n_jobs=-1)

# Fit the model
grid.fit(x_train_scaled, y_train)

# Best parameters
print("Best parameters found: ", grid.best_params_)

# Predict using the best model
y_pred = grid.best_estimator_.predict(x_test_scaled)

# Isoler les lignes de result.csv qui ont été utilisées pour tester le modèle
test_df = df.loc[x_test.index]
test_df.drop(train_predictor_columns, axis=1, inplace=True)
test_df.drop(['sym_index', 'species'], axis=1, inplace=True)

# Évaluation de la performance
classification_report = metrics.classification_report(y_test, y_pred)
print("Classification Report:")
print(classification_report)

# Export des résultats
test_df['SVM'] = y_pred
test_df['SVM_Recognition'] = test_df['bug type'] == test_df['SVM']
test_df.to_csv("project_data/machineLearning/test_results.csv", index=False)
