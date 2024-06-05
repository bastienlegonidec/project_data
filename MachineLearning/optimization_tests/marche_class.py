import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# Charger les données
df = pd.read_csv("project_data/dataVisualization/result.csv")

# Supprimer les valeurs manquantes
df = df.dropna()

# Reclassify classes to "Other" except "Bee" and "Bumblebee"
df['bug type'] = df['bug type'].apply(lambda x: x if x in ['Bee', 'Bumblebee'] else 'Other')

# Sélectionner les colonnes de caractéristiques et la cible
train_predictor_columns = ["Orthogonal_Lines","Mean_Intensity","Bug_to_Total_Ratio","Min_R_bug","Min_G_bug","Min_B_bug","Max_R_bug","Max_G_bug","Max_B_bug","Mean_R_bug","Mean_G_bug","Mean_B_bug"]
target_labels = df["bug type"]
train_feats = df[train_predictor_columns]

# Division des données en ensembles d'entraînement et de test
x_train, x_test, y_train, y_test = train_test_split(train_feats, target_labels, test_size=0.2, random_state=0)

# Scaling the features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Define the MLP model with the best parameters found for "Bee"
best_params_mlp = {
    'hidden_layer_sizes': (32, 16, 8),
    'alpha': 0.001,
    'max_iter': 2000
}
mlp = MLPClassifier(solver='adam', random_state=0, tol=1e-9, **best_params_mlp)

# Train the MLP model
mlp.fit(x_train_scaled, y_train == 'Bee')

# Define the SVM model with the best parameters found for "Bumblebee"
svm = SVC(C=100, kernel='linear', degree=3, gamma='scale', coef0=0, probability=True)

# Train the SVM model
svm.fit(x_train_scaled, y_train == 'Bumblebee')

# Predict using both models
mlp_pred = mlp.predict(x_test_scaled)
svm_pred = svm.predict(x_test_scaled)

# Combine predictions
y_pred = []
for mlp_p, svm_p in zip(mlp_pred, svm_pred):
    if mlp_p:
        y_pred.append('Bee')
    elif svm_p:
        y_pred.append('Bumblebee')
    else:
        y_pred.append('Other')

# Convert to a Series for evaluation
y_pred = pd.Series(y_pred, index=x_test.index)

# Isoler les lignes de result.csv qui ont été utilisées pour tester le modèle
test_df = df.loc[x_test.index]
test_df.drop(train_predictor_columns, axis=1, inplace=True)
test_df.drop(['sym_index', 'species'], axis=1, inplace=True)

# Évaluation de la performance
classification_report = metrics.classification_report(y_test, y_pred)
print("Classification Report:")
print(classification_report)

# Export des résultats
test_df['Hybrid_Prediction'] = y_pred
test_df['Hybrid_Recognition'] = test_df['bug type'] == test_df['Hybrid_Prediction']
test_df.to_csv("project_data/machineLearning/hybrid_results.csv", index=False)
