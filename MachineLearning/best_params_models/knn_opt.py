import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Charger les données
df = pd.read_csv("project_data/dataVisualization/result.csv")

# Supprimer les valeurs manquantes
df = df.dropna()

# Combine butterfly, dragonfly, hoverfly, and wasp into "other"
df['bug type'] = df['bug type'].replace(['Butterfly', 'Dragonfly', 'Hover fly', 'Wasp'], 'Other')

# Sélectionner les colonnes de caractéristiques et la cible
train_predictor_columns = ["Median_R","Median_G","Median_B","Std_R","Std_G","Std_B","Area"]
target_labels = df["bug type"]
train_feats = df[train_predictor_columns]

# Division des données en ensembles d'entraînement et de test
x_train, x_test, y_train, y_test = train_test_split(train_feats, target_labels, test_size=0.2, random_state=0)

# Appliquer KNN avec les meilleurs paramètres trouvés
knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', metric='euclidean')
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)

# Isoler les lignes de result.csv qui ont été utilisées pour tester le modèle
test_df = df.loc[x_test.index]
test_df.drop(train_predictor_columns, axis=1, inplace=True)
test_df.drop(['species'], axis=1, inplace=True)

# Évaluation de la performance
classification_report = metrics.classification_report(y_test, y_pred)
print("Classification Report:")
print(classification_report)

# Export des résultats
test_df['KNN'] = y_pred
test_df['KNN_Recognition'] = test_df['bug type'] == test_df['KNN']
test_df.to_csv("project_data/machineLearning/test_results.csv", index=False)
