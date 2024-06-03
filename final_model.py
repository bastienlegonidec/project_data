import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

def process_bug_data(file_path, output_path):
    # Charger les données
    df = pd.read_csv(file_path)

    # Supprimer les valeurs manquantes
    df.dropna(inplace=True)

    # Supprimer les valeurs 'Bee & Bumblebee'
    df = df[df['bug type'] != 'Bee & Bumblebee']

    # Combine butterfly, dragonfly, hoverfly, and wasp into "other"
    df['bug type'] = df['bug type'].replace(['Butterfly', 'Dragonfly', 'Hover fly', 'Wasp'], 'Other')

    # Ajouter des caractéristiques supplémentaires si nécessaire
    # df['Color_Ratio'] = df['Median_R'] / (df['Median_G'] + df['Median_B'])  # Exemple de caractéristique supplémentaire

    # Sélectionner les colonnes de caractéristiques et la cible
    train_predictor_columns = df.columns.difference(['bug type', 'species'])
    target_labels = df['bug type']
    train_feats = df[train_predictor_columns]

    # Encodage des étiquettes de la cible
    y_encoded, y_labels = pd.factorize(target_labels)

    # Division des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(train_feats, y_encoded, test_size=0.2, random_state=42)

    # Suréchantillonnage des classes minoritaires
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    # Normalisation des données
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)

    # Définir les modèles de base
    mlp = MLPClassifier(hidden_layer_sizes=(32, 16, 8), alpha=0.001, max_iter=5000, solver='adam', random_state=0, tol=1e-9)
    svm = SVC(C=100, kernel='linear', degree=3, gamma='scale', coef0=0, probability=True)

    # Créer un classifieur de vote avec les modèles MLP et SVM
    voting_clf = VotingClassifier(estimators=[('mlp', mlp), ('svm', svm)], voting='soft')

    # Entraîner le classifieur de vote
    voting_clf.fit(X_train_scaled, y_train_res)

    # Prédiction avec le classifieur de vote
    y_pred = voting_clf.predict(X_test_scaled)

    # Évaluation de la performance
    labels = np.unique(y_test)
    classification_rep = classification_report(y_test, y_pred, labels=labels, target_names=y_labels[labels])
    print("Classification Report:")
    print(classification_rep)

    # Création du DataFrame avec les résultats
    results_df = pd.DataFrame({
        'Bug_type': y_labels[y_test],
        'Recognition': pd.Series(y_labels[y_test]),
        'Predicted_Bug': y_labels[y_pred],
        'True_or_False': (pd.Series(y_labels[y_test]) == pd.Series(y_labels[y_pred])).map({True: 'True', False: 'False'})
    })

    # Sélectionner les colonnes nécessaires
    results_df = results_df[['Bug_type', 'Recognition', 'Predicted_Bug', 'True_or_False']]

    # Enregistrer le DataFrame dans un fichier CSV
    results_df.to_csv(output_path, index=False)

# Exemple d'appel de la fonction
process_bug_data("dataVisualization/result.csv", "stacking_recognition.csv")
