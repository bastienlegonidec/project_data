import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report

def process_bug_data(train_features_csv, test_features_csv, output_path):
    # Charger les données de train et test
    df_train = pd.read_csv(train_features_csv)
    df_test = pd.read_csv(test_features_csv)

    # Supprimer les valeurs manquantes
    df_train.dropna(inplace=True)
    df_test.dropna(inplace=True)

    # Supprimer les valeurs 'Bee & Bumblebee'
    df_train = df_train[df_train['bug type'] != 'Bee & Bumblebee']

    # Combine butterfly, dragonfly, hoverfly, and wasp into "other"
    df_train['bug type'] = df_train['bug type'].replace(['Butterfly', 'Dragonfly', 'Hover fly', 'Wasp'], 'Other')

    # Sélectionner les colonnes de caractéristiques et la cible
    train_predictor_columns = df_train.columns.difference(['bug type', 'species', 'image_id'])
    target_labels = df_train['bug type']
    train_feats = df_train[train_predictor_columns]
    test_feats = df_test[train_predictor_columns]

    # Encodage des étiquettes de la cible
    y_encoded, y_labels = pd.factorize(target_labels)

    # Division des données en ensembles d'entraînement et de test
    X_train, X_val, y_train, y_val = train_test_split(train_feats, y_encoded, test_size=0.2, random_state=42)

    # Suréchantillonnage des classes minoritaires
    #sm = SMOTE(random_state=42)
    #X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    # Normalisation des données
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(test_feats)

    # Définir les modèles de base
    mlp = MLPClassifier(hidden_layer_sizes=(32, 16, 8), alpha=0.001, max_iter=5000, solver='adam', random_state=0, tol=1e-9)
    svm = SVC(C=100, kernel='linear', degree=3, gamma='scale', coef0=0, probability=True)

    # Créer un classifieur de vote avec les modèles MLP et SVM
    voting_clf = VotingClassifier(estimators=[('mlp', mlp), ('svm', svm)], voting='soft')

    # Entraîner le classifieur de vote
    voting_clf.fit(X_train_scaled, y_train_res)

    # Prédiction avec le classifieur de vote
    y_val_pred = voting_clf.predict(X_val_scaled)
    y_test_pred = voting_clf.predict(X_test_scaled)

    # Évaluation de la performance sur les données de validation
    labels_val = np.unique(y_val)
    classification_rep_val = classification_report(y_val, y_val_pred, labels=labels_val, target_names=y_labels[labels_val])
    print("Classification Report on Validation Data:")
    print(classification_rep_val)

    # Prédiction avec le classifieur de vote sur les données de test
    y_test_pred = voting_clf.predict(X_test_scaled)

    # Création du DataFrame avec les résultats sur les données de test
    results_df = pd.DataFrame({
        'Image_ID': df_test['image_id'],
        'Predicted_Bug_Type': y_labels[y_test_pred]
    })

    # Enregistrer le DataFrame dans un fichier CSV
    results_df.to_csv(output_path, index=False)

    print(f"Predictions saved to {output_path}")
