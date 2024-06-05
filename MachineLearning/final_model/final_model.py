import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from imblearn.over_sampling import SMOTE

def train_and_test_model(train_features, test_features):
    # Load training labels
    train_df = pd.read_excel('train/classif.xlsx', index_col=0, engine='openpyxl')

    # Merge features with labels
    train_df = train_df.merge(train_features, left_index=True, right_on='ID')

    # Prepare training data
    train_predictor_columns = train_df.columns.difference(['bug type', 'species', 'ID'])
    train_feats = train_df[train_predictor_columns]
    target_labels = train_df['bug type']
    y_encoded, y_labels = pd.factorize(target_labels)

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    train_feats_imputed = imputer.fit_transform(train_feats)

    # Vérifier la distribution des classes
    class_distribution = np.bincount(y_encoded)
    print("Distribution des classes dans l'ensemble d'entraînement :", class_distribution)

    # Filtrer les classes ayant moins de 2 échantillons
    min_samples = 2
    indices_to_keep = [i for i, count in enumerate(class_distribution) if count >= min_samples]
    filtered_indices = [i for i in range(len(y_encoded)) if y_encoded[i] in indices_to_keep]
    X_filtered = train_feats_imputed[filtered_indices]
    y_filtered = y_encoded[filtered_indices]

    # Recalculer la distribution des classes après filtrage
    filtered_class_distribution = np.bincount(y_filtered)
    print("Distribution des classes après filtrage :", filtered_class_distribution)

    # Split data for training and validation
    X_train, X_val, y_train, y_val = train_test_split(X_filtered, y_filtered, test_size=0.2, random_state=42)

    # Handle imbalanced classes with SMOTE
    # Adjust n_neighbors to be less than the number of samples in the smallest class
    min_class_samples = np.min(np.bincount(y_train))
    n_neighbors = min(5, min_class_samples - 1)

    sm = SMOTE(sampling_strategy='auto', k_neighbors=max(1, n_neighbors), random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    # Normalize data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_val_scaled = scaler.transform(X_val)

    # Define classifiers
    mlp = MLPClassifier(hidden_layer_sizes=(32, 16, 8), alpha=0.001, max_iter=5000, solver='adam', random_state=0, tol=1e-9)
    svm = SVC(C=100, kernel='linear', degree=3, gamma='scale', coef0=0, probability=True)

    # Voting classifier
    voting_clf = VotingClassifier(estimators=[('mlp', mlp), ('svm', svm)], voting='soft')

    # Train classifier
    voting_clf.fit(X_train_scaled, y_train_res)

    # Prepare test data
    test_predictor_columns = test_features.columns.difference(['ID'])
    test_feats = test_features[test_predictor_columns]

    # Impute missing values in test data
    test_feats_imputed = imputer.transform(test_feats)

    # Normalize test data
    X_test_scaled = scaler.transform(test_feats_imputed)

    # Predict using the trained model
    y_test_pred = voting_clf.predict(X_test_scaled)
    y_test_pred_labels = y_labels[y_test_pred]

    # Prepare output DataFrame
    output_df = pd.DataFrame({'ID': test_features['ID'], 'bug type': y_test_pred_labels})

    # Save output to CSV
    output_df.to_csv('test_results.csv', index=False)
