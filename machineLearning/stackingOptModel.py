import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

# Load the data
df = pd.read_csv("project_data/dataVisualization/result.csv")

# Drop missing values
df.dropna(inplace=True)

# Combine butterfly, dragonfly, hoverfly, and wasp into "other"
df['bug type'] = df['bug type'].replace(['Butterfly', 'Dragonfly', 'Hoverfly', 'Wasp'], 'Other')

# Feature columns
train_predictor_columns = df.columns.difference(['bug type', 'species'])

# Target labels
target_labels = df['bug type']
train_feats = df[train_predictor_columns]

# Encode target labels
y_encoded, y_labels = pd.factorize(target_labels)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(train_feats, y_encoded, test_size=0.2, random_state=42)

# Normalize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SVM Model Training
param_grid_svm = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
    'degree': [3, 4, 5],
    'coef0': [0, 0.1, 0.5, 1]
}

grid_svm = GridSearchCV(SVC(), param_grid_svm, refit=True, verbose=2, n_jobs=-1)
grid_svm.fit(X_train_scaled, y_train)

# Best SVM model
svm_model = grid_svm.best_estimator_

# MLP Model Training
param_grid_mlp = {
    'hidden_layer_sizes': [(64, 32, 16), (128, 64, 32), (32, 16, 8)],
    'alpha': [0.001, 0.0001, 0.01],
    'max_iter': [2000]
}

grid_mlp = GridSearchCV(MLPClassifier(solver='adam', random_state=42, tol=1e-9), param_grid_mlp, refit=True, verbose=2, n_jobs=-1)
grid_mlp.fit(X_train_scaled, y_train)

# Best MLP model
mlp_model = grid_mlp.best_estimator_

# Stacking Classifier
estimators = [('svm', svm_model), ('mlp', mlp_model)]
final_estimator = LogisticRegression()

stacking_clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator)

# Train Stacking Classifier
stacking_clf.fit(X_train_scaled, y_train)

# Predict using Stacking Classifier
y_pred = stacking_clf.predict(X_test_scaled)

# Evaluate performance
print("Stacking Classifier Classification Report:")
print(classification_report(y_test, y_pred, target_names=y_labels))

# Best Test Accuracy
print(f'Best Test Accuracy: {accuracy_score(y_test, y_pred)}')

# Create the DataFrame with the best results
results_df = pd.DataFrame({
    'bug type': y_labels[y_test],
    'Predicted_Bug_Type': y_labels[y_pred],
    'Recognition': pd.Series(y_labels[y_test]) == pd.Series(y_labels[y_pred])
})

# Convert booleans to strings "True" or "False"
results_df['Recognition'] = results_df['Recognition'].map({True: 'True', False: 'False'})

# Save the DataFrame to a CSV file
results_df.to_csv('project_data/machineLearning/result.csv', index=False)

# Output results
print("Stacking classifier predictions have been saved to 'stacking_classifier_best_predicted_results.csv'.")