# Import Required Libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import re
import joblib
import os

# Step 1: Define Directory for Saving Models
save_dir = 'models/'
os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist

# Step 2: Load and Preprocess Data
def preprocess_text(text):
    """Remove special characters and lowercase text."""
    if pd.isna(text):
        return " "
    text = re.sub(r'[^a-zA-Z0-9\s]', '', str(text).lower())
    return text

# Load datasets
hpv_data = pd.read_excel('HPV_corpus.xlsx')
papd_data = pd.read_excel('PAPD_corpus.xlsx')

# Diagnose Label column
print("HPV Label unique values before mapping:", hpv_data['Label'].unique())
print("PAPD Label unique values before mapping:", papd_data['Label'].unique())

# Convert 'Label' column to binary (1 for 'Include', 0 for 'Exclude') and handle invalid entries
for df in [hpv_data, papd_data]:
    df['Label'] = df['Label'].map({'Include': 1, 'include': 1, 'Exclude': 0})
    initial_rows = len(df)
    df.dropna(subset=['Label'], inplace=True)
    print(f"Dropped {initial_rows - len(df)} rows with NaN labels from {df.name if hasattr(df, 'name') else 'dataset'}")

# Name datasets for printing
hpv_data.name = 'HPV'
papd_data.name = 'PAPD'

# Check labels after mapping
print("HPV Label unique values after mapping:", hpv_data['Label'].unique())
print("PAPD Label unique values after mapping:", papd_data['Label'].unique())

# Preprocess and combine features
for df in [hpv_data, papd_data]:
    df['title_abstract'] = (df['Title'].apply(preprocess_text) + ' ' +
                            df['Abstract'].apply(preprocess_text))
    df['all_features'] = (df['title_abstract'] + ' ' +
                          df['MeSH Terms'].fillna('').apply(preprocess_text) + ' ' +
                          df['Authors'].fillna('').apply(preprocess_text) + ' ' +
                          df['Keywords'].fillna('').apply(preprocess_text) + ' ' +
                          df['Journal'].fillna('').apply(preprocess_text) + ' ' +
                          df['Publication Type'].fillna('').apply(preprocess_text))

# Step 3: Feature Extraction
tfidf = TfidfVectorizer(stop_words='english')  # No max_features limit

# HPV features
X_hpv_baseline = tfidf.fit_transform(hpv_data['title_abstract'])
X_hpv_all = tfidf.fit_transform(hpv_data['all_features'])
y_hpv = hpv_data['Label']

# PAPD features
X_papd_baseline = tfidf.fit_transform(papd_data['title_abstract'])
X_papd_all = tfidf.fit_transform(papd_data['all_features'])
y_papd = papd_data['Label']

# Step 4: Define Models with Paper's Hyperparameters
models = {
    'XGBoost': XGBClassifier(max_depth=3, n_estimators=200, learning_rate=0.5, eval_metric='logloss'),
    'SVM': SVC(C=100, gamma=0.005, kernel='rbf'),
    'Logistic Regression': LogisticRegression(C=5, penalty='l2', max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=14)
}

# Step 5: Split Data into 8:1:1 (Train:Validation:Test)
def split_data(X, y):
    """Split data into 80% training, 10% validation, and 10% testing."""
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)  # 80% train
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  # 10% val, 10% test
    return X_train, X_val, X_test, y_train, y_val, y_test


# Step 6: Model Evaluation and Saving
def evaluate_and_save_model(model, X_train, X_val, X_test, y_train, y_val, y_test, model_name, dataset, feature_set):
    """Train, evaluate, and save models with validation and test sets."""
    # Train the model on training set
    model.fit(X_train, y_train)

    # Evaluate on validation set
    y_val_pred = model.predict(X_val)
    val_results = {
        'Accuracy': accuracy_score(y_val, y_val_pred),
        'Precision': precision_score(y_val, y_val_pred),
        'Recall': recall_score(y_val, y_val_pred),
        'F1': f1_score(y_val, y_val_pred)
    }

    print(f"\n{model_name} - {dataset} ({feature_set}) - Validation Results:")
    for metric, value in val_results.items():
        print(f"{metric}: {value:.2f}")

    # Evaluate on test set
    y_test_pred = model.predict(X_test)
    test_results = {
        'Accuracy': accuracy_score(y_test, y_test_pred),
        'Precision': precision_score(y_test, y_test_pred),
        'Recall': recall_score(y_test, y_test_pred),
        'F1': f1_score(y_test, y_test_pred)
    }

    print(f"\n{model_name} - {dataset} ({feature_set}) - Test Results:")
    for metric, value in test_results.items():
        print(f"{metric}: {value:.2f}")

    # Save the trained model
    filename = f"{save_dir}{model_name}_{dataset}_{feature_set.replace(' ', '_')}.model"
    if model_name == 'XGBoost':
        model.save_model(filename)  # XGBoost has a specific save method
    else:
        joblib.dump(model, filename)  # Save scikit-learn models with joblib

    # Confirm if the file is saved
    if os.path.exists(filename):
        print(f"Model successfully saved to: {filename}")
    else:
        print(f"Error: Model not saved to {filename}")

# Step 7: Train, Evaluate, and Save Models
datasets = {'HPV': (X_hpv_baseline, X_hpv_all, y_hpv), 'PAPD': (X_papd_baseline, X_papd_all, y_papd)}

for dataset_name, (X_baseline, X_all, y) in datasets.items():
    for model_name, model in models.items():
        # Split baseline features
        X_train_baseline, X_val_baseline, X_test_baseline, y_train_baseline, y_val_baseline, y_test_baseline = split_data(X_baseline, y)
        # Evaluate on baseline features
        evaluate_and_save_model(model, X_train_baseline, X_val_baseline, X_test_baseline, y_train_baseline, y_val_baseline, y_test_baseline, model_name, dataset_name, 'Baseline')

        # Split all features
        X_train_all, X_val_all, X_test_all, y_train_all, y_val_all, y_test_all = split_data(X_all, y)
        # Evaluate on all features
        evaluate_and_save_model(model, X_train_all, X_val_all, X_test_all, y_train_all, y_val_all, y_test_all, model_name, dataset_name, 'All Features')

# Step 8: Save Results to a Text File
results_file = os.path.join(save_dir, "model_evaluation_results.txt")

with open(results_file, "w") as f:
    f.write("Model Evaluation Results\n")
    f.write("=" * 50 + "\n")

    for dataset_name, (X_baseline, X_all, y) in datasets.items():
        for model_name, model in models.items():
            # Split baseline features
            X_train_baseline, X_val_baseline, X_test_baseline, y_train_baseline, y_val_baseline, y_test_baseline = split_data(X_baseline, y)

            model.fit(X_train_baseline, y_train_baseline)

            y_val_pred_baseline = model.predict(X_val_baseline)
            y_test_pred_baseline = model.predict(X_test_baseline)

            f.write(f"\n{model_name} - {dataset_name} (Baseline):\n")
            f.write(f"Validation Accuracy: {accuracy_score(y_val_baseline, y_val_pred_baseline):.2f}\n")
            f.write(f"Test Accuracy: {accuracy_score(y_test_baseline, y_test_pred_baseline):.2f}\n")

            f.write(f"Validation Precision: {precision_score(y_val_baseline, y_val_pred_baseline):.2f}\n")
            f.write(f"Test Precision: {precision_score(y_test_baseline, y_test_pred_baseline):.2f}\n")
            f.write(f"Validation Recall: {recall_score(y_val_baseline, y_val_pred_baseline):.2f}\n")
            f.write(f"Test Recall: {recall_score(y_test_baseline, y_test_pred_baseline):.2f}\n")
            f.write(f"Validation F1 Score: {f1_score(y_val_baseline, y_val_pred_baseline):.2f}\n")
            f.write(f"Test F1 Score: {f1_score(y_test_baseline, y_test_pred_baseline):.2f}\n")


            f.write(f"Validation F1 Score: {f1_score(y_val_baseline, y_val_pred_baseline):.2f}\n")
            f.write(f"Test F1 Score: {f1_score(y_test_baseline, y_test_pred_baseline):.2f}\n")

            f.write("-" * 50 + "\n")

            # Split all features
            X_train_all, X_val_all, X_test_all, y_train_all, y_val_all, y_test_all = split_data(X_all, y)

            model.fit(X_train_all, y_train_all)

            y_val_pred_all = model.predict(X_val_all)
            y_test_pred_all = model.predict(X_test_all)

            f.write(f"\n{model_name} - {dataset_name} (All Features):\n")
            f.write(f"Validation Accuracy: {accuracy_score(y_val_all, y_val_pred_all):.2f}\n")
            f.write(f"Test Accuracy: {accuracy_score(y_test_all, y_test_pred_all):.2f}\n")
            f.write(f"Validation F1 Score: {f1_score(y_val_all, y_val_pred_all):.2f}\n")
            f.write(f"Test F1 Score: {f1_score(y_test_all, y_test_pred_all):.2f}\n")
            f.write("-" * 50 + "\n")

print(f"Results saved to: {results_file}")

# Step 9: Save Results to a Tabular CSV File
csv_results_file = os.path.join(save_dir, "model_evaluation_results.csv")
results_data = []

# Prepare results for CSV
for dataset_name, (X_baseline, X_all, y) in datasets.items():
    for model_name, model in models.items():
        # Split baseline features
        X_train_baseline, X_val_baseline, X_test_baseline, y_train_baseline, y_val_baseline, y_test_baseline = split_data(X_baseline, y)
        
        model.fit(X_train_baseline, y_train_baseline)
        y_val_pred_baseline = model.predict(X_val_baseline)
        y_test_pred_baseline = model.predict(X_test_baseline)

        # Append baseline results
        results_data.append({
            'Model': model_name,
            'Dataset': dataset_name,
            'Feature Set': 'Baseline',
            'Validation Accuracy': round(accuracy_score(y_val_baseline, y_val_pred_baseline), 2),
            'Test Accuracy': round(accuracy_score(y_test_baseline, y_test_pred_baseline), 2),
            'Validation Precision': round(precision_score(y_val_baseline, y_val_pred_baseline), 2),
            'Test Precision': round(precision_score(y_test_baseline, y_test_pred_baseline), 2),
            'Validation Recall': round(recall_score(y_val_baseline, y_val_pred_baseline), 2),
            'Test Recall': round(recall_score(y_test_baseline, y_test_pred_baseline), 2),
            'Validation F1 Score': round(f1_score(y_val_baseline, y_val_pred_baseline), 2),
            'Test F1 Score': round(f1_score(y_test_baseline, y_test_pred_baseline), 2)
        })

        # Split all features
        X_train_all, X_val_all, X_test_all, y_train_all, y_val_all, y_test_all = split_data(X_all, y)

        model.fit(X_train_all, y_train_all)
        y_val_pred_all = model.predict(X_val_all)
        y_test_pred_all = model.predict(X_test_all)

        # Append all feature results
        results_data.append({
            'Model': model_name,
            'Dataset': dataset_name,
            'Feature Set': 'All Features',
            'Validation Accuracy': round(accuracy_score(y_val_all, y_val_pred_all), 2),
            'Test Accuracy': round(accuracy_score(y_test_all, y_test_pred_all), 2),
            'Validation Precision': round(precision_score(y_val_all, y_val_pred_all), 2),
            'Test Precision': round(precision_score(y_test_all, y_test_pred_all), 2),
            'Validation Recall': round(recall_score(y_val_all, y_val_pred_all), 2),
            'Test Recall': round(recall_score(y_test_all, y_test_pred_all), 2),
            'Validation F1 Score': round(f1_score(y_val_all, y_val_pred_all), 2),
            'Test F1 Score': round(f1_score(y_test_all, y_test_pred_all), 2)
        })

# Create a DataFrame and save to CSV
results_df = pd.DataFrame(results_data)
results_df.to_csv(csv_results_file, index=False)

print(f"Tabular results saved to: {csv_results_file}")
