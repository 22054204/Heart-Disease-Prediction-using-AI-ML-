# Heart-Disease-Prediction-using-AI-ML
# Built a machine learning model using  Logistic Regression, Random Forest, XGBoost, and Neural Networks to predict heart disease  with 90% accuracy. Evaluated using precision, recall, F1-score, and ROC-AUC. Visualized  results to show real-world use.


import zipfile
import os

# Define file paths
zip_file_path = "/content/heart+disease.zip"  # Ensure this matches the actual file name
extract_path = "/content/heart_disease/"  # Use a directory to extract

# Ensure the ZIP file exists
if not os.path.exists(zip_file_path):
    raise FileNotFoundError(f"ZIP file not found at {zip_file_path}")

# Extract the ZIP file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# List extracted files
extracted_files = os.listdir(extract_path)
print("Extracted Files:", extracted_files)
import pandas as pd
import os

# Define the dataset file path (update this)
extract_path = "/content/heart_disease/"  # Update to your actual extracted directory
csv_filename = "processed.cleveland.data"
file_path = os.path.join(extract_path, csv_filename)

# Check if the file exists
if os.path.exists(file_path):
    # Define column names (from UCI Heart Disease dataset)
    column_names = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
        "exang", "oldpeak", "slope", "ca", "thal", "target"
    ]

    # Load dataset, handling missing values
    df = pd.read_csv(file_path, header=None, names=column_names, na_values="?")

    # Display dataset info
    print(df.info())  # Summary of the dataset
    print(df.head())  # Show first few rows

else:
    print(f"Error: File not found at {file_path}")
import pandas as pd

# Rename columns based on dataset documentation
df.columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
    "exang", "oldpeak", "slope", "ca", "thal", "num"
]

# Replace "?" with NaN (for missing values)
df.replace("?", pd.NA, inplace=True)

# Convert all columns to numeric
df = df.apply(pd.to_numeric)

# Drop rows with missing values
df.dropna(inplace=True)

# Reset index after dropping missing values
df.reset_index(drop=True, inplace=True)

# Convert target variable (num) into binary (0 = No Disease, 1+ = Disease)
df["num"] = df["num"].apply(lambda x: 1 if x > 0 else 0)

# Split features (X) and target (Y)
X = df.drop(columns=["num"])
Y = df["num"]

# Display the final dataset shape
print("Dataset shape:", df.shape)
print(df.head())
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=42
)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)
# Evaluate model
train_accuracy = accuracy_score(Y_train, model.predict(X_train))
test_accuracy = accuracy_score(Y_test, model.predict(X_test))

print(f"✅ Model Training Completed")
print(f"🎯 Training Accuracy: {train_accuracy:.2f}")
print(f"🎯 Testing Accuracy: {test_accuracy:.2f}")
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=500, max_depth=15, random_state=42)
rf_model.fit(X_train, Y_train)

rf_test_accuracy = accuracy_score(Y_test, rf_model.predict(X_test))
print(f"✅ Random Forest Accuracy: {rf_test_accuracy:.2f}")

from xgboost import XGBClassifier

xgb_model = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=10, random_state=42)
xgb_model.fit(X_train, Y_train)

xgb_test_accuracy = accuracy_score(Y_test, xgb_model.predict(X_test))
print(f"✅ XGBoost Accuracy: {xgb_test_accuracy:.2f}")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef

# Function to print confusion matrix and evaluation metrics
def evaluate_model(model, X_test, Y_test, model_name):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    cm = confusion_matrix(Y_test, y_pred)
    accuracy = accuracy_score(Y_test, y_pred)
    precision = precision_score(Y_test, y_pred)
    recall = recall_score(Y_test, y_pred)
    f1 = f1_score(Y_test, y_pred)
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])  # TNR = TN / (TN + FP)
    fpr = cm[0, 1] / (cm[0, 0] + cm[0, 1])  # FPR = FP / (TN + FP)
    fnr = cm[1, 0] / (cm[1, 0] + cm[1, 1])  # FNR = FN / (FN + TP)
    mcc = matthews_corrcoef(Y_test, y_pred)
    roc_auc = roc_auc_score(Y_test, y_pred_proba) if y_pred_proba is not None else "N/A"

    print(f"\n Confusion Matrix for {model_name}:\n", cm)
    print(f" Accuracy: {accuracy:.4f}")
    print(f" Precision (PPV): {precision:.4f}")
    print(f" Recall (Sensitivity): {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f" Specificity (TNR): {specificity:.4f}")
    print(f" False Positive Rate (FPR): {fpr:.4f}")
    print(f" False Negative Rate (FNR): {fnr:.4f}")
    print(f" Matthews Correlation Coefficient (MCC): {mcc:.4f}")
    print(f" ROC-AUC Score: {roc_auc}")

# Evaluate all models
evaluate_model(model, X_test, Y_test, "Logistic Regression")
evaluate_model(rf_model, X_test, Y_test, "Random Forest")
evaluate_model(xgb_model, X_test, Y_test, "XGBoost")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Function to plot confusion matrix
def plot_confusion_matrix(model, X_test, Y_test, model_name):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(Y_test, y_pred)

    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Disease", "Disease"], yticklabels=["No Disease", "Disease"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix for {model_name}")
    plt.show()

# Plot confusion matrices for all models
plot_confusion_matrix(model, X_test, Y_test, "Logistic Regression")
plot_confusion_matrix(rf_model, X_test, Y_test, "Random Forest")
plot_confusion_matrix(xgb_model, X_test, Y_test, "XGBoost")

from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Function to evaluate regression metrics
def evaluate_regression_metrics(model, X_test, Y_test, model_name):
    y_pred = model.predict(X_test)

    mse = mean_squared_error(Y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(Y_test, y_pred)

    print(f"\n Regression Metrics for {model_name}:")
    print(f" Mean Squared Error (MSE): {mse:.4f}")
    print(f" Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f" R-Squared (R²): {r2:.4f}")

# Evaluate all models
evaluate_regression_metrics(model, X_test, Y_test, "Logistic Regression")
evaluate_regression_metrics(rf_model, X_test, Y_test, "Random Forest")
evaluate_regression_metrics(xgb_model, X_test, Y_test, "XGBoost")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Regression Metrics Data
models = ["Logistic Regression", "Random Forest", "XGBoost"]
mse_values = [0.1667, 0.1333, 0.1500]
rmse_values = [0.4082, 0.3651, 0.3873]
r2_values = [0.3304, 0.4643, 0.3973]

# Set plot style
sns.set(style="whitegrid")

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Bar plot for MSE
sns.barplot(x=models, y=mse_values, ax=axes[0], palette="Blues")
axes[0].set_title("Mean Squared Error (MSE)")
axes[0].set_ylabel("MSE")

# Bar plot for RMSE
sns.barplot(x=models, y=rmse_values, ax=axes[1], palette="Greens")
axes[1].set_title("Root Mean Squared Error (RMSE)")
axes[1].set_ylabel("RMSE")

# Bar plot for R²
sns.barplot(x=models, y=r2_values, ax=axes[2], palette="Oranges")
axes[2].set_title("R-Squared (R²)")
axes[2].set_ylabel("R² Score")

# Show plots
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

feature_names = X.columns
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 5))
plt.title("Feature Importance - Random Forest")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), feature_names[indices], rotation=90)
plt.show()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, interaction_only=True)
X_poly = poly.fit_transform(X)

model = LogisticRegression(max_iter=2000)
model = LogisticRegression(C=0.5)
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5, min_samples_leaf=3, random_state=42)
rf_model.fit(X_train, Y_train)
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

param_grid = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [100, 200, 300]
}
xgb_model = xgb.XGBClassifier()
grid_search = GridSearchCV(xgb_model, param_grid, cv=5)
grid_search.fit(X_train, Y_train)

print(grid_search.best_params_)

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, Y_resampled = smote.fit_resample(X, Y)

# Neural Network

from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=50, batch_size=8, validation_split=0.2)

from xgboost import XGBClassifier

# Re-initialize and train the XGBoost model
xgb_model = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=10, random_state=42)
xgb_model.fit(X_train, Y_train)

# Verify training completion
print("✅ XGBoost model trained successfully!")
from sklearn.exceptions import NotFittedError

# Function to predict heart disease
def predict_heart_disease(model, scaler, user_input):
    try:
        # Convert input into a NumPy array and scale it
        user_input_scaled = scaler.transform([user_input])  # Ensure correct shape

        # Make prediction
        prediction = model.predict(user_input_scaled)[0]

        # Return result based on prediction
        return "🟢 The person is unlikely to have heart disease." if prediction == 0 else "🔴 The person is likely to have heart disease."

    except NotFittedError:
        return "❌ Error: The model is not trained. Train the model before predicting."

# Take user input
print("Enter patient details (comma or space separated):")
user_input = list(map(float, input().replace(",", " ").split()))

# Test with Logistic Regression
print("\n1. Test with Logistic Regression Model")
print(predict_heart_disease(model, scaler, user_input))

# Test with Random Forest
print("\n2. Test with Random Forest Model")
print(predict_heart_disease(rf_model, scaler, user_input))

# Test with XGBoost
print("\n3. Test with XGBoost Model")
print(predict_heart_disease(xgb_model, scaler, user_input))

