# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import resample
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = "C:/users/kasutaja/downloads/simulated_employee_lifecycle.csv"
data = pd.read_csv(file_path)

# Data Preparation
# Parse date columns
date_columns = ['RecruitmentDate', 'ExitDate']
for col in date_columns:
    if col in data.columns:
        data[col] = pd.to_datetime(data[col], errors='coerce')

# Create new features from dates
if 'RecruitmentDate' in data.columns:
    data['YearsSinceRecruitment'] = 2024 - data['RecruitmentDate'].dt.year  # Example calculation
if 'ExitDate' in data.columns:
    data['ExitYear'] = data['ExitDate'].dt.year

# Encode categorical variables
le = LabelEncoder()
data['Attrition'] = le.fit_transform(data['Attrition'])  # Yes = 1, No = 0
categorical_columns = ['Department', 'JobRole', 'Gender']
data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# Handle missing values
data = data.dropna()

# Select numerical columns (exclude date columns)
numerical_columns = ['Age', 'SatisfactionLevel', 'YearsAtCompany', 'WorkLifeBalance', 'PerformanceScore', 'YearsSinceRecruitment']
scaler = StandardScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# Feature Selection
X = data.drop(['Attrition', 'EmployeeID', 'RecruitmentDate', 'ExitDate'], axis=1, errors='ignore')
y = data['Attrition']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Handle class imbalance using undersampling
train_data = pd.concat([X_train, y_train], axis=1)
majority = train_data[train_data['Attrition'] == 0]
minority = train_data[train_data['Attrition'] == 1]

# Downsample majority class
majority_downsampled = resample(
    majority, 
    replace=False,  # Sample without replacement
    n_samples=len(minority),  # Match the minority class count
    random_state=42
)

# Combine downsampled majority and minority class
train_balanced = pd.concat([majority_downsampled, minority])
X_train_balanced = train_balanced.drop('Attrition', axis=1)
y_train_balanced = train_balanced['Attrition']

# Model Selection and Training
model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train_balanced, y_train_balanced)

# Model Evaluation
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=1))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC-AUC Score
y_pred_prob = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_prob)
print(f"ROC-AUC Score: {roc_auc:.2f}")

# ROC Curve
RocCurveDisplay.from_estimator(model, X_test, y_test)
plt.title("ROC Curve")
plt.show()

# Feature Importance
feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
feature_importance.plot(kind='bar', color='skyblue')
plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()
