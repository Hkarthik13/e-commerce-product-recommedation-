import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Step 1: Load the Dataset
df = pd.read_csv('Cleaned_Amazon_Customer_Behavior_Survey.csv')

# Drop the Timestamp column as it cannot be converted to float
df = df.drop('Timestamp', axis=1)

# Step 2: Exploratory Data Analysis (EDA)

# 2.1 Basic Information and Descriptive Statistics
print(df.info())
print(df.describe(include='all'))

# 2.2 Visualize Distributions of Numerical Features
df.hist(bins=15, figsize=(15, 10))
plt.show()

# 2.3 Visualize Correlations
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

# Step 3: Data Preprocessing

# 3.1 Handle Categorical Variables
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# 3.2 Feature Scaling (if needed)
scaler = StandardScaler()
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Step 4: Model Training

# 4.1 Define Features and Target
X = df.drop('Shopping_Satisfaction', axis=1)
y = df['Shopping_Satisfaction']

# 4.2 Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4.3 Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 5: Model Evaluation

# 5.1 Make Predictions
y_pred = model.predict(X_test)

# 5.2 Evaluate the Model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))

# Step 6: Feature Importance
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 8))
plt.title("Feature Importance")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.tight_layout()
plt.show()
