# Diabetes Prediction Using Machine Learning (Google Sheets Data)

# 📌 Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# 📌 Step 2: Load Dataset from Google Sheets
sheet_url = "https://docs.google.com/spreadsheets/d/1p_WuY33JZo00wRFvtI7kEAITRHrwG0OM/edit?usp=sharing"
csv_export_url = sheet_url.replace("/edit?usp=sharing", "/export?format=csv")

df = pd.read_csv(csv_export_url)
print("First 5 rows of the dataset:")
print(df.head())

# 📌 Step 3: Basic EDA
print("\nDataset Info:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())
print("\nMissing values:")
print(df.isnull().sum())

# Correlation heatmap
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# 📌 Step 4: Prepare Data
X = df.drop('Outcome', axis=1)
y = df['Outcome']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 📌 Step 5: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 📌 Step 6: Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 📌 Step 7: Model Evaluation
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 📌 Step 8: Feature Importance
importances = model.feature_importances_
features = df.columns[:-1]

plt.figure(figsize=(10,6))
sns.barplot(x=importances, y=features)
plt.title("Feature Importance for Diabetes Prediction")
plt.show()

# 📌 Step 9: Save the Model
joblib.dump(model, 'diabetes_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# 📌 Step 10: Test the Model with a Sample Input
# Format: [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
sample_input = np.array([[2, 120, 70, 25, 80, 30.5, 0.5, 25]])
sample_scaled = scaler.transform(sample_input)
prediction = model.predict(sample_scaled)
result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"

print("\nSample Prediction Result:", result)
