
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("student_data.csv")

# Encode categorical data
le = LabelEncoder()
df['ParentalEducation'] = le.fit_transform(df['ParentalEducation'])
df['InternetAccess'] = df['InternetAccess'].map({'Yes': 1, 'No': 0})

# Features and target
X = df.drop(['StudentID', 'Dropout'], axis=1)
y = df['Dropout']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Logistic Regression
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
log_preds = log_model.predict(X_test)

# Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

# Results
print("=== Logistic Regression Report ===")
print(classification_report(y_test, log_preds))

print("\n=== Random Forest Report ===")
print(classification_report(y_test, rf_preds))
