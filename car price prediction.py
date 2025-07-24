import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv("quikr_car.csv")

# Drop rows with "Ask For Price"
df = df[~df['Price'].str.contains("Ask", na=False)]

# Clean 'Price' column
df['Price'] = df['Price'].str.replace(',', '').astype(int)

# Clean 'kms_driven'
df['kms_driven'] = df['kms_driven'].str.replace(' kms','').str.replace(',', '')
df['kms_driven'] = pd.to_numeric(df['kms_driven'], errors='coerce')

# Drop rows with missing values
df.dropna(inplace=True)

# Select features
df = df[['name', 'company', 'year', 'Price', 'kms_driven', 'fuel_type']]

# One-hot encoding
df = pd.get_dummies(df, columns=['company', 'fuel_type'])

# Features and target
X = df.drop(['Price', 'name'], axis=1)
y = df['Price']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train XGBoost Regressor
xgb_model = XGBRegressor(random_state=42)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)

# Evaluation
xgb_r2 = r2_score(y_test, xgb_preds)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
xgb_mae = mean_absolute_error(y_test, xgb_preds)

print("\n⚡ XGBoost Regressor Performance:")
print(f"R² Score: {xgb_r2:.3f}")
print(f"RMSE: ₹{xgb_rmse:,.0f}")
print(f"MAE: ₹{xgb_mae:,.0f}")