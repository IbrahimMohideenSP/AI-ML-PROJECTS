import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL

# === Step 1: Load the Data ===
df = pd.read_csv("stock_data_2000.csv", parse_dates=["Date"])
df.sort_values("Date", inplace=True)
df.set_index("Date", inplace=True)

# === Step 2: Decomposition ===
series = df['Close']
stl = STL(series, seasonal=365, robust=True)
result = stl.fit()

# === Step 3: Attach components to DataFrame ===
df['Trend'] = result.trend
df['Seasonal'] = result.seasonal
df['Residual'] = result.resid

# === Step 4: Summary Statistics ===
print("=== Summary Statistics ===")
print(df[['Close', 'Trend', 'Seasonal', 'Residual']].describe())

# === Step 5: Visualization ===
plt.figure(figsize=(14, 10))

plt.subplot(4, 1, 1)
plt.plot(df['Close'], color='black', label='Original')
plt.title('📈 Original Stock Price')
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(df['Trend'], color='blue', label='Trend')
plt.title('📉 Trend Component')
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(df['Seasonal'], color='orange', label='Seasonality')
plt.title('📊 Seasonality Component')
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(df['Residual'], color='red', label='Residual')
plt.title('🔁 Residual (Noise)')
plt.legend()

plt.tight_layout()
plt.show()

# === Step 6: Export Decomposition to CSV ===
df.to_csv("stock_data_2000_decomposed.csv")
print("\n✅ Decomposed components saved to 'stock_data_2000_decomposed.csv'")
