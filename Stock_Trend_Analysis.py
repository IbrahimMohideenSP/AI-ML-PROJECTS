
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL

# Load dataset
df = pd.read_csv("stock_data.csv", parse_dates=['Date'])
df.set_index('Date', inplace=True)

# STL Decomposition
stl = STL(df['Close'], period=3)
res = stl.fit()

# Plotting results
res.plot()
plt.suptitle("Stock Price Time Series Decomposition (STL)", fontsize=14)
plt.tight_layout()
plt.savefig("decomposition_result.png")
plt.show()
