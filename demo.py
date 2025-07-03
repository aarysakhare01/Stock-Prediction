import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Step 1: Load Dataset
dataset_path = "/Users/aarysakhare/Downloads/archive"  
stock_data = pd.read_csv(dataset_path)

# Step 2: Explore and Preprocess Data
# Print dataset info
print(stock_data.info())

# Check for missing values
print(stock_data.isnull().sum())

# Select relevant columns (assumes the dataset has 'Date', 'Close', 'Volume', etc.)
stock_data['Date'] = pd.to_datetime(stock_data['Date'])
stock_data.set_index('Date', inplace=True)
stock_data = stock_data[['Close', 'Volume']]  # Selecting Close price and Volume

# Feature Engineering
stock_data['Moving_Avg_10'] = stock_data['Close'].rolling(window=10).mean()
stock_data['Moving_Avg_50'] = stock_data['Close'].rolling(window=50).mean()
stock_data['Lagged_Close'] = stock_data['Close'].shift(1)

# Drop rows with NaN values due to rolling averages
stock_data.dropna(inplace=True)

# Step 3: Define Features and Target
features = ['Moving_Avg_10', 'Moving_Avg_50', 'Lagged_Close', 'Volume']
target = 'Close'

X = stock_data[features]
y = stock_data[target]

# Step 4: Split Data into Train and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Step 5: Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Make Predictions
y_pred = model.predict(X_test)

# Step 7: Evaluate the Model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")

# Step 8: Visualization
plt.figure(figsize=(12, 6))
plt.plot(stock_data.index[-len(y_test):], y_test, label='Actual Prices', color='blue')
plt.plot(stock_data.index[-len(y_test):], y_pred, label='Predicted Prices', color='orange')
plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()
