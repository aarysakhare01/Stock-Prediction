# Stock-Prediction With Linear Regression
This project demonstrates a simple implementation of stock price prediction using Linear Regression in Python. It utilizes historical stock data to engineer features like moving averages and lagged closing prices, and evaluates a regression model to predict future closing prices.

📁 Project Structure :
bash
Copy
Edit

🚀 Features
Reads and processes historical stock data from a CSV file

Creates technical indicators like 10-day and 50-day moving averages

Splits data into training and testing sets

Trains a Linear Regression model

Evaluates model performance using MAE and RMSE

Visualizes predictions vs. actual prices

📦 Requirements
Make sure to install the following Python packages:
bash
Copy
Edit
pip install numpy pandas matplotlib scikit-learn

🧠 How It Works
Data Loading:
Loads a stock dataset from a CSV file.

Preprocessing:
Converts the Date column to datetime format

Sets Date as index

Selects Close and Volume columns

Calculates:
Moving_Avg_10 (10-day rolling average)

Moving_Avg_50 (50-day rolling average)

Lagged_Close (previous day’s closing price)

Modeling:
Features: Volume, Moving_Avg_10, Moving_Avg_50, Lagged_Close

Target: Close

Splits the data into train and test sets

Trains a LinearRegression model

Evaluation:
Calculates Mean Absolute Error (MAE)

Calculates Root Mean Squared Error (RMSE)

Plots actual vs predicted stock prices

📊 Output
The script prints evaluation metrics and displays a plot showing actual vs. predicted closing prices for the test set.

📝 Notes
Make sure to update the dataset_path variable with the correct path to your CSV dataset.

The script assumes the CSV contains columns: Date, Close, and Volume.

🔍 Sample Visualization
Example plot output:

mathematica
Copy
Edit
Actual Closing Prices vs Predicted Prices
