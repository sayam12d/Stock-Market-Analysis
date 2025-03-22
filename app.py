from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from datetime import datetime

app = Flask(__name__)

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle form submission and stock analysis
@app.route('/analyze', methods=['POST'])
def analyze():
    # Get user inputs from the form
    stock_symbol = request.form['symbol']
    start_date = request.form['start_date']
    end_date = request.form['end_date']

    # Check if start_date and end_date are provided
    if not start_date or not end_date:
        return render_template('index.html', error="Please provide both start and end dates.")

    # Convert start and end dates to datetime objects
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')

    # Fetch stock data from Yahoo Finance
    data = yf.download(stock_symbol, start=start_date, end=end_date)

    if data.empty:
        return render_template('index.html', error="No data found for the given stock symbol or date range.")

    # Calculate technical indicators
    data['SMA'] = data['Close'].rolling(window=10).mean()  # 10-day Simple Moving Average
    data['Price Change'] = data['Close'].pct_change()  # Percentage change in price

    # Create target variable: 1 if price goes up in the next 5 days, else 0
    data['Target'] = (data['Close'].shift(-5) > data['Close']).astype(int)

    # Drop rows with NaN values that may have appeared due to rolling windows
    data = data.dropna()

    # Prepare data for model training
    X = data[['Close', 'SMA', 'Price Change']]  # Features: closing price, SMA, and price change
    y = data['Target']  # Target: 1 if the stock price is predicted to go up, else 0

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train a RandomForest model
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    # Predict the stock movement for the test set
    predictions = model.predict(X_test)

    # Calculate the overall recommendation
    positive_predictions = sum(predictions)  # count of "1"s (Buy)
    negative_predictions = len(predictions) - positive_predictions  # count of "0"s (Sell)

    if positive_predictions > negative_predictions:
        overall_recommendation = 'Buy'  # Stock is predicted to go up, so "Buy"
    elif negative_predictions > positive_predictions:
        overall_recommendation = 'Sell'  # Stock is predicted to go down, so "Sell"
    else:
        overall_recommendation = 'Hold'  # Stock has neutral trend, so "Hold"

    # Generate a closing price plot
    plt.figure(figsize=(10, 6))
    plt.plot(data['Close'], label='Closing Price', color='blue')
    plt.title(f"{stock_symbol} Closing Prices")
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plot_path = f'static/{stock_symbol}_plot.png'
    plt.savefig(plot_path)
    plt.close()

    # Pass results to the UI
    return render_template(
        'index.html',
        stock_symbol=stock_symbol,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        plot_path=plot_path,
        overall_recommendation=overall_recommendation  # Show the overall recommendation
    )

if __name__ == '__main__':
    app.run(debug=True)
