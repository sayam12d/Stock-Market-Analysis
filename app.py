import os
from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)

datasets_folder = 'datasets'

def find_stock_file(stock_symbol):
    """Search for the file in the datasets folder matching the stock symbol."""
    files = os.listdir(datasets_folder)
    for file in files:
        if file.lower().startswith(stock_symbol.lower()) and file.endswith('.csv'):
            return os.path.join(datasets_folder, file)
    return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    stock_symbol = request.form['symbol'].upper()
    start_date = request.form['start_date']
    end_date = request.form['end_date']
    
    if not start_date or not end_date:
        return render_template('index.html', error="Please provide both start and end dates.")
    
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    
    data_file = find_stock_file(stock_symbol)
    if not data_file:
        return render_template('index.html', error=f"File not found for the given stock symbol: {stock_symbol}.")
    
    data = pd.read_csv(data_file)
    data['Date'] = pd.to_datetime(data['Date'])
    data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]
    
    if data.empty:
        return render_template('index.html', error="No data found for the given date range.")
    
    data['SMA'] = data['Close'].rolling(window=10).mean()
    data['Price Change'] = data['Close'].pct_change()
    data['Target'] = (data['Close'].shift(-5) > data['Close']).astype(int)
    data = data.dropna()
    
    X = data[['Close', 'SMA', 'Price Change']]
    y = data['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    
    positive_predictions = sum(predictions)
    negative_predictions = len(predictions) - positive_predictions
    
    if positive_predictions > negative_predictions:
        overall_recommendation = 'Buy'
    elif negative_predictions > positive_predictions:
        overall_recommendation = 'Sell'
    else:
        overall_recommendation = 'Hold'
    
    plt.figure(figsize=(10, 6))
    plt.plot(data['Date'], data['Close'], label='Closing Price', color='blue')
    plt.title(f"{stock_symbol} Closing Prices")
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    
    plot_path = f'static/{stock_symbol}_plot.png'
    plt.savefig(plot_path)
    plt.close()
    
    return render_template(
        'index.html',
        stock_symbol=stock_symbol,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        plot_path=plot_path,
        overall_recommendation=overall_recommendation
    )

if __name__ == '__main__':
    app.run(debug=True)
