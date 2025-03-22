from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from datetime import datetime

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
   
    stock_symbol = request.form['symbol']
    start_date = request.form['start_date']
    end_date = request.form['end_date']

   
    if not start_date or not end_date:
        return render_template('index.html', error="Please provide both start and end dates.")

  
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')

  
    data = yf.download(stock_symbol, start=start_date, end=end_date)

    if data.empty:
        return render_template('index.html', error="No data found for the given stock symbol or date range.")

    
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
    plt.plot(data['Close'], label='Closing Price', color='blue')
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
