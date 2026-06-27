from flask import Flask, render_template, request, jsonify
import yfinance as yf
from model import final_result
from get_news import fetch_news

app=Flask(__name__)

COMPANIES = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "^GSPC", "GC=F"]

@app.get('/')
def index_get():
    return render_template('index.html')

@app.post('/predict')
def predict():
    user_text = request.get_json().get('message')
    
    detected_ticker = None
    for ticker in COMPANIES:
        if ticker.lower() in user_text.lower():
            detected_ticker = ticker
            break
            
    live_data_str = "No real-time market data requested or available for this query."
    
    if detected_ticker:
        try:
            print(f"Fetching {detected_ticker} data")
            stock = yf.Ticker(detected_ticker)
            price = stock.fast_info['lastPrice']
            volume = stock.fast_info['threeMonthAverageVolume']
            
            live_data_str = f"Current trading price for {detected_ticker} is ${price:.2f}. 3-Month Average Volume is {volume:,.0f}."
            print(live_data_str)
        except Exception as e:
            print(f"Failed to fetch yfinance data: {e}")
            live_data_str = "Real-time data source temporarily unavailable."

    print("getting articles")
    articles = fetch_news(detected_ticker)
    print(articles)
    response = final_result(user_text, live_data_str, articles)
    
    answer = response['result']
    return jsonify({'answer': answer})

if __name__=='__main__':
    app.run(debug=False)