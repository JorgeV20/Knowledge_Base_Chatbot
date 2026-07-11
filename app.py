from flask import Flask, render_template, request, jsonify
import yfinance as yf
from model import final_result
from get_news import fetch_news

app=Flask(__name__)

COMPANY_MAP = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "amazon": "AMZN",
    "tesla": "TSLA",
    "nvidia": "NVDA",
    "meta": "META",
    "facebook": "META"
}

# Global variable to store the conversation log
conversation_history_text = ""
last_detected_ticker = None
last_detected_name = None

@app.get('/')
def index_get():
    return render_template('index.html')

@app.post('/predict')
def predict():
    global conversation_history_text, last_detected_ticker, last_detected_name
    user_text = request.get_json().get('message')
    user_text_lower = user_text.lower()
    
    detected_ticker = None
    detected_name = None

    for company_name, ticker in COMPANY_MAP.items():
        if company_name in user_text_lower:
            detected_ticker = ticker
            detected_name = company_name
            break

    if not detected_ticker and last_detected_ticker:
        detected_ticker = last_detected_ticker
        detected_name = last_detected_name
    elif detected_ticker:
        last_detected_ticker = detected_ticker
        last_detected_name = detected_name
      
    live_data_str = "No real-time market data requested or available for this query."
    articles = []

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
        stock_news=f"{detected_name} finance"
        articles = fetch_news(stock_news)
        print(articles)
    
    response = final_result(user_text, live_data_str, articles, conversation_history_text)
    
    answer = response['result']

    conversation_history_text += f"User: {user_text}\nAssistant: {answer}\n\n"
    return jsonify({'answer': answer})

if __name__=='__main__':
    app.run(debug=False)