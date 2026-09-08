from flask import Flask, render_template, request, jsonify
import yfinance as yf
from model import final_result
from get_news import fetch_news
import markdown

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
last_detected_companies = {}
last_detected_name = None

@app.get('/')
def index_get():
    return render_template('index.html')

@app.post('/predict')
def predict():
    global conversation_history_text, last_detected_companies
    user_text = request.get_json().get('message')
    user_text_lower = user_text.lower()
    
    detected_companies = {}

    for company_name, ticker in COMPANY_MAP.items():
        if company_name in user_text_lower:

            if ticker not in detected_companies:
                detected_companies[ticker] = company_name

    if not detected_companies and last_detected_companies:
        detected_companies = last_detected_companies
    elif detected_companies:
        last_detected_companies = detected_companies
      
    live_data_str = "No real-time market data requested or available for this query."
    articles_by_company = {}

    if detected_companies:
        live_data_list = []
        
        for ticker, name in detected_companies.items():
            try:
                stock = yf.Ticker(ticker)
                price = stock.fast_info['lastPrice']
                volume = stock.fast_info['threeMonthAverageVolume']
                
                company_str = f"Current trading price for {ticker} ({name.capitalize()}) is ${price:.2f}. 3-Month Average Volume is {volume:,.0f}."
                live_data_list.append(company_str)
                
            except Exception as e:
                print(f"Failed to fetch yfinance data for {ticker}: {e}")
                live_data_list.append(f"Real-time data source temporarily unavailable for {ticker}.")

            print(f"Getting articles for {name}...")
            company_articles = fetch_news(f"{name} finance")
            articles_by_company[name] = company_articles or []
                
        live_data_str = "\n".join(live_data_list)

    articles_formatted_string = ""
    if articles_by_company:
        for comp_name, comp_articles in articles_by_company.items():
            articles_formatted_string += f"Articles for {comp_name.capitalize()}:\n"
            if comp_articles:
                for article in comp_articles:
                    title = article.get('title', 'No Title')
                    url = article.get('url', '#')
                    articles_formatted_string += f"- [{title}]({url})\n"
            else:
                articles_formatted_string += "- No recent news found.\n"
            articles_formatted_string += "\n"
    else:
        articles_formatted_string = "No recent news available."

    response = final_result(user_text, live_data_str, articles_formatted_string, conversation_history_text)
    
    answer = response['result']
    answer = markdown.markdown(answer)

    conversation_history_text += f"User: {user_text}\nAssistant: {answer}\n\n"
    return jsonify({'answer': answer})

if __name__=='__main__':
    app.run(debug=False)