import yfinance as yf
import time

companies = ["AAPL", "GOOGL", "TSLA"]

for company in companies:
    ticker = yf.Ticker(company)

    for i in range(2):
        current_price = ticker.fast_info['lastPrice']
        market_cap = ticker.fast_info['marketCap']
        
        print(f"[{time.strftime('%H:%M:%S')}] {company} is currently trading at: ${current_price:.2f}")
        
        time.sleep(5)