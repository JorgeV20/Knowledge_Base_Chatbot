import os
import json
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("STOCK_API_KEY")

if not API_KEY:
    raise ValueError("API Key not found! Make sure STOCK_API_KEY is defined in your .env file.")

TICKER = "AAPL"
URL = f"https://api.example.com/v1/stocks/{TICKER}" 

params = {
    "apikey": API_KEY
}

def fetch_and_save_stock_data():
    print(f"Fetching data for {TICKER}...")
    
    try:

        response = requests.get(URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        json_filename = f"{TICKER}_data.json"
        with open(json_filename, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Successfully saved raw data to {json_filename}")
        
        display_as_table(json_filename)

    except requests.exceptions.RequestException as e:
        print(f"An error occurred while fetching data: {e}")

def display_as_table(json_filepath):
    with open(json_filepath, "r") as f:
        data = json.load(f)

    try:
        df = pd.DataFrame(data) 
        
        print("\n--- Stock Data Table Preview ---")
        print(df.to_string(index=False))
    except Exception as e:
        print(f"\nCould not automatically format JSON to table. Error: {e}")
        print("Need to normalize the JSON structure using pd.json_normalize(data)")

if __name__ == "__main__":
    fetch_and_save_stock_data()