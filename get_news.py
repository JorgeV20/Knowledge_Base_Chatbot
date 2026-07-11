import os
import requests

def fetch_news(topic):
    # Replace with API_KEY from NEWSAPI.ORG
    API_KEY = os.getenv("NEWS_API_KEY")

    url = f"https://newsapi.org/v2/everything?q={topic}&sortBy=publishedAt&pageSize=3&language=en&apiKey={API_KEY}"
    
    articles_headlines = []

    try:
        response = requests.get(url)
        data = response.json()
        
        if data["status"] == "ok" and data["totalResults"] > 0:
            articles = data["articles"]
            print(f"Here are the latest headlines about {topic}:")
            
            for i, article in enumerate(articles, 1):
                print(f"Headline {i}: {article['title']}")
                print(f"Link to headline {i}: {article['url']}, published {article['publishedAt']}")
                articles_headlines.append(f"Headline {i}: {article['title']}")

        else:
            print(f"I couldn't find any recent news about {topic}.")
            
    except Exception as e:
        print("Sorry, I had trouble connecting to the news service.")
    
    return articles_headlines
