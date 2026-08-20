import os
import requests

def fetch_news(topic):
    # Retrieve API_KEY from environment variables
    API_KEY = os.getenv("NEWS_API_KEY")

    url = f"https://newsapi.org/v2/everything?q={topic}&sortBy=publishedAt&pageSize=3&language=en&apiKey={API_KEY}"
    
    articles_list = []

    try:
        response = requests.get(url)
        data = response.json()
        
        if data.get("status") == "ok" and data.get("totalResults", 0) > 0:
            articles = data["articles"]
            print(f"Here are the latest headlines about {topic}:")
            
            for i, article in enumerate(articles, 1):
                print(f"Headline {i}: {article['title']}")
                print(f"Link: {article['url']}")
                
                # Append a dictionary with title and url
                articles_list.append({
                    "title": article["title"],
                    "url": article["url"]
                })

        else:
            print(f"I couldn't find any recent news about {topic}.")
            
    except Exception as e:
        print(f"Sorry, I had trouble connecting to the news service: {e}")
    
    return articles_list