import os
from newsapi import NewsApiClient
from textblob import TextBlob
from dotenv import load_dotenv
load_dotenv()

# FIX: load API key from environment variable instead of hardcoding it
API_KEY = os.getenv("NEWSAPI_KEY")

# FIX: map tickers to company names so NewsAPI returns relevant articles
TICKER_TO_NAME = {
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS": "Tata Consultancy Services",
    "INFY.NS": "Infosys",
    "HDFCBANK.NS": "HDFC Bank"
}

def get_sentiment(company):
    try:
        query = TICKER_TO_NAME.get(company, company)  # FIX: use company name, fall back to ticker
        newsapi = NewsApiClient(api_key=API_KEY)
        articles = newsapi.get_everything(q=query, language="en", page_size=10)

        scores = []

        for article in articles["articles"]:
            title = article.get("title")
            if title:
                polarity = TextBlob(title).sentiment.polarity
                scores.append(polarity)

        if len(scores) == 0:
            return 0

        return sum(scores) / len(scores)

    except Exception as e:
        # FIX: log the actual error instead of silently returning 0
        print(f"Sentiment fetch error for {company}: {e}")
        return 0
