import feedparser
import pandas as pd

def fetch_crime_articles():
    try:
        feed_url = "https://news.google.com/rss/search?q=crime+India&hl=en-IN&gl=IN&ceid=IN:en"
        feed = feedparser.parse(feed_url)

        articles = []
        for entry in feed.entries:
            title = entry.title
            link = entry.link
            articles.append({
                "title": title,
                "link": link,
                "text": title  # use title for classification
            })

        return pd.DataFrame(articles)
    except Exception as e:
        print(f"[ERROR] RSS fetch failed: {e}")
        return pd.DataFrame(columns=["title", "link", "text"])
