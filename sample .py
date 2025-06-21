from news_scraper import fetch_crime_articles

df = fetch_crime_articles()
print(df.head())
