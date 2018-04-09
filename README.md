

```python
# Dependencies
import tweepy
import json
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from config import consumer_key, consumer_secret, access_token, access_token_secret
```


```python
# Import and Initialize Sentiment Analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
```


```python
# Twitter API Keys
consumer_key = consumer_key
consumer_secret = consumer_secret
access_token = access_token
access_token_secret = access_token_secret
```


```python
# Setup Tweepy API Authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())
```


```python
# Target news organizations
news_orgs = ["BBC", "CBS", "CNN", "FoxNews", "nytimes"]
org_handles = ["@BBC", "@CBS", "@CNN", "@FoxNews", "@nytimes"]
```


```python
#Empty dictionary for news_orgs' compound scores
compound_dict = {}
for org in news_orgs :
    compound_dict[org]=[]

#Empty lists to hold compound source account, tweets_ago, text, date, 
#sentiments (compound, positive, neutral, and negative) for each news org
source_account = []
tweets_ago = []
text = []
date = []
compound = []
positive = []
neutral = []
negative = []
```


```python
#Fill in lists
for org, handle in zip(news_orgs,org_handles) :
    
    # Counter
    counter = 1

    # Variable for max_id
    oldest_tweet = None

    # Loop through 5 pages of tweets (total 100 tweets)
    for x in range(5):

        # Get all tweets from home feed
        public_tweets = api.user_timeline(handle, max_id = oldest_tweet)

        # Loop through all tweets 
        for tweet in public_tweets:
            
            source_account.append(org)
            tweets_ago.append(counter)
        
            text.append(tweet["text"])
            date.append(tweet["created_at"])
            
            # Run Vader Analysis on each tweet
            results = analyzer.polarity_scores(tweet["text"])
            compound.append(results["compound"])
            compound_dict[org].append(results["compound"])
            positive.append(results["pos"])
            neutral.append(results["neu"])
            negative.append(results["neg"])

            # Get Tweet ID, subtract 1, and assign to oldest_tweet
            oldest_tweet = tweet['id'] - 1

            # Add to counter 
            counter += 1
```


```python
# PLOT ONE
```


```python
# Create plot 
x_vals = tweets_ago
y_vals = compound
for org,c in zip(news_orgs,['red','blue','purple','green','yellow']) :
    plt.plot(range(100), compound_dict[org], markeredgecolor='black', color=c, alpha=0.6, label="%s" %x, marker='o', linewidth=0)   

lgnd = plt.legend(news_orgs, title= 'News Organizations', bbox_to_anchor=(1, 0.75))

# # Incorporate the other graph properties
now = datetime.now()
now = now.strftime("%m/%d/%y")
plt.title(f"Sentiment Analysis of Media Tweets ({now})")
plt.xlim([max(x_vals),min(x_vals)])
plt.ylim([min(y_vals)-.1,max(y_vals)+.1]) 
plt.yticks(np.arange(-1, 1.1, 0.5))
plt.ylabel("Tweet Polarity")
plt.xlabel("Tweets Ago")

# Save the figure
plt.savefig("plot1.png")

#Show the figure
plt.show()

```


![png](output_8_0.png)



```python
#PLOT TWO
```


```python
# Create plot 
x_vals = tweets_ago
y_vals = compound
for org,c in zip(news_orgs,['red','blue','purple','green','yellow']) :
    plt.bar(org, np.mean(compound_dict[org]), color=c, alpha=0.6, label="%s" %x, linewidth=0)   

#lgnd = plt.legend(news_orgs, title= 'News Organizations', bbox_to_anchor=(1, 0.75))

# # Incorporate the other graph properties
now = datetime.now()
now = now.strftime("%m/%d/%y")
plt.title(f"Overall Media Sentiment Based on Twitter ({now})")
plt.ylabel("Tweet Polarity")
plt.xlabel("Tweets Ago")

# Save the figure
plt.savefig("plot2.png")

#Show the figure
plt.show()
```


![png](output_10_0.png)



```python
#Pull info into a dataframe and export as a csv
```


```python
# Create a DataFrame to store info
tweets_df= pd.DataFrame(
    {"Source Account": source_account,
     "Date": date,
     "Text": text,
     "Compound Rating": compound,
     "Positive Rating": positive,
     "Neutral Rating": neutral,
     "Negative Rating": negative
    })
```


```python
# Export df to a csv
tweets_df.to_csv("news_sentiments.csv", encoding="UTF-8")
```
