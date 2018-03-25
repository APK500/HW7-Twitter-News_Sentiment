
#                  *****OBSERVATIONS OF TRENDS*****

# CBS Tends to have more positivly stated tweets and overall sentiment (compared to peers).


# BBC seems to have a pretty even dispersion of sentiment among the tweeters.

# CNN, and New York times, and Fox News seem to have more negative mood sentiment.  
# I think this is easily observable in the case of fox news (in the real-world, from a qualitive perspective).

# CNN could also be considered "neutral", compared to the others in the group.

# Timing of the tweets seems to occur pretty evenly, relative to eachother (both within their own news source and compared to other outlets).



```python
# Import our dependencies needed for twitter sentiment analysis:

import tweepy
import json
import pandas as pd
import numpy as np
from datetime import datetime 
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
from config import consumer_key,consumer_secret,access_token,access_token_secret

```


```python
# Now set up the TWEEPY API Authentication codes:
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())
```


```python
# Now we put in the actual news outlets (sites) that we will be pulling from 
# ...and enter into a llist:

target_user = ["@BBC","@CBS","@CNN","@FoxNews","@NYTimes"]
sentiment = []
```


```python
# Now we put in the looping for the tweets/twitter to extrapolate them
#.. and ultimetly to get the sentiments of the users (from the news sources being read)

for each in target_user:
    count1 = 0
    
    tweets = api.user_timeline(each,count=100)
    
    for tweet in tweets:
        
        compound = analyzer.polarity_scores(tweet["text"])["compound"]
        pos = analyzer.polarity_scores(tweet["text"])["pos"]
        neu = analyzer.polarity_scores(tweet["text"])["neu"]
        neg = analyzer.polarity_scores(tweet["text"])["neg"]
        tweets_ago = count1
        tweet_text = tweet["text"] 
        sentiment.append({"User": each,
                         "Date": tweet["created_at"],
                         "Compound": compound,
                         "Positive": pos,
                         "Negative": neg,
                         "Neutral": neu,
                         "Tweets Ago": count1,
                         "Text": tweet_text})
        count1+=1
        

# Next part (2) we create the sentiments (from news) dataframes.       
        
sentiments_news = pd.DataFrame.from_dict(sentiment)
sentiments_news.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Compound</th>
      <th>Date</th>
      <th>Negative</th>
      <th>Neutral</th>
      <th>Positive</th>
      <th>Text</th>
      <th>Tweets Ago</th>
      <th>User</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.3818</td>
      <td>Sat Mar 24 20:34:01 +0000 2018</td>
      <td>0.115</td>
      <td>0.885</td>
      <td>0.000</td>
      <td>"I still see people screaming for help."\n\nTh...</td>
      <td>0</td>
      <td>@BBC</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.4767</td>
      <td>Sat Mar 24 20:00:11 +0000 2018</td>
      <td>0.000</td>
      <td>0.866</td>
      <td>0.134</td>
      <td>The story of the last decade of Picasso's life...</td>
      <td>1</td>
      <td>@BBC</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.5859</td>
      <td>Sat Mar 24 19:16:04 +0000 2018</td>
      <td>0.000</td>
      <td>0.817</td>
      <td>0.183</td>
      <td>"Beauty is your inside, it's your personality ...</td>
      <td>2</td>
      <td>@BBC</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.5719</td>
      <td>Sat Mar 24 18:44:04 +0000 2018</td>
      <td>0.000</td>
      <td>0.748</td>
      <td>0.252</td>
      <td>Cambridge have won the men's and women's Boat ...</td>
      <td>3</td>
      <td>@BBC</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.3182</td>
      <td>Sat Mar 24 18:02:01 +0000 2018</td>
      <td>0.000</td>
      <td>0.777</td>
      <td>0.223</td>
      <td>📺😂 @RomeshRanga is NOT a fan of Gogglebox. #Li...</td>
      <td>4</td>
      <td>@BBC</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Now we export the above sentiment tweets into a CSV:
sentiments_news.to_csv("Collected_News_Sentiments_from_Twitter.csv", index = False)
Date = datetime.now().strftime("(%m/%d/%Y)")
```


```python
#  Now we plot those sentiments that are collected (below)
#...They are of the last 100 tweets (aka "ago")
for each in target_user:
    users_df = sentiments_news.loc[sentiments_news["User"] == each]
    plt.scatter(users_df["Tweets Ago"],users_df["Compound"],label = each)
    
#  And now plot: for the 100 tweets ago per each of the news outlets:
plt.xlim(100,-1)
plt.legend(bbox_to_anchor=(1,1))
plt.title("Vader Sentiment Analysis of News Tweets "+str(Date))
plt.xlabel("Number of Tweets Ago")
plt.ylabel("Tweet sentiment polarity")
plt.savefig("Sentiment Analysis of News Tweets For outlets")
plt.grid(True)
plt.show()



```


![png](output_6_0.png)



```python
# Now we structure the bar graph plot-verison of the overall sentiments of the last 
# ...100 tweets:
# This is aggregating the compound sentiments analyzed by Vader, etc etc....


# calcs of avg sentiments:
avg_sent = sentiments_news.groupby("User")
means_sentiments = avg_sent["Compound"].mean()
means_sentiments.head()


```




    User
    @BBC        0.110301
    @CBS        0.331690
    @CNN       -0.020002
    @FoxNews   -0.031420
    @NYTimes   -0.040135
    Name: Compound, dtype: float64




```python
# Now the actual plot, from the above:

fig, ax = plt.subplots()

x_axis = np.arange(len(means_sentiments))
count2 = 0
count = 0
for sent in means_sentiments:
    ax.text(count2, sent+.01, str(round(sent,2)))
    count2+=1
plt.ylim(-.15,.35)
plt.bar(x_axis, means_sentiments, tick_label = target_user, color = ['b', 'y', 'g', 'r', 'c'])
plt.title("Overall Sentiment of last 100 Tweets by Organization " +str(Date))
plt.xlabel("News Accounts")
plt.ylabel("Tweet Sentiment Polarity")
plt.savefig("Overall Sentiment of last 100 Tweets by Organization")
plt.show()
```


![png](output_8_0.png)

