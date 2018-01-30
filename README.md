
# Custom News Feed
*24-01-18*

## Introduction

Principle is to build an email service app and flask web app that will deliver a daily email with 5 recommended articles based on what articles a person is interested in according to a machine learning algorithm. This will initally mean creating a model based on data pulled from pocket using the API. Laterly this will mean creating a service that will pull news from RSS feeds and testing them against the model to find top recommendations. Then creating a second service that will send out automatic emails with recommended articles.

## Creating supervised dataset

Using the Pocket API to create the initial dataset. Having curated a dataset by adding ~200 articles to pocket and tagged them with 'y' or 'n' depending on whether the user is interested in the article. Ultimately this will give a dataframe of article urls and a label of whether the user is interested or not.

## Scraping article content

Once the URLs have been recovered it is necessary to scrape the text from these articles in order to carry out the NLP steps. This requires the use of scraping on a number of different web sources. This would be a time consuming process if I were to write a bespoke web scraper for each website. Therefore I decided to use an link embedding service with an api to query. Initially I tried to use embed.ly however this is now a very expensive paid service. I instead have used embed.rocks. This is tested below and applied to the whole list of article URLs from above. 

Once the raw HTML has been extracted from the API, it was necessary to add just the text as a new column which is done using BeautifulSoup.


```python
import urllib
def get_html(x):
    qurl = urllib.parse.quote(x)
    rhtml = requests.get('https://api.embed.rocks/api/?url=' + qurl + '&key=' + config.embed_rocks_key)
    try:
        ctnt = json.loads(rhtml.text).get('article')
    except ValueError:
        ctnt = None
    return ctnt
df.loc[:, 'html'] = df['urls'].map(get_html)
df.dropna(inplace=True)
df.head()
```


```python
from bs4 import BeautifulSoup
def get_text(x):
    soup = BeautifulSoup(x, 'lxml')
    text = soup.get_text()
    return text
df.loc[:, 'text'] = df['html'].map(get_text)
df.dropna(inplace=True)
df.head()
```

## Natural Language Processing

From the text column it is possible to call a vectorizer in order to turn the text data into a usable matrix format for Machine Learning. 

For this project the `TfidfVectorizer` was used. This calculates a score for words based on the inverse of how commonly they occur. Thus rare words (which intuitively have greater explanatory power) are weighted most highly. 

As well as this the `stop_words` argument was also called to eliminate the most common words (i.e. 'the', 'a', 'said' etc.) This gives the vector matrix the maximum possible explanatory power.


```python
from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer(ngram_range=(1,3), stop_words='english', min_df=3)
vector_matrix = vect.fit_transform(df['text'])
```

## Support Vector Machines

Building a Support Vector Machine model from the vectorised data. This step will require some evaluation of the quality of the model. As well as simple building of the model a GridSearch cross validation was called to establish the most effective parameters to call. This also meant splitting the data into train/test in order to score the various parameter combinations.


```python
from sklearn.svm import LinearSVC, SVC
clf = SVC()
model = clf.fit(vector_matrix, df['wanted'])
```


```python
X = vector_matrix
y = df['wanted']
```


```python
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import numpy as np

X_train, X_test, y_train, y_test = train_test_split(X, y)

params = {
    "C" : [1],
    "kernel" : ['rbf'],
    "gamma" : np.linspace(0, 100, 10)
}

gridSearch = GridSearchCV(clf, params, cv=5, n_jobs=1, verbose=1)
model = gridSearch.fit(X_train, y_train)
```


```python
print (model.best_params_)
best_model = model.best_estimator_
best = best_model.fit(X_train, y_train)
score = best.score(X_test, y_test)
print ('Score:\t', score)
```

## Output Model to Pickle

This step involves pickling the model in order that it is callable from elsewhere in the system. In particular that it is callable by both the Flask app and by the script which will provide the email service from AWS.


```python
import pickle
pickle.dump(model, open(r'news_model_pickle.pkl', 'wb'))
pickle.dump(vect, open(r'news_vect_pickle.pkl', 'wb'))
```

## Delivering the app

The two methods of deliver were email and Flask app.

### Flask App

A flask app was produced that would call a stripped down method of the model and return the top five results as links. This app can be viewed in its entirety in the [GitHub Repo](https://github.com/LEO-E-100/custom_news) however it is relatively simple and gathers the RSS sources from an IFTTT system which delivers news articles to a Google Sheet which can then be loaded into Python using `gspread` the google library for Python. The data is then dynamically loaded into the HTML using the simple Flask method.

### Email App

A script was written which would gather results in the same was as the Flask app but would send them via IFTTT to a specified email address. This script was then run from an Amazon Web Services server through a cron job which would run once a day. Thus delivering new recommonded stories each day to the users inbox.
