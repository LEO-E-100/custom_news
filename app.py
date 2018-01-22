from flask import Flask, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import pickle
import json
import gspread
import requests
from oauth2client.service_account import ServiceAccountCredentials
from bs4 import BeautifulSoup


app = Flask(__name__)


@app.route('/')
def display_links():
    return 'Hello World!'

@app.route('/stories')
# def test():
#     test_result = {
#     'value1' : 'Title 1',
#     'value2' : 'Title 2',
#     'value3' : 'Title 3',
#     'value4' : 'Title 4',
#     'value5' : 'Title 5',
#     'value6' : 'Title 6',
#     }
#     return render_template('display.html', result = test_result)
def fetch_news():
    vect = pickle.load(open(r'news_vect_pickle.pkl', 'rb'))
    model = pickle.load(open(r'news_model_pickle.pkl', 'rb'))

    scope = ['https://spreadsheets.google.com/feeds']
    credentials = ServiceAccountCredentials.from_json_keyfile_name('Custom News Feed-d7876b12a476.json', scope)
    gc = gspread.authorize(credentials)

    ws = gc.open('NewsFeed')
    sh = ws.sheet1
    zd = list(zip(sh.col_values(2), sh.col_values(3), sh.col_values(4)))
    zf = pd.DataFrame(zd, columns=['title', 'urls', 'html'])
    zf.replace('', pd.np.nan, inplace=True)
    zf.dropna(inplace=True)

    def get_text(x):
        soup = BeautifulSoup(x, 'lxml')
        text = soup.get_text()
        return text

    zf.loc[:, 'text'] = zf['html'].map(get_text)

    tv = vect.transform(zf['text'])
    res = model.predict(tv)

    rf = pd.DataFrame(res, columns=['wanted'])
    rez = pd.merge(rf, zf, left_index=True, right_index=True)

    mask = rez['wanted'] == 'y'
    yes_df = rez.loc[mask]
    short_df = yes_df.head()

    news_dict = {}
    for t, u in zip(short_df['title'], short_df['urls']):
        title = t
        url = u
        news_dict[t] = u

    return render_template('display.html', result=news_dict)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
