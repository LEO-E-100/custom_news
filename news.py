import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import schedule
import time
import pickle
import json
import gspread
import requests
from oauth2client.service_account import ServiceAccountCredentials
from bs4 import BeautifulSoup

pd.set_option('display.max_colwidth', 250)

def fetch_news():
    try:
        vect = pickle.load(open(r'news_vect_pickle.pkl', 'rb'))
        model = pickle.load(open(r'news_model_pickle.pkl', 'rb'))

        scope = ['https://spreadsheets.google.com/feeds']
        credentials = ServiceAccountCredentials.from_json_keyfile_name('/Users/Leo/Documents/Programming/CONFIG_NOT_SHARED/Custom News Feed-d7876b12a476.json', scope)
        gc = gspread.authorize(credentials)

        ws = gs.open('NewsFeed')
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

        news_str = ''
        for t, u in zip(rez[rez['wanted'] == 'y']['title'], rez[rez['wanted'] == 'y']['urls']):
            news_str = news_str + t + '\n' + u + '\n'

        payload = {'value1' : news_str}
        r = requests.post('https://maker.ifttt.com/use/2ITUA8L7j8IF1m3wAfQnA', data=payload)

        lenv = len(sh.col_values(1))
        cell_list = sh.range('A1:F' + str(lenv))
        for cell in cell_list:
            cell.value = ''
        sh.update_cells(cell_list)

        print(r.text)

    except:
        print('Failed')

schedule.every(60).minutes.do(fetch_news)

while 1:
    schedule.run_pending()
    time.sleep(1)
