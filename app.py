from flask import Flask, render_template
# import news


app = Flask(__name__)


@app.route('/')
def display_links():
    result = news.payload
    return render_template('display.html', result=result)

if __name__ == '__main__':
    app.run()
