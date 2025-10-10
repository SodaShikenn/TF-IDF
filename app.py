from flask import Flask, render_template
from flask import request
from utils import search


app = Flask(__name__)

@app.route('/')
def index():
    word = request.values.get('word', '')
    items = search(word, 5)[1]
    context = {
        'word': word,
        'items': items
    }
    return render_template('index.html', **context)

if __name__ == '__main__':
    app.run(debug=True, port=5001)