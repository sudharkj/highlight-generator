import requests
from flask import Flask

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/generate')
def generate_highlights():
    url_link = 'junk'
    path = 'downloads/random.mp4'
    chunk_size = 8192

    with requests.get(url_link, stream=True) as r:
        with open(path, 'wb') as out:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    out.write(chunk)

    return 'Hello World!'


if __name__ == '__main__':
    app.run()
