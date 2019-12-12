import math

import cv2
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

    MAX_FRAMES = 100
    MIN_TIME_STAMP = 5
    frame_skip_count = 1

    video_cap = cv2.VideoCapture(path)
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    frame_count = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = math.ceil(frame_count / fps)
    time_skip = max(MIN_TIME_STAMP, math.ceil(duration / MAX_FRAMES))
    frame_skip_count = max(frame_skip_count, math.ceil(time_skip * fps))

    images_path = 'images'
    success, image = video_cap.read()
    count = 1
    while success:
        if count % frame_skip_count == 0:
            cv2.imwrite("%s/frame_%d.jpg" % (images_path, int(count / frame_skip_count)), image)
        success, image = video_cap.read()
        count += 1
    video_cap.release()

    return 'Hello World!'


if __name__ == '__main__':
    app.run()
