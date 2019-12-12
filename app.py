import json
import math
import os

import cv2
import requests
from flask import Flask, send_from_directory

import predict
import utils

app = Flask(__name__, static_url_path='')

VIDEOS_BASE_PATH = '/Users/sudharkj/Developer/python/highlight-generator/videos'
IMAGES_BASE_PATH = '/Users/sudharkj/Developer/python/highlight-generator/images'
PREDICTIONS_BASE_PATH = '/Users/sudharkj/Developer/python/highlight-generator/predictions'


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/images/<path:path>')
def send_image(path):
    return send_from_directory(IMAGES_BASE_PATH, path)


def generate_output_item(prediction, base_path, file_ext):
    return {
        'image_id': "%s/%s.%s" % (base_path, prediction['image_id'], file_ext),
        'mean_score_prediction': prediction['mean_score_prediction']
    }


@app.route('/generate')
def generate_highlights():
    chunk_size = 8192
    MAX_FRAMES = 10
    MIN_TIME_STAMP = 1
    frame_skip_count = 1
    file_ext = 'jpg'
    PREDICTIONS_LIMIT = 2
    BASE_MODEL = 'MobileNet'
    WEIGHTS_FILE = 'weights_mobilenet_aesthetic_0.07.hdf5'

    url_link = 'https://oregonstate.app.box.com/index.php?rm=box_download_shared_file&shared_name=5n4bd25v6qanesj5n3sa3y85ww0pp9y4&file_id=f_575083376594'
    video_name = 'waves.mov'

    request_uid = utils.rand_gen()
    video_path = VIDEOS_BASE_PATH + '/' + request_uid
    images_path = IMAGES_BASE_PATH + '/' + request_uid
    predictions_path = PREDICTIONS_BASE_PATH + '/' + request_uid

    for dir_path in [video_path, images_path, predictions_path]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    video_file_path = video_path + '/' + video_name
    with requests.get(url_link, stream=True) as r:
        with open(video_file_path, 'wb') as out:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    out.write(chunk)

    video_cap = cv2.VideoCapture(video_file_path)
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    frame_count = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = math.ceil(frame_count / fps)
    time_skip = max(MIN_TIME_STAMP, math.ceil(duration / MAX_FRAMES))
    frame_skip_count = max(frame_skip_count, math.ceil(time_skip * fps))

    success, image = video_cap.read()
    count = 1
    while success:
        if count % frame_skip_count == 0:
            cv2.imwrite("%s/frame_%d.%s" % (images_path, int(count / frame_skip_count), file_ext), image)
        success, image = video_cap.read()
        count += 1
    video_cap.release()

    predictions_file = '%s/predicts.json' % predictions_path
    predictions = predict.score(BASE_MODEL, WEIGHTS_FILE, images_path, predictions_file)
    predictions = sorted(predictions, key=lambda k: k['mean_score_prediction'], reverse=True)
    predictions = list(map(lambda prediction: generate_output_item(prediction, images_path, file_ext),
                           predictions[:PREDICTIONS_LIMIT]))

    return json.dumps(predictions[:PREDICTIONS_LIMIT], indent=2)


if __name__ == '__main__':
    app.run()
