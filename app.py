import json
import math
import os
import re
import shutil

import cv2
from flask import Flask, send_from_directory, request
from werkzeug.utils import secure_filename

import predict
import utils

app = Flask(__name__, static_url_path='')

VIDEOS_BASE_PATH = '/Users/sudharkj/Developer/flask/highlight-generator/videos'
IMAGES_BASE_PATH = '/Users/sudharkj/Developer/flask/highlight-generator/images'
PREDICTIONS_BASE_PATH = '/Users/sudharkj/Developer/flask/highlight-generator/predictions'
OUTPUT_IMAGES_BASE_PATH = '/Users/sudharkj/Developer/flask/highlight-generator/output/images'

ALLOWED_EXTENSIONS = {'mov', 'mp4'}


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/images/<path:path>')
def send_image(path):
    return send_from_directory(OUTPUT_IMAGES_BASE_PATH, path)


def generate_output_item(prediction, request_uuid, file_ext):
    timestamp = [int(s) for s in re.findall(r'\d+', prediction['image_id'])][0]
    return {
        'image_id': "/images/{}/{}.{}".format(request_uuid, prediction['image_id'], file_ext),
        'mean_score_prediction': prediction['mean_score_prediction'],
        'timestamp': timestamp
    }


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/generate', methods=['POST'])
def generate_highlights():
    MAX_FRAMES = 500
    MIN_TIME_STAMP = 1
    TIME_FRAME = 1 * 60
    frame_skip_count = 1
    file_ext = 'jpg'
    PREDICTIONS_LIMIT = 15
    BASE_MODEL = 'MobileNet'
    AESTHETIC_WEIGHTS_FILE = 'weights_mobilenet_aesthetic_0.07.hdf5'
    TECHNICAL_WEIGHTS_FILE = 'weights_mobilenet_technical_0.11.hdf5'

    request_uid = utils.rand_gen()
    path_format = '{}/{}'
    video_path = path_format.format(VIDEOS_BASE_PATH, request_uid)
    images_path = path_format.format(IMAGES_BASE_PATH, request_uid)
    output_images_path = path_format.format(OUTPUT_IMAGES_BASE_PATH, request_uid)
    # predictions_path = path_format.format(PREDICTIONS_BASE_PATH, request_uid)
    for dir_path in [video_path, images_path, output_images_path]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    print('[DEBUG] Created required directories')

    file = request.files['file']
    if not file or not allowed_file(file.filename):
        print('[INFO] No file uploaded')
        return json.dumps([], indent=2)

    video_name = secure_filename(file.filename)
    video_file_path = os.path.join(video_path, video_name)
    file.save(video_file_path)
    print('[DEBUG] Saved file at {}'.format(video_file_path))

    video_cap = cv2.VideoCapture(video_file_path)
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    frame_count = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = math.ceil(frame_count / fps)
    if duration > TIME_FRAME:
        duration = TIME_FRAME
    print('[DEBUG] Duration', duration)
    time_skip = max(MIN_TIME_STAMP, math.ceil(duration / MAX_FRAMES))
    print('[DEBUG] Time Skip', time_skip)
    frame_skip_count = max(frame_skip_count, math.ceil(time_skip * fps))
    print('[DEBUG] Sampling the video after {} frames'.format(frame_skip_count))

    success, image = video_cap.read()
    timestamp = int(video_cap.get(cv2.CAP_PROP_POS_MSEC))
    iteration = 1
    count = 1

    while success and timestamp <= iteration * duration * 1000:
        cur_path_format = '{}/{}_{}'
        cur_images_path = cur_path_format.format(IMAGES_BASE_PATH, request_uid, iteration)
        # cur_predictions_path = cur_path_format.format(PREDICTIONS_BASE_PATH, request_uid, iteration)

        for dir_path in [cur_images_path]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
        print('[Iteration {}] Created required directories'.format(iteration))

        while success and timestamp <= iteration * duration * 1000:
            if count % frame_skip_count == 0:
                cv2.imwrite("{}/frame_{}.{}".format(cur_images_path, timestamp, file_ext), image)
                print('[DEBUG] timestamp={} count={} saved'.format(timestamp, count))
            success, image = video_cap.read()
            timestamp = int(video_cap.get(cv2.CAP_PROP_POS_MSEC))
            count += 1
        print('[Iteration {}] Completed sampling and saving the frames'.format(iteration))

        if len(os.listdir(cur_images_path)) == 0:
            print('[Iteration {}] Empty {} and so no prediction'.format(iteration, cur_images_path))
        else:
            print('[Iteration {}] Running {} to predict the scores of the sampled frames'.format(iteration, BASE_MODEL))
            predictions = predict.score(BASE_MODEL, TECHNICAL_WEIGHTS_FILE, cur_images_path, None)
            print('[Iteration {}] Completed prediction of frames'.format(iteration))

            prediction = max(predictions, key=lambda k: k['mean_score_prediction'])
            print('[Iteration {}] Top {} Results:'.format(iteration, PREDICTIONS_LIMIT))
            cur_location = '{}/{}.{}'.format(cur_images_path, prediction['image_id'], file_ext)
            new_location = '{}/{}.{}'.format(images_path, prediction['image_id'], file_ext)
            shutil.move(cur_location, new_location)
            print('[Iteration {}] Moved {} to {}:'.format(iteration, cur_location, new_location))
        shutil.rmtree(cur_images_path)
        iteration += 1

    video_cap.release()
    shutil.rmtree(video_path)
    if len(os.listdir(images_path)) == 0:
        print('[Final] Empty {} and so no prediction'.format(images_path))
        predictions = []
    else:
        # predictions_file = '{}/predicts.json'.format(predictions_path)
        predictions = predict.score(BASE_MODEL, AESTHETIC_WEIGHTS_FILE, images_path, None)
        print('[Final] Completed prediction of frames')
        predictions = sorted(predictions, key=lambda k: k['mean_score_prediction'], reverse=True)
        predictions = predictions[:PREDICTIONS_LIMIT]
        print('[Final] Top {} Results:'.format(PREDICTIONS_LIMIT))
        print(json.dumps(predictions, indent=2))

    for prediction in predictions:
        cur_location = '{}/{}.{}'.format(images_path, prediction['image_id'], file_ext)
        new_location = '{}/{}.{}'.format(output_images_path, prediction['image_id'], file_ext)
        shutil.move(cur_location, new_location)
        print('[Final] Moved {} to {}:'.format(cur_location, new_location))
    shutil.rmtree(images_path)
    predictions = list(map(lambda prediction: generate_output_item(prediction, request_uid, file_ext), predictions))
    predictions = sorted(predictions, key=lambda k: k['timestamp'])
    print('[Final] Response:')
    print(json.dumps(predictions, indent=2))

    return json.dumps(predictions, indent=2)


if __name__ == '__main__':
    app.run()
