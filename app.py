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
    BASE_MODEL = 'MobileNet'
    AESTHETIC_WEIGHTS_FILE = 'weights_mobilenet_aesthetic_0.07.hdf5'
    TECHNICAL_WEIGHTS_FILE = 'weights_mobilenet_technical_0.11.hdf5'

    MIN_TIME_FRAME = 1 * 60
    MIN_SAMPLING_RATE = 1 * 60
    MIN_SCENE_IMAGES = 1
    MIN_SUMMARY_IMAGES = 10
    MIN_FRAME_SKIP_COUNT = 1

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

    file = request.files['video']
    if not file or not allowed_file(file.filename):
        print('[INFO] No file uploaded')
        return json.dumps([], indent=2)

    # get other params
    time_frame = int(request.form['time-frame'])  # in minutes
    sampling_rate = int(request.form['sampling-rate'])
    scene_images = int(request.form['scene-images'])
    summary_images = int(request.form['summary-images'])
    file_ext = 'jpg'
    print('[DEBUG] time_frame', time_frame)
    print('[DEBUG] sampling_rate', sampling_rate)
    print('[DEBUG] scene_images', scene_images)
    print('[DEBUG] summary_images', summary_images)

    # make sure that the request-params are within the limits
    time_frame = max(time_frame * 60, MIN_TIME_FRAME)
    sampling_rate = max(sampling_rate, MIN_SAMPLING_RATE)
    scene_images = max(scene_images, MIN_SCENE_IMAGES)
    summary_images = max(summary_images, MIN_SUMMARY_IMAGES)
    print('[DEBUG] After ensuring the limits')
    print('[DEBUG] time_frame', time_frame)
    print('[DEBUG] sampling_rate', sampling_rate)
    print('[DEBUG] scene_images', scene_images)
    print('[DEBUG] summary_images', summary_images)

    video_name = secure_filename(file.filename)
    video_file_path = os.path.join(video_path, video_name)
    file.save(video_file_path)
    print('[DEBUG] Saved file at {}'.format(video_file_path))

    video_cap = cv2.VideoCapture(video_file_path)
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    frame_count = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = math.ceil(frame_count / fps)
    if duration > time_frame:
        duration = time_frame
    print('[DEBUG] Duration', duration)
    sampling_rate = min(sampling_rate, math.floor(duration * fps))
    print('[DEBUG] sampling_rate', sampling_rate)
    frame_skip_count = max(MIN_FRAME_SKIP_COUNT, math.ceil(duration * fps / sampling_rate))
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
            predictions = []
        else:
            print('[Iteration {}] Running {} to predict the scores of the sampled frames'.format(iteration, BASE_MODEL))
            predictions = predict.score(BASE_MODEL, TECHNICAL_WEIGHTS_FILE, cur_images_path, None)
            print('[Iteration {}] Completed prediction of frames'.format(iteration))

            predictions = sorted(predictions, key=lambda k: k['mean_score_prediction'], reverse=True)
            predictions = predictions[:scene_images]
            print('[Iteration {}] Top {} Results:'.format(iteration, scene_images))
            print(json.dumps(predictions, indent=2))

        for prediction in predictions:
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
        predictions = predictions[:summary_images]
        print('[Final] Top {} Results:'.format(summary_images))
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
