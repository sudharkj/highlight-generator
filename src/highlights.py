import math
import os
import re
import shutil

import cv2
from flask import Blueprint, current_app, request, send_from_directory

from nima import utils, predict
from utils import create_base_dirs, is_supported_video_type, get_validated_arg, is_supported_image_type, save_file, \
    create_scene_dirs, get_print_string

BASE_MODEL = 'MobileNet'
# constraints on the params
MIN_TIME_FRAME = 1
MIN_SAMPLING_RATE = 1 * 60
MIN_SCENE_IMAGES = 1
MIN_SUMMARY_IMAGES = 10
MIN_FRAME_SKIP_COUNT = 1
DEFAULT_IMAGE_EXTENSION = 'jpg'

highlights = Blueprint("highlights", __name__, url_prefix="/highlights")


@highlights.route("/")
def hello_highlights():
    return "Hello, Highlights!"


def get_frame_skip_count(video_cap, time_frame, sampling_rate):
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    frame_count = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = math.ceil(frame_count / fps)
    if duration > time_frame:
        duration = time_frame
    sampling_rate = min(sampling_rate, math.floor(duration * fps))
    return max(MIN_FRAME_SKIP_COUNT, math.ceil(duration * fps / sampling_rate))


def get_predictions(tag, images_path, weights_file_path, prediction_limit):
    if len(os.listdir(images_path)) == 0:
        predictions = []
    else:
        current_app.logger.debug("{} Running {} to predict the scores".format(tag, BASE_MODEL))
        predictions = predict.score(BASE_MODEL, weights_file_path, images_path, None)
        current_app.logger.debug("{} Completed prediction of scores".format(tag))
        predictions = sorted(predictions, key=lambda k: k['mean_score_prediction'], reverse=True)
        predictions = predictions[:prediction_limit]
    current_app.logger.debug("{} Top {} Results: {}".format(tag, prediction_limit, get_print_string(predictions)))
    return predictions


def save_only_best_images(tag, predictions, cur_images_path, new_images_path, image_extension):
    for prediction in predictions:
        cur_location, new_location = tuple(map(
            lambda dir_path: '{}/{}.{}'.format(dir_path, prediction['image_id'], image_extension),
            [cur_images_path, new_images_path])
        )
        shutil.move(cur_location, new_location)
        current_app.logger.debug("{} Moved {} to {}:".format(tag, cur_location, new_location))
    shutil.rmtree(cur_images_path)


def generate_output_item(prediction, request_uuid, image_extension):
    timestamp = list(map(lambda s: int(s), re.findall(r'\d+', prediction['image_id'])))[0]
    return {
        'image_id': "/highlights/images/{}/{}.{}".format(request_uuid, prediction['image_id'], image_extension),
        'mean_score_prediction': prediction['mean_score_prediction'],
        'timestamp': timestamp
    }


@highlights.route('/generate', methods=['POST'])
def generate_highlights():
    # return immediately if video is not sent in the request
    if 'video' not in request.files or not is_supported_video_type(request.files['video'].filename):
        current_app.logger.info('No file uploaded')
        return get_print_string([])
    video_file = request.files['video']

    # create base folders and get their paths
    request_uid = utils.rand_gen()
    base_folders = create_base_dirs(current_app, request_uid)
    temp_videos_path, temp_images_path, temp_predictions_path, output_images_path = base_folders

    # gracefully get other params
    # time_frame
    time_frame = get_validated_arg(current_app, request.form, 'time-frame', int, MIN_TIME_FRAME)  # in minutes
    time_frame = max(time_frame, MIN_TIME_FRAME) * 60  # in seconds
    # sampling_rate
    sampling_rate = get_validated_arg(current_app, request.form, 'sampling-rate', int, MIN_SAMPLING_RATE)
    sampling_rate = max(sampling_rate, MIN_SAMPLING_RATE)
    # scene_images
    scene_images = get_validated_arg(current_app, request.form, 'scene-images', int, MIN_SCENE_IMAGES)
    scene_images = max(scene_images, MIN_SCENE_IMAGES)
    # summary_images
    summary_images = get_validated_arg(current_app, request.form, 'summary-images', int, MIN_SUMMARY_IMAGES)
    summary_images = max(summary_images, MIN_SUMMARY_IMAGES)
    # image_extension
    image_extension = get_validated_arg(current_app, request.form, 'image-extension', str, DEFAULT_IMAGE_EXTENSION)
    image_extension = image_extension.lower() if is_supported_image_type(image_extension) else DEFAULT_IMAGE_EXTENSION
    current_app.logger.debug("Values used for generating highlights")
    current_app.logger.debug("time_frame: {}".format(time_frame))
    current_app.logger.debug("sampling_rate: {}".format(sampling_rate))
    current_app.logger.debug("scene_images: {}".format(scene_images))
    current_app.logger.debug("summary_images: {}".format(summary_images))
    current_app.logger.debug("image_extension: {}".format(image_extension))

    # download the video
    video_file_path = save_file(current_app, video_file, temp_videos_path)

    # figuring out frame_skip_count value
    video_cap = cv2.VideoCapture(video_file_path)
    frame_skip_count = get_frame_skip_count(video_cap, time_frame, sampling_rate)
    current_app.logger.debug("Sampling video after {} frames".format(frame_skip_count))

    success, image = video_cap.read()
    timestamp = int(video_cap.get(cv2.CAP_PROP_POS_MSEC))
    scene = 1
    count = 1

    while success and timestamp <= scene * time_frame * 1000:
        cur_images_path, cur_predictions_path = create_scene_dirs(current_app, request_uid, scene)
        cur_tag = "[Scene {}]".format(scene)

        while success and timestamp <= scene * time_frame * 1000:
            if count % frame_skip_count == 0:
                cur_image_file_name = '{}/frame_{}.{}'.format(cur_images_path, timestamp, image_extension)
                cv2.imwrite(cur_image_file_name, image)
                current_app.logger.debug(
                    "{} timestamp={} count={} saved at {}".format(cur_tag, timestamp, count, cur_image_file_name)
                )
            success, image = video_cap.read()
            timestamp = int(video_cap.get(cv2.CAP_PROP_POS_MSEC))
            count += 1
        current_app.logger.debug("{} Completed sampling and saving the frames".format(cur_tag))

        predictions = get_predictions(
            cur_tag, cur_images_path, current_app.config['TECHNICAL_WEIGHTS_FILE_PATH'], scene_images
        )
        save_only_best_images(cur_tag, predictions, cur_images_path, temp_images_path, image_extension)
        shutil.rmtree(cur_predictions_path)
        scene += 1

    # release video handles and delete it with the containing folder
    video_cap.release()
    shutil.rmtree(temp_videos_path)

    cur_tag = "[Final]"
    predictions = get_predictions(
        cur_tag, temp_images_path, current_app.config['AESTHETIC_WEIGHTS_FILE_PATH'], summary_images
    )
    save_only_best_images(cur_tag, predictions, temp_images_path, output_images_path, image_extension)
    shutil.rmtree(temp_predictions_path)

    predictions = list(
        map(lambda prediction: generate_output_item(prediction, request_uid, image_extension), predictions)
    )
    predictions = sorted(predictions, key=lambda k: k['timestamp'])
    current_app.logger.debug("{} Response: {}".format(cur_tag, get_print_string(predictions)))

    return get_print_string(predictions)


@highlights.route('/images/<path:path>')
def send_image(path):
    # send_from_directory does not work with relative path
    # so, get the absolute path of the folder and send that value for the directory location
    base_dir = os.path.abspath(current_app.config['OUTPUT_IMAGES_PATH'])
    return send_from_directory(base_dir, path)
