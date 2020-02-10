import math
import os
import random
import re
import shutil
import time
from concurrent.futures.process import ProcessPoolExecutor

import cv2
from flask import Blueprint, current_app, request, send_from_directory
from tqdm import tqdm

import nima
import utils

BASE_MODEL = 'MobileNet'
# constraints on the params
MIN_TIME_FRAME = 1
MIN_SCENE_IMAGES = 1
DEFAULT_IMAGE_EXTENSION = utils.SUPPORTED_IMAGE_EXTENSIONS[0]
# human eye params
RAND_INT_START = 2/3
RAND_INT_END = 1

highlights = Blueprint("highlights", __name__, url_prefix="/highlights")


@highlights.route("/")
def hello_highlights():
    return "Hello, Highlights!"


@highlights.route('/video-types', methods=['GET'])
def get_video_types():
    return utils.get_print_string({
        'videoTypes': utils.SUPPORTED_VIDEO_EXTENSIONS
    })


@highlights.route('/image-types', methods=['GET'])
def get_image_types():
    return utils.get_print_string({
        'imageTypes': utils.SUPPORTED_IMAGE_EXTENSIONS
    })


def get_frame_skip_limit(frames_per_second, factor):
    end = max(RAND_INT_START, RAND_INT_END)
    end = frames_per_second * end * factor
    end = math.ceil(end)

    start = min(RAND_INT_START, RAND_INT_END)
    start = frames_per_second * start
    start = end - math.floor(start)

    random.seed()
    return random.randint(start, end)


def get_predictions(tag, images_path, weights_file_path, prediction_limit):
    predictions = []
    if len(os.listdir(images_path)) > 0:
        current_app.logger.debug("{} Running {} to predict the scores".format(tag, BASE_MODEL))
        debug_on = os.environ.get("FLASK_DEBUG", default=0)
        predictions = nima.score(BASE_MODEL, weights_file_path, images_path, is_verbose=debug_on)
        current_app.logger.debug("{} Completed prediction of scores".format(tag))
        predictions = sorted(predictions, key=lambda k: k['mean_score_prediction'], reverse=True)
        predictions = predictions[:prediction_limit]
    current_app.logger.debug("{} Top {} Results: {}".format(tag, prediction_limit, utils.get_print_string(predictions)))
    return predictions


def save_only_best_images(tag, predictions, cur_images_path, new_images_path, image_extension):
    current_app.logger.debug("{} Moving best images from {} to {}:".format(tag, cur_images_path, new_images_path))
    for prediction in tqdm(predictions, desc="{} Moving".format(tag)):
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
        'imageUrl': "/highlights/images/{}/{}.{}".format(request_uuid, prediction['image_id'], image_extension),
        'meanScorePrediction': prediction['mean_score_prediction'],
        'timestamp': timestamp
    }


def get_technical_predictions(state):
    try:
        request_uid = state['request_uid']
        video_file_path = state['video_file_path']
        time_frame = state['time_frame']  # in minutes
        image_extension = state['image_extension']
        scene_images = state['scene_images']
        temp_images_path = state['temp_images_path']
        scene = state['scene']

        cur_images_path, cur_predictions_path = utils.create_scene_dirs(request_uid, scene)
        cur_tag = "[Scene {}]".format(scene)

        video_cap = cv2.VideoCapture(video_file_path)
        frames_per_second = video_cap.get(cv2.CAP_PROP_FPS)
        frame_count = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        total_time = math.floor(frame_count / frames_per_second)
        scene_start_time = int((scene-1) * time_frame * 60) + 1
        scene_end_time = int(scene * time_frame * 60)
        scene_end_time = min(scene_end_time, total_time)
        frames_in_scene = int(frames_per_second * (scene_end_time - scene_start_time - 1))

        current_app.logger.debug("{} Saving {} frames with random sampling".format(cur_tag, frames_in_scene))
        video_cap.set(cv2.CAP_PROP_POS_MSEC, scene_start_time * 1000)
        count = 0
        with tqdm(total=frames_in_scene, desc="{} Processing".format(cur_tag)) as frame_bar:
            for frame_id in range(frames_in_scene):
                success, image = video_cap.read()
                frame_skip_limit = get_frame_skip_limit(frames_per_second, time_frame)
                count += 1

                if success:
                    frame_bar.update(1)
                    if count >= frame_skip_limit:
                        count = 0
                        timestamp = int(video_cap.get(cv2.CAP_PROP_POS_MSEC))
                        cur_image_file_name = '{}/frame_{}.{}'.format(cur_images_path, timestamp, image_extension)
                        cv2.imwrite(cur_image_file_name, image)
                else:
                    pending_frames = frames_in_scene - frame_id
                    if pending_frames > 1:
                        current_app.logger.error("{} Unable to read {} frames".format(cur_tag, pending_frames-1))
                    frame_bar.update(pending_frames)
        current_app.logger.debug("{} Completed saving frames with random sampling".format(cur_tag))

        predictions = get_predictions(
            cur_tag, cur_images_path, current_app.config['TECHNICAL_WEIGHTS_FILE_PATH'], scene_images
        )
        save_only_best_images(cur_tag, predictions, cur_images_path, temp_images_path, image_extension)
        shutil.rmtree(cur_predictions_path)

        # release video handles and delete it with the containing folder
        video_cap.release()

    except Exception as ex:
        # any exception occurred in child process puts the parent process in deadlock
        # catching the exception to avoid that
        # https://stackoverflow.com/a/51076484
        current_app.logger.error("Exception occurred in get_technical_predictions: {}".format(ex))


@highlights.route('/generate', methods=['POST'])
def generate_highlights():
    start_time = time.time()
    # return immediately if video is not sent in the request
    if 'video' not in request.files or not utils.is_supported_video_type(request.files['video'].filename):
        current_app.logger.info('No file uploaded')
        return utils.get_print_string({
            'predictions': [],
            'timeTaken': time.time() - start_time
        })
    video_file = request.files['video']

    # create base folders and get their paths
    request_uid = utils.rand_gen()
    base_folders = utils.create_base_dirs(request_uid)
    temp_videos_path, temp_images_path, temp_predictions_path, output_images_path = base_folders

    # gracefully get other params
    # time_frame
    time_frame = utils.get_validated_arg(request.form, 'time-frame', int, MIN_TIME_FRAME)  # in minutes
    time_frame = max(time_frame, MIN_TIME_FRAME)  # in seconds
    # scene_images
    scene_images = utils.get_validated_arg(request.form, 'scene-images', int, MIN_SCENE_IMAGES)
    scene_images = max(scene_images, MIN_SCENE_IMAGES)
    # image_extension
    image_extension = utils.get_validated_arg(request.form, 'image-extension', str, DEFAULT_IMAGE_EXTENSION)
    image_extension = image_extension.lower()
    image_extension = image_extension if utils.is_supported_image_type(image_extension) else DEFAULT_IMAGE_EXTENSION
    current_app.logger.debug("Values used for generating highlights")
    current_app.logger.debug("time_frame: {}".format(time_frame))
    current_app.logger.debug("scene_images: {}".format(scene_images))
    current_app.logger.debug("image_extension: {}".format(image_extension))

    # download the video
    video_file_path = utils.save_uploaded_file(video_file, temp_videos_path)

    video_cap = cv2.VideoCapture(video_file_path)
    frames_per_second = video_cap.get(cv2.CAP_PROP_FPS)
    frame_count = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
    total_time = frame_count / frames_per_second
    total_scenes = math.ceil(total_time / (time_frame * 60))

    states = [{
        'request_uid': request_uid,
        'video_file_path': video_file_path,
        'time_frame': time_frame,
        'image_extension': image_extension,
        'scene_images': scene_images,
        'temp_images_path': temp_images_path,
        'scene': scene + 1
    } for scene in range(total_scenes)]

    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    chunk_size = math.ceil(total_scenes / cpu_count)

    with ProcessPoolExecutor(max_workers=cpu_count) as executor:
        # larger chunksize improves performance
        # ref: https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Executor.map
        for _ in executor.map(get_technical_predictions, states, chunksize=chunk_size):
            # If a func call raises an exception,
            # then that exception will be raised when its value is retrieved from the iterator.
            # ref: https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Executor.map
            # hack to know all the exceptions
            continue

    # force non deletion of video until the end
    video_cap.release()
    shutil.rmtree(temp_videos_path)

    cur_tag = "[Final]"
    predictions = get_predictions(
        cur_tag, temp_images_path, current_app.config['AESTHETIC_WEIGHTS_FILE_PATH'], total_scenes * scene_images
    )
    save_only_best_images(cur_tag, predictions, temp_images_path, output_images_path, image_extension)
    shutil.rmtree(temp_predictions_path)

    predictions = list(
        map(lambda prediction: generate_output_item(prediction, request_uid, image_extension), predictions)
    )
    predictions = sorted(predictions, key=lambda k: k['timestamp'])

    return utils.get_print_string({
        'predictions': predictions,
        'timeTaken': time.time() - start_time
    })


@highlights.route('/images/<path:path>')
def send_image(path):
    # send_from_directory does not work with relative path
    # so, get the absolute path of the folder and send that value for the directory location
    base_dir = os.path.abspath(current_app.config['OUTPUT_IMAGES_PATH'])
    return send_from_directory(base_dir, path)
