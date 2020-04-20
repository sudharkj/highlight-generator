import math
import os
import shutil
import time

import cv2
from flask import Blueprint, current_app, request, send_from_directory

from generator import get_clip_frames, utils, extract_predicted_frames, append_timestamp, get_predictions

# constraints on the params
MIN_CLIP_TIME = 1
MIN_IMAGES_PER_CLIP = 1
DEFAULT_IMAGE_EXTENSION = utils.SUPPORTED_IMAGE_EXTENSIONS[0]
DEFAULT_MODE = utils.SUPPORTED_MODES[0]

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


@highlights.route('/modes', methods=['GET'])
def get_modes():
    return utils.get_print_string({
        'modes': utils.SUPPORTED_MODES
    })


def generate_output_item(prediction, request_uuid, image_extension):
    prediction = append_timestamp(prediction)
    return {
        'imageUrl': "/highlights/images/{}/{}.{}".format(request_uuid, prediction['image_id'], image_extension),
        'meanScorePrediction': prediction['mean_score_prediction'],
        'timestamp': prediction['timestamp']
    }


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
    # clip_time
    clip_time = utils.get_validated_arg(request.form, 'clip_time', int, MIN_CLIP_TIME)  # in minutes
    clip_time = max(clip_time, MIN_CLIP_TIME)  # in seconds
    # images_per_clip
    images_per_clip = utils.get_validated_arg(request.form, 'images_per_clip', int, MIN_IMAGES_PER_CLIP)
    images_per_clip = max(images_per_clip, MIN_IMAGES_PER_CLIP)
    # mode
    mode = utils.get_validated_arg(request.form, 'mode', str, DEFAULT_MODE)
    mode = mode.lower()
    mode = mode if utils.is_supported_mode(mode) else DEFAULT_MODE
    # image_extension
    image_extension = utils.get_validated_arg(request.form, 'image_extension', str, DEFAULT_IMAGE_EXTENSION)
    image_extension = image_extension.lower()
    image_extension = image_extension if utils.is_supported_image_type(image_extension) else DEFAULT_IMAGE_EXTENSION
    current_app.logger.debug("Values used for generating highlights")
    current_app.logger.debug("clip_time: {}".format(clip_time))
    current_app.logger.debug("images_per_clip: {}".format(images_per_clip))
    current_app.logger.debug("mode: {}".format(mode))
    current_app.logger.debug("image_extension: {}".format(image_extension))

    # download the video
    video_file_path = utils.save_uploaded_file(video_file, temp_videos_path)

    video_cap = cv2.VideoCapture(video_file_path)
    frames_per_second = video_cap.get(cv2.CAP_PROP_FPS)
    frame_count = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
    total_time = frame_count / frames_per_second
    total_clips = math.ceil(total_time / (clip_time * 60))
    video_cap.release()

    states = [{
        'request_uid': request_uid,
        'mode': mode,
        'clip_id': clip_id + 1,
        'video_file_path': video_file_path,
        'clip_time': clip_time,
        'images_per_clip': images_per_clip,
        'frames_per_second': frames_per_second,
        'frame_count': frame_count,
        'total_time': total_time,
        'temp_images_path': temp_images_path,
        'image_extension': image_extension,
    } for clip_id in range(total_clips)]
    # manual task assignment in python is making tensorflow to run on cpu even when gpu is available
    # so implemented sequential requests and removed task scheduling on threads that uses ProcessPoolExecutioner
    is_failure = [not get_clip_frames(state) for state in states]
    if any(is_failure):
        return utils.get_print_string({
            'statusText': 'Internal Server Error'
        }), 500

    tag = "[Final]"
    predictions = get_predictions(
        tag, temp_images_path, current_app.config['AESTHETIC_WEIGHTS_FILE_PATH']
    )
    shutil.rmtree(temp_images_path)
    shutil.rmtree(temp_predictions_path)

    predictions = list(
        map(lambda prediction: generate_output_item(prediction, request_uid, image_extension), predictions)
    )
    predictions = sorted(predictions, key=lambda k: k['timestamp'])

    # extract original quality images
    extract_predicted_frames(predictions, video_file_path, output_images_path, image_extension)
    shutil.rmtree(temp_videos_path)

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
