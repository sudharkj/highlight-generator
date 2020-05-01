import os
import shutil
import time

from flask import Blueprint, current_app, request, send_from_directory

from generator import utils, get_predictions
from generator.utils import SUPPORTED_MODES, SUPPORTED_IMAGE_EXTENSIONS, SUPPORTED_VIDEO_EXTENSIONS

highlights = Blueprint("highlights", __name__, url_prefix="/highlights")


@highlights.route("/")
def hello_highlights():
    return "Hello, Highlights!"


def is_supported_video_type(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in SUPPORTED_VIDEO_EXTENSIONS


def generate_result(prediction, request_uuid, image_extension):
    return {
        'imageUrl': "/highlights/images/{}/{}.{}".format(request_uuid, prediction['image_id'], image_extension),
        'meanScorePrediction': prediction['mean_score_prediction'],
        'timestamp': prediction['timestamp']
    }


@highlights.route('/generate', methods=['POST'])
def generate_highlights():
    start_time = time.time()
    request_uid = utils.rand_gen()
    tag = "[{}]".format(request_uid)
    # return immediately if video is not sent in the request
    if 'video' not in request.files or not is_supported_video_type(request.files['video'].filename):
        current_app.logger.info("{} No file uploaded".format(tag))
        return utils.get_print_string({
            'predictions': [],
            'timeTaken': time.time() - start_time
        })
    video_file = request.files['video']

    # gracefully get params from form
    mode = utils.get_param_value(request.form, {
        'name': "mode",
        'data_type': str,
        'allowed': SUPPORTED_MODES
    })
    images_per_clip = utils.get_param_value(request.form, {
        'name': "images_per_clip",
        'data_type': int,
        'allowed': list(range(1, 6, 1))
    })
    image_extension = utils.get_param_value(request.form, {
        'name': "image_extension",
        'data_type': str,
        'allowed': SUPPORTED_IMAGE_EXTENSIONS
    })
    total_clips = utils.get_param_value(request.form, {
        'name': "total_clips",
        'data_type': int,
        'allowed': list(range(1, 26, 1))
    })
    current_app.logger.debug("Values used for generating highlights")
    current_app.logger.debug("{} mode: {}".format(tag, mode))
    current_app.logger.debug("{} images_per_clip: {}".format(tag, images_per_clip))
    current_app.logger.debug("{} image_extension: {}".format(tag, image_extension))
    current_app.logger.debug("{} total_clips: {}".format(tag, total_clips))

    # create request directories
    request_dirs = [
        current_app.config['TEMP_VIDEOS_PATH'],
        current_app.config['TEMP_IMAGES_PATH'],
        current_app.config['OUTPUT_IMAGES_PATH']
    ]
    request_dirs = list(map(lambda base_path: '{}/{}'.format(base_path, request_uid), request_dirs))
    temp_videos_path, temp_images_path, output_images_path = request_dirs
    utils.create_dirs(request_dirs, current_app.logger, tag)

    # download the video
    video_file_path = utils.save_uploaded_file(video_file, temp_videos_path)

    state = {
        'request_uid': request_uid,
        'mode': mode,
        'video_file_path': video_file_path,
        'total_clips': total_clips,
        'images_per_clip': images_per_clip,
        'temp_images_path': temp_images_path,
        'image_extension': image_extension,
        'predicts_path': output_images_path
    }

    predictions = get_predictions(state)
    predictions = list(map(lambda prediction: generate_result(prediction, request_uid, image_extension), predictions))
    # delete request temp images directory
    shutil.rmtree(temp_images_path)
    # delete request video directory
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
