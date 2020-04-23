import os
import shutil
import time

from flask import Blueprint, current_app, request, send_from_directory

from generator import utils, get_predictions

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


def generate_result(prediction, request_uuid, image_extension):
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

    # create request directories
    request_uid = utils.rand_gen()
    request_dirs = [
        current_app.config['TEMP_VIDEOS_PATH'],
        current_app.config['TEMP_IMAGES_PATH'],
        current_app.config['OUTPUT_IMAGES_PATH']
    ]
    request_dirs = list(map(lambda base_path: '{}/{}'.format(base_path, request_uid), request_dirs))
    temp_videos_path, temp_images_path, output_images_path = request_dirs
    utils.create_dirs(request_dirs, current_app.logger, "[{}]".format(request_uid))

    # download the video
    video_file_path = utils.save_uploaded_file(video_file, temp_videos_path)

    state = {
        'request_uid': request_uid,
        'mode': mode,
        'video_file_path': video_file_path,
        'clip_time': clip_time,
        'images_per_clip': images_per_clip,
        'threshold': 0.90,
        'temp_images_path': temp_images_path,
        'image_extension': image_extension,
        'predicts_path': output_images_path,
        'is_verbose': os.environ.get("FLASK_DEBUG", default=0)
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
