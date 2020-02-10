import json
import os
import random
import shutil
import string

from flask import current_app
from werkzeug.utils import secure_filename

SUPPORTED_VIDEO_EXTENSIONS = ['mov', 'mp4']
SUPPORTED_IMAGE_EXTENSIONS = ['jpg']


def rand_gen(size=32, chars=string.ascii_uppercase + string.digits):
    random.seed()
    return ''.join(random.choice(chars) for _ in range(size))


def get_dirs_paths(config):
    return [
        config['TEMP_VIDEOS_PATH'],
        config['TEMP_IMAGES_PATH'],
        config['TEMP_PREDICTIONS_PATH'],
        config['OUTPUT_IMAGES_PATH']
    ]


def reset_generated_dirs(dirs_path):
    # NOTE that the app context is available only when it is serving a request.
    # So, sending dirs_path instead
    for dir_path in dirs_path:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)


def create_base_dirs(request_uid):
    dirs_path = get_dirs_paths(current_app.config)
    dirs_path = list(map(lambda base_path: '{}/{}'.format(base_path, request_uid), dirs_path))
    # this is the first time creating the folder and so create it without any checks
    for dir_path in dirs_path:
        os.makedirs(dir_path)
    current_app.logger.debug("[Base] Created directories: {}".format(dirs_path))
    return tuple(dirs_path)


def create_scene_dirs(request_uid, scene):
    dirs_path = list(map(
        lambda base_path: '{}/{}_{}'.format(base_path, request_uid, scene),
        [current_app.config['TEMP_IMAGES_PATH'], current_app.config['TEMP_PREDICTIONS_PATH']]
    ))
    # this is the first time creating the folder and so create it without any checks
    for dir_path in dirs_path:
        os.makedirs(dir_path)
    current_app.logger.debug("[Scene {}] Created directories: {}".format(scene, dirs_path))
    return tuple(dirs_path)


def is_supported_video_type(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in SUPPORTED_VIDEO_EXTENSIONS


def is_supported_image_type(image_type):
    return image_type.lower() in SUPPORTED_VIDEO_EXTENSIONS


def get_validated_arg(request_form, request_param, data_type, default_value=None):
    return_value = default_value
    if request_param in request_form:
        try:
            return_value = data_type(request_form[request_param])
        except ValueError as vErr:
            current_app.logger.error(vErr)
            return_value = default_value
    return return_value


def save_uploaded_file(file, base_path):
    file_name = secure_filename(file.filename)
    file_path = os.path.join(base_path, file_name)
    file.save(file_path)
    current_app.logger.debug('Saved uploaded file at {}'.format(file_path))
    return file_path


def get_print_string(json_object):
    return json.dumps(json_object, indent=2)
