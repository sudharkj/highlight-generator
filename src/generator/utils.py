import json
import os
import random
import shutil
import string

from flask import current_app
from werkzeug.utils import secure_filename

SUPPORTED_VIDEO_EXTENSIONS = ['mov', 'mp4']
SUPPORTED_IMAGE_EXTENSIONS = ['jpg']
SUPPORTED_MODES = ['human_eye', 'scene_detect']


def rand_gen(size=8, chars=string.ascii_lowercase + string.digits):
    random.seed()
    return ''.join(random.choice(chars) for _ in range(size))


def create_dirs(dirs_path, app_logger, tag=""):
    for dir_path in dirs_path:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)
    app_logger.debug("{} Created directories: {}".format(tag, dirs_path))


def is_supported_image_type(image_type):
    return image_type.lower() in SUPPORTED_VIDEO_EXTENSIONS


def is_supported_mode(mode):
    return mode.lower() in SUPPORTED_MODES


def get_param_value(request_form, request_param):
    if request_param['name'] in request_form:
        try:
            return_value = request_param['data_type'](request_form[request_param['name']])
        except ValueError as vErr:
            current_app.logger.error(vErr)
            return request_param['allowed'][0]

        return_value = return_value.lower() if request_param['data_type'] is str else return_value
        return return_value if return_value in request_param['allowed'] else request_param['allowed'][0]
    return request_param['allowed'][0]


def save_uploaded_file(file, base_path):
    file_name = secure_filename(file.filename)
    file_path = os.path.join(base_path, file_name)
    file.save(file_path)
    current_app.logger.debug('Saved uploaded file at {}'.format(file_path))
    return file_path


def get_print_string(json_object):
    return json.dumps(json_object, indent=2)
