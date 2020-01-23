import json
import os
import shutil

from werkzeug.utils import secure_filename

SUPPORTED_VIDEO_EXTENSIONS = ['mov', 'mp4']
SUPPORTED_IMAGE_EXTENSIONS = ['jpg']


def get_dirs_paths(cur_app_cfg):
    return [
        cur_app_cfg['TEMP_VIDEOS_PATH'],
        cur_app_cfg['TEMP_IMAGES_PATH'],
        cur_app_cfg['TEMP_PREDICTIONS_PATH'],
        cur_app_cfg['OUTPUT_IMAGES_PATH']
    ]


def reset_generated_dirs(cur_app):
    dirs_path = get_dirs_paths(cur_app.config)
    for dir_path in dirs_path:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
    cur_app.logger.debug("Deleted directories: {}".format(dirs_path))


def create_base_dirs(cur_app, request_uid):
    dirs_path = get_dirs_paths(cur_app.config)
    dirs_path = list(map(lambda base_path: '{}/{}'.format(base_path, request_uid), dirs_path))
    # this is the first time creating the folder and so create it without any checks
    for dir_path in dirs_path:
        os.makedirs(dir_path)
    cur_app.logger.debug("Created directories: {}".format(dirs_path))
    return tuple(dirs_path)


def create_scene_dirs(cur_app, request_uid, scene):
    dirs_path = list(map(
        lambda base_path: '{}/{}_{}'.format(base_path, request_uid, scene),
        [cur_app.config['TEMP_IMAGES_PATH'], cur_app.config['TEMP_PREDICTIONS_PATH']]
    ))
    # this is the first time creating the folder and so create it without any checks
    for dir_path in dirs_path:
        os.makedirs(dir_path)
    cur_app.logger.debug("[Scene {}] Created directories: {}".format(scene, dirs_path))
    return tuple(dirs_path)


def is_supported_video_type(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in SUPPORTED_VIDEO_EXTENSIONS


def is_supported_image_type(image_type):
    return image_type.lower() in SUPPORTED_VIDEO_EXTENSIONS


def get_validated_arg(cur_app, request_form, request_param, data_type, default_value=None):
    return_value = default_value
    if request_param in request_form:
        try:
            return_value = data_type(request_form[request_param])
        except ValueError as verr:
            cur_app.logger.error(verr)
            return_value = default_value
    return return_value


def save_file(cur_app, file, base_path):
    file_name = secure_filename(file.filename)
    file_path = os.path.join(base_path, file_name)
    file.save(file_path)
    cur_app.logger.debug('Saved file at {}'.format(file_path))
    return file_path


def get_print_string(json_object):
    return json.dumps(json_object, indent=2)
