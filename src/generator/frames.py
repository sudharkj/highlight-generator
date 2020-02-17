import shutil
import traceback

import cv2
from flask import current_app
from tqdm import tqdm

from generator import utils
from generator.human_eye_mode import HumanEyeMode
from generator.scene_detect_mode import SceneDetectMode


def get_clip_frames(state):
    try:
        request_uid = state['request_uid']
        temp_images_path = state['temp_images_path']
        mode = state['mode']
        clip_id = state['clip_id']

        cur_images_path, cur_predictions_path = utils.create_clip_dirs(request_uid, clip_id)

        state['temp_images_path'] = cur_images_path
        if mode == utils.SUPPORTED_MODES[1]:
            method = SceneDetectMode(state)
        else:
            method = HumanEyeMode(state)
        method.save_clip_frames()
        predictions = method.get_predictions()
        method.save_only_best_images(predictions, temp_images_path)
        shutil.rmtree(cur_predictions_path)

    except Exception as ex:
        # any exception occurred in child process puts the parent process in deadlock
        # catching the exception to avoid that
        # https://stackoverflow.com/a/51076484
        current_app.logger.error("Exception occurred in get_clip_frames: {}".format(ex))
        traceback.print_exc()
        return False
    return True


def extract_predicted_frames(predictions, video_file_path, images_path, image_extension):
    tag = "[Final]"
    current_app.logger.debug("{} Saving predicted frames at {}:".format(tag, images_path))
    video_cap = cv2.VideoCapture(video_file_path)
    for prediction in tqdm(predictions, desc="{} Saving".format(tag)):
        video_cap.set(cv2.CAP_PROP_POS_MSEC, prediction['timestamp'])
        success, image = video_cap.read()
        if success:
            cur_image_file_name = '{}/frame_{}.{}'.format(images_path, prediction['timestamp'], image_extension)
            cv2.imwrite(cur_image_file_name, image)
        else:
            current_app.logger.error("{} Unable to read frame at {}ms".format(tag, prediction['timestamp']))
    current_app.logger.debug("{} Completed saving predicted frames at {}.".format(tag, images_path))
    # release video handles and delete it with the containing folder
    video_cap.release()
