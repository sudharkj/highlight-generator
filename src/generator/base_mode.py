import os
import re
import shutil

import cv2
from flask import current_app
from tqdm import tqdm

import nima
from generator import utils

BASE_MODEL = 'MobileNet'


class BaseMode:
    def __init__(self, state):
        self.clip_id = state['clip_id']
        self.tag = "[Clip {}]".format(self.clip_id)
        self.video_file_path = state['video_file_path']
        self.clip_time = state['clip_time']  # in minutes
        self.prediction_limit = state['images_per_clip']
        self.frames_per_second = state['frames_per_second']
        self.frame_count = state['frame_count']
        self.total_time = state['total_time']
        self.images_path = state['temp_images_path']
        self.image_extension = state['image_extension']

    def save_clip_frames(self):
        pass

    def get_clip_details(self):
        clip_start_time = int((self.clip_id-1) * self.clip_time * 60) + 1
        clip_end_time = int(self.clip_id * self.clip_time * 60)
        clip_end_time = min(clip_end_time, self.total_time)
        frames_in_clip = int(self.frames_per_second * (clip_end_time - clip_start_time - 1))
        return clip_start_time, clip_end_time, frames_in_clip

    def save_frame_for_prediction(self, timestamp, image):
        image_file_name = '{}/frame_{}.{}'.format(self.images_path, timestamp, self.image_extension)
        image = cv2.resize(image, (224, 224))
        cv2.imwrite(image_file_name, image)

    def get_predictions(self):
        pass

    def save_only_best_images(self, predictions, new_images_path):
        for prediction in tqdm(predictions, desc="{} Moving".format(self.tag)):
            cur_location, new_location = tuple(map(
                lambda dir_path: '{}/{}.{}'.format(dir_path, prediction['image_id'], self.image_extension),
                [self.images_path, new_images_path])
            )
            shutil.move(cur_location, new_location)
        shutil.rmtree(self.images_path)


def get_predictions(tag, images_path, weights_file_path, prediction_limit=None):
    predictions = []
    if len(os.listdir(images_path)) > 0:
        debug_on = os.environ.get("FLASK_DEBUG", default=0)
        predictions = nima.score(BASE_MODEL, weights_file_path, images_path, is_verbose=debug_on)
        predictions = sorted(predictions, key=lambda k: k['mean_score_prediction'], reverse=True)
        predictions = predictions[:prediction_limit] if prediction_limit is not None else predictions
    return predictions


def append_timestamp(prediction):
    timestamp = list(map(lambda s: int(s), re.findall(r'\d+', prediction['image_id'])))[0]
    return {
        'image_id': prediction['image_id'],
        'mean_score_prediction': prediction['mean_score_prediction'],
        'timestamp': timestamp
    }
