import re
import shutil

import cv2
from flask import current_app
from tqdm import tqdm

import nima

BASE_MODEL = 'MobileNet'


def save_frame(image, dir_path, timestamp, file_extension, resize=False):
    image_name = "{}/frame_{}.{}".format(dir_path, timestamp, file_extension)
    if resize:
        image = cv2.resize(image, (224, 224))
    cv2.imwrite(image_name, image)


def append_timestamp(prediction):
    timestamp = list(map(lambda s: int(s), re.findall(r'\d+', prediction['image_id'])))[0]
    return {
        'image_id': prediction['image_id'],
        'mean_score_prediction': prediction['mean_score_prediction'],
        'timestamp': timestamp
    }


class BaseMode:
    def __init__(self, state):
        self.total_clips = state['total_clips']
        self.clip_id = state['clip_id']
        self.tag = state['tag']
        self.video_file_path = state['video_file_path']
        self.clip_time = state['clip_time']  # in minutes
        self.prediction_limit = state['images_per_clip']
        self.frames_per_second = state['frames_per_second']
        self.frame_count = state['frame_count']
        self.total_time = state['total_time']
        self.image_extension = state['image_extension']
        # swap directories
        self.swap_path = state['swap_path']
        self.swap_preds = state['swap_preds']
        self.swap_buffer = state['swap_buffer']
        # other directories
        self.temps_path = state['temps_path']
        self.extracts_path = state['extracts_path']
        self.samples_path = state['samples_path']
        # is_verbose
        self.is_verbose = state['is_verbose']

    def get_clip_details(self):
        clip_start_time = int((self.clip_id-1) * self.clip_time * 60) + 1
        clip_end_time = int(self.clip_id * self.clip_time * 60)
        clip_end_time = min(clip_end_time, self.total_time)
        frames_in_clip = int(self.frames_per_second * (clip_end_time - clip_start_time - 1))
        return clip_start_time, clip_end_time, frames_in_clip

    def save_samples(self, predictions, cur_path, new_path, desc="N/A"):
        for prediction in tqdm(predictions, desc=desc):
            cur_location, new_location = tuple(map(
                lambda dir_path: '{}/{}.{}'.format(dir_path, prediction['image_id'], self.image_extension),
                [cur_path, new_path]
            ))
            shutil.move(cur_location, new_location)

    def save_tech_samples(self, cur_path, new_path):
        # get technical predictions
        predictions = nima.score(
            base_model_name=BASE_MODEL,
            image_source=cur_path,
            swap_pred=self.swap_preds,
            swap_buffer=self.swap_buffer,
            weights_file=current_app.config['TECHNICAL_WEIGHTS_FILE_PATH'],
            is_verbose=self.is_verbose
        )
        # get samples
        predictions = sorted(predictions, key=lambda k: k['mean_score_prediction'], reverse=True)
        predictions = predictions[:self.prediction_limit]
        # save samples
        desc = "{} Moving samples to **{}".format(self.tag, new_path[new_path.rindex("/"):])
        self.save_samples(predictions, cur_path, new_path, desc)

    def extract(self, prev_state=None):
        pass

    def sample(self, prev_state=None):
        pass


def predict(state):
    # get aesthetic predictions
    predictions = nima.score(
        base_model_name=BASE_MODEL,
        image_source=state['samples_path'],
        swap_pred=state['swap_preds'],
        swap_buffer=state['swap_buffer'],
        weights_file=current_app.config['AESTHETIC_WEIGHTS_FILE_PATH'],
        is_verbose=state['is_verbose']
    )
    # append timestamp to the predictions
    predictions = list(map(lambda timed_pred: append_timestamp(timed_pred), predictions))
    # sort the predictions by timestamp
    predictions = sorted(predictions, key=lambda k: k['timestamp'])

    # extract original resolution frames
    video_cap = cv2.VideoCapture(state['video_file_path'])
    for prediction in tqdm(predictions, desc="{} Saving".format(state['tag'])):
        video_cap.set(cv2.CAP_PROP_POS_MSEC, prediction['timestamp'])
        success, image = video_cap.read()
        if success:
            save_frame(image, state['predicts_path'], prediction['timestamp'], state['image_extension'])
        else:
            current_app.logger.error("{} Unable to read frame at {}ms".format(state['tag'], prediction['timestamp']))
    # release video handles
    video_cap.release()
    # return predictions
    return predictions
