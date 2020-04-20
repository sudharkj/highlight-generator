import os

import cv2
from flask import current_app
from tqdm import tqdm

from generator.base_mode import BaseMode, get_predictions, append_timestamp

# consts related to histogram calculations
# Use the 0-th and 1-st channels
CHANNELS = [0, 1]
# bins for hue and saturation
H_BINS = 50
S_BINS = 60
HIST_SIZE = [H_BINS, S_BINS]
# hue varies from 0 to 179, saturation from 0 to 255
H_RANGES = [0, 180]
S_RANGES = [0, 256]
RANGES = H_RANGES + S_RANGES  # concat lists
# threshold ratio for scene change detection
THRESHOLD = 0.9


def is_scene_detected(hist, base_hist):
    if base_hist is None:
        return False
    comp_value = cv2.compareHist(base_hist, hist, cv2.HISTCMP_CORREL)
    return comp_value < THRESHOLD


class SceneDetectMode(BaseMode):
    def __init__(self, state):
        super().__init__(state)

    def save_clip_frames(self):
        clip_start_time, _, frames_in_clip = self.get_clip_details()
        video_cap = cv2.VideoCapture(self.video_file_path)
        video_cap.set(cv2.CAP_PROP_POS_MSEC, clip_start_time * 1000)

        with tqdm(total=frames_in_clip, desc="{} Sampling".format(self.tag)) as frame_bar:
            for frame_id in range(frames_in_clip):
                success, image = video_cap.read()

                if success:
                    frame_bar.update(1)
                    timestamp = int(video_cap.get(cv2.CAP_PROP_POS_MSEC))
                    self.save_frame_for_prediction(timestamp, image)
                else:
                    pending_frames = frames_in_clip - frame_id
                    if pending_frames > 1:
                        current_app.logger.error("{} Unable to read {} frames".format(self.tag, pending_frames-1))
                    frame_bar.update(pending_frames)
        # release video handles and delete it with the containing folder
        video_cap.release()

    def get_predictions(self):
        predictions = get_predictions(self.tag, self.images_path, current_app.config['AESTHETIC_WEIGHTS_FILE_PATH'])
        predictions = list(map(lambda pred: append_timestamp(pred), predictions))
        predictions = sorted(predictions, key=lambda k: k['timestamp'])

        to_delete_images = []
        base_hist = None
        start_index = -1
        for index, prediction in tqdm(enumerate(predictions), desc="{} Extracting".format(self.tag)):
            cur_image = cv2.imread('{}/{}.{}'.format(self.images_path, prediction['image_id'], self.image_extension))
            cur_image = cv2.cvtColor(cur_image, cv2.COLOR_BGR2HSV)
            cur_hist = cv2.calcHist([cur_image], CHANNELS, None, HIST_SIZE, RANGES, accumulate=False)
            cv2.normalize(cur_hist, cur_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

            # obtain and compare histograms
            # when there is a scene change detection or it is the last image
            scene_detected = is_scene_detected(cur_hist, base_hist)
            is_terminal = (index + 1) == len(predictions)  # reached end of predictions
            if scene_detected or is_terminal:
                end_index = index if scene_detected else index + 1
                scene_predictions = predictions[start_index:end_index]
                scene_predictions = sorted(
                    scene_predictions, key=lambda k: k['mean_score_prediction'], reverse=True
                )
                aesthetic_image_name = scene_predictions[0]['image_id']
                current_app.logger.debug('aesthetic_image_name: {}'.format(aesthetic_image_name))
                to_delete_images.extend([
                    '{}/{}.{}'.format(self.images_path, scene_pred['image_id'], self.image_extension)
                    for scene_pred in scene_predictions
                    if scene_pred['image_id'] != aesthetic_image_name
                ])
                base_hist = None

            if base_hist is None:
                base_hist = cur_hist
                start_index = index

        for to_delete_image in tqdm(to_delete_images, desc="{} Deleting".format(self.tag)):
            os.remove(to_delete_image)

        return get_predictions(
            self.tag, self.images_path,
            current_app.config['TECHNICAL_WEIGHTS_FILE_PATH'],
            self.prediction_limit
        )
