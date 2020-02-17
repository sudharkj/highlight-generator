import math
import random

import cv2
from flask import current_app
from tqdm import tqdm

from generator.base_mode import BaseMode, get_predictions

# human eye params
RAND_INT_START = 2/3
RAND_INT_END = 1


class HumanEyeMode(BaseMode):
    def __init__(self, state):
        super().__init__(state)

    def get_frame_skip_limit(self):
        end = max(RAND_INT_START, RAND_INT_END)
        end = self.frames_per_second * end * self.clip_time
        end = math.ceil(end)

        start = min(RAND_INT_START, RAND_INT_END)
        start = self.frames_per_second * start
        start = end - math.floor(start)

        random.seed()
        return random.randint(start, end)

    def save_clip_frames(self):
        clip_start_time, _, frames_in_clip = self.get_clip_details()
        video_cap = cv2.VideoCapture(self.video_file_path)
        video_cap.set(cv2.CAP_PROP_POS_MSEC, clip_start_time * 1000)

        current_app.logger.debug("{} Sampling {} frames".format(self.tag, frames_in_clip))
        count = 0
        with tqdm(total=frames_in_clip, desc="{} Sampling".format(self.tag)) as frame_bar:
            for frame_id in range(frames_in_clip):
                success, image = video_cap.read()
                frame_skip_limit = self.get_frame_skip_limit()
                count += 1

                if success:
                    frame_bar.update(1)
                    if count >= frame_skip_limit:
                        count = 0
                        timestamp = int(video_cap.get(cv2.CAP_PROP_POS_MSEC))
                        self.save_frame_for_prediction(timestamp, image)
                else:
                    pending_frames = frames_in_clip - frame_id
                    if pending_frames > 1:
                        current_app.logger.error("{} Unable to read {} frames".format(self.tag, pending_frames-1))
                    frame_bar.update(pending_frames)
        current_app.logger.debug("{} Completed sampling frames".format(self.tag))
        # release video handles and delete it with the containing folder
        video_cap.release()

    def get_predictions(self):
        return get_predictions(
            self.tag, self.images_path,
            current_app.config['TECHNICAL_WEIGHTS_FILE_PATH'],
            self.prediction_limit
        )
