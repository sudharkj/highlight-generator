import cv2
from flask import current_app
from tqdm import tqdm

import nima
from generator.base_mode import BaseMode, append_timestamp, save_frame, BASE_MODEL

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
THRESHOLD = 0.90


def is_scene_detected(hist, base_hist):
    if base_hist is None:
        return False
    comp_value = cv2.compareHist(base_hist, hist, cv2.HISTCMP_CORREL)
    return comp_value < THRESHOLD


class SceneDetectMode(BaseMode):
    def __init__(self, state):
        super().__init__(state)

    def extract(self, prev_state=None):
        clip_start_time, _, frames_in_clip = self.get_clip_details()
        video_cap = cv2.VideoCapture(self.video_file_path)
        video_cap.set(cv2.CAP_PROP_POS_MSEC, clip_start_time * 1000)

        with tqdm(total=frames_in_clip, desc="{} Extracting".format(self.tag)) as frame_bar:
            for frame_id in range(frames_in_clip):
                success, image = video_cap.read()

                if success:
                    frame_bar.update(1)
                    timestamp = int(video_cap.get(cv2.CAP_PROP_POS_MSEC))
                    save_frame(image, self.extracts_path, timestamp, self.image_extension, True)
                else:
                    pending_frames = frames_in_clip - frame_id
                    current_app.logger.error("{} Unable to read {} frames".format(self.tag, pending_frames - 1))
                    frame_bar.update(pending_frames)
                    break
        # release video handles and delete it with the containing folder
        video_cap.release()
        return prev_state

    def sample(self, prev_state=None):
        # get aesthetic predictions
        predictions = nima.score(
            base_model_name=BASE_MODEL,
            image_source=self.extracts_path,
            swap_pred=self.swap_preds,
            swap_buffer=self.swap_buffer,
            weights_file=current_app.config['AESTHETIC_WEIGHTS_FILE_PATH'],
            is_verbose=self.is_verbose
        )
        # sort them in increasing timestamp for detecting scene change
        predictions = list(map(lambda timed_pred: append_timestamp(timed_pred), predictions))
        predictions = sorted(predictions, key=lambda k: k['timestamp'])

        # load from previous state if available
        base_hist, cur_preds = prev_state if prev_state else (None, [])
        # get best predictions in each scene
        best_preds = []
        for index, aest_pred in enumerate(tqdm(predictions, desc="{} Extracting".format(self.tag))):
            cur_image = cv2.imread('{}/{}.{}'.format(self.extracts_path, aest_pred['image_id'], self.image_extension))
            cur_image = cv2.cvtColor(cur_image, cv2.COLOR_BGR2HSV)
            cur_hist = cv2.calcHist([cur_image], CHANNELS, None, HIST_SIZE, RANGES, accumulate=False)
            cv2.normalize(cur_hist, cur_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

            # obtain and compare histograms
            # when there is a scene change detection or it is the last image in the video
            scene_detected = is_scene_detected(cur_hist, base_hist)
            is_terminal = self.clip_id == self.total_clips and (index + 1) == len(predictions)
            if not scene_detected:
                cur_preds.append(aest_pred)
            if scene_detected or is_terminal:
                # get only the best frame in the scene
                best_pred = max(cur_preds, key=lambda cur_pred: cur_pred['mean_score_prediction'])
                best_preds.append(best_pred)
                cur_preds = [aest_pred]
                base_hist = None

            if base_hist is None:
                base_hist = cur_hist

        # save unprocesed frames to swap path
        desc = "{} Saving unprocessed frames to **{}".format(self.tag, self.swap_path[self.swap_path.rindex("/"):])
        self.save_samples(cur_preds, self.extracts_path, self.swap_path, desc)
        # move processed frames to temps path
        desc = "{} Moving processed frames to **{}".format(self.tag, self.temps_path[self.temps_path.rindex("/"):])
        self.save_samples(best_preds, self.extracts_path, self.temps_path, desc)
        # apply technical predictions on temps path and move the samples to samples path
        self.save_tech_samples(self.temps_path, self.samples_path)
        # (base_hist, cur_preds) contains the termination state of this operation, so return it
        return base_hist, cur_preds
