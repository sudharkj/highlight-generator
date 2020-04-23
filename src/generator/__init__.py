from generator import utils


def get_predictions(state):
    import math
    import os
    import shutil

    import cv2
    from flask import current_app
    from tqdm import tqdm

    video_cap = cv2.VideoCapture(state['video_file_path'])
    frames_per_second = video_cap.get(cv2.CAP_PROP_FPS)
    frame_count = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
    total_time = frame_count / frames_per_second
    total_clips = math.ceil(total_time / (state['clip_time'] * 60))
    video_cap.release()

    # update the state object
    state.update({
        'frames_per_second': frames_per_second,
        'frame_count': frame_count,
        'total_time': total_time,
        'total_clips': total_clips,
        'tag': "[{}]".format(state['request_uid'])
    })

    # set additional directories
    request_dirs = ["temps", "extracts", "samples", "swap", "swap_preds", "swap_buffer"]
    request_dirs = list(map(lambda add_path: '{}/{}'.format(state['temp_images_path'], add_path), request_dirs))
    state['temps_path'], state['extracts_path'] = request_dirs[:2]
    state['samples_path'], state['swap_path'] = request_dirs[2:4]
    state['swap_preds'], state['swap_buffer'] = request_dirs[4:]
    # reset samples and swap directories
    utils.create_dirs(request_dirs[2:], current_app.logger, state['tag'])

    # manual task assignment in python is making TensorFlow to run on cpu even when gpu is available
    # so implemented sequential requests and removed task scheduling on threads that uses ProcessPoolExecutioner
    extract_state = None
    sample_state = None
    for clip_id in range(1, total_clips+1):
        # generate clip state
        clip_state = state
        clip_state.update({
            'clip_id': clip_id,
            'tag': "[{}:{}]".format(state['request_uid'], clip_id)
        })

        # reset temp and extracts directories
        utils.create_dirs(request_dirs[:2], current_app.logger, clip_state['tag'])
        # move swap images from previous iteration to extract for current iteration
        for file_name in tqdm(
                os.listdir(state['swap_path']),
                desc="{} Loading unprocessed frames".format(clip_state['tag'])
        ):
            cur_location, new_location = tuple(map(
                lambda dir_path: '{}/{}'.format(dir_path, file_name),
                [state['swap_path'], state['extracts_path']])
            )
            shutil.move(cur_location, new_location)
        # reset swap directories
        utils.create_dirs(request_dirs[3:], current_app.logger, clip_state['tag'])

        # process the clip
        mode = state['mode']
        if mode == utils.SUPPORTED_MODES[1]:
            from generator.scene_detect_mode import SceneDetectMode
            mode = SceneDetectMode(clip_state)
        else:
            from generator.human_eye_mode import HumanEyeMode
            mode = HumanEyeMode(clip_state)
        extract_state = mode.extract(extract_state)
        sample_state = mode.sample(sample_state)

    # get final predictions
    from generator.base_mode import predict
    predictions = predict(state)
    # delete the directories created in this request
    for request_dir in tqdm(request_dirs, desc="{} Deleting temp folders".format(state['tag'])):
        shutil.rmtree(request_dir)
    # return final predictions
    return predictions
