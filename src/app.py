import argparse
import json
import os
from logging.config import dictConfig

from flask import Flask

from highlights import highlights
from utils import reset_generated_dirs


def create_app():
    # load log config
    with open(args.log_config_file_path) as log_config_file:
        log_config = json.load(log_config_file)
    # set application log config
    dictConfig(log_config)

    # create flask app and set the configs
    app = Flask(__name__)
    app.config.from_mapping(
        TEMP_VIDEOS_PATH=args.temp_videos_path,
        TEMP_IMAGES_PATH=args.temp_images_path,
        TEMP_PREDICTIONS_PATH=args.temp_predictions_path,
        TECHNICAL_WEIGHTS_FILE_PATH=args.technical_weights_file_path,
        AESTHETIC_WEIGHTS_FILE_PATH=args.aesthetic_weights_file_path,
        OUTPUT_IMAGES_PATH=args.output_images_path
    )
    reset_generated_dirs(app)

    @app.route('/')
    def hello_world():
        return 'Hello, World!'

    # register modules
    app.register_blueprint(highlights)

    return app


if __name__ == '__main__':
    # ref: https://docs.python.org/3/howto/argparse.html
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-config-file-path',
                        default='./resources/log-dev.json',
                        help='specify the path to logging configuration')
    parser.add_argument('--temp-videos-path',
                        default='./temp/videos',
                        help='specify the path to store the downloaded videos')
    parser.add_argument('--temp-images-path',
                        default='./temp/images',
                        help='specify the path to store the extracted images')
    parser.add_argument('--temp-predictions-path',
                        default='./temp/predictions',
                        help='specify the path to store the generated predictions')
    parser.add_argument('--technical-weights-file-path',
                        default='./resources/weights/weights_mobilenet_technical_0.11.hdf5',
                        help='specify the file path of technical weights')
    parser.add_argument('--aesthetic-weights-file-path',
                        default='./resources/weights/weights_mobilenet_aesthetic_0.07.hdf5',
                        help='specify the file path of aesthetic weights')
    parser.add_argument('--output-images-path',
                        default='./output/images',
                        help='specify the path to store the output images')

    args = parser.parse_args()

    # get flask app
    cur_app = create_app()
    # run in the required environment
    env_name = os.environ.get("FLASK_ENV", default='development')
    if env_name == 'development':
        debug_on = True
    else:
        debug_on = False
    cur_app.run(debug=debug_on)
