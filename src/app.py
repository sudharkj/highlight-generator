import argparse
import json
import os
from logging.config import dictConfig

from flask import Flask

from generator import utils
from highlights import highlights


def create_app():
    # load log config
    with open(args.log_config_path) as log_config_file:
        log_config = json.load(log_config_file)
    # set application log config
    dictConfig(log_config)

    # create flask app and set the configs
    app = Flask(__name__)
    app.config.from_mapping(
        TEMP_VIDEOS_PATH="{}/videos".format(args.temp_path),
        TEMP_IMAGES_PATH="{}/images".format(args.temp_path),
        TEMP_PREDICTIONS_PATH="{}/predictions".format(args.temp_path),
        TECHNICAL_WEIGHTS_FILE_PATH='./resources/weights/weights_mobilenet_technical_0.11.hdf5',
        AESTHETIC_WEIGHTS_FILE_PATH='./resources/weights/weights_mobilenet_aesthetic_0.07.hdf5',
        OUTPUT_IMAGES_PATH="{}/images".format(args.output_path)
    )
    dirs_to_resets = [
        app.config['TEMP_VIDEOS_PATH'],
        app.config['TEMP_IMAGES_PATH'],
        app.config['TEMP_PREDICTIONS_PATH'],
        app.config['OUTPUT_IMAGES_PATH']
    ]
    utils.reset_generated_dirs(dirs_to_resets)
    app.logger.debug("Deleted directories: {}".format(dirs_to_resets))

    @app.route('/')
    def hello_world():
        return 'Hello, World!'

    # register modules
    app.register_blueprint(highlights)

    return app


if __name__ == '__main__':
    # ref: https://docs.python.org/3/howto/argparse.html
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-config-path',
                        default='./resources/log.json',
                        help='logging configuration file, defaults to "./resources/log.json"')
    parser.add_argument('--temp-path',
                        default='./temp',
                        help='folder for temporary use, defaults to "./temp"')
    parser.add_argument('--output-path',
                        default='./output',
                        help='folder for output, defaults to "./output"')

    args = parser.parse_args()

    # get flask app
    cur_app = create_app()
    # run in the required environment
    env_name = os.environ.get("FLASK_ENV", default='development')
    debug_on = bool(os.environ.get("FLASK_DEBUG", default=0))
    cur_app.run(debug=debug_on)
