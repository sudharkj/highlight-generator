#!/bin/bash
set -e

# set the python env
conda activate base

# install the required packages
pip install -r requirements/pip-requirements.txt
conda install --file requirements/conda-requirements.txt

# define option values
SRC_PATH=.
RESOURCES_PATH=${SRC_PATH}/resources
LOG_CONFIG_FILE_PATH=${RESOURCES_PATH}/log-dev.json
TEMP_PATH=${SRC_PATH}/temp
TEMP_VIDEOS_PATH=${TEMP_PATH}/videos
TEMP_IMAGES_PATH=${TEMP_PATH}/images
TEMP_PREDICTIONS_PATH=${TEMP_PATH}/predictions
WEIGHTS_PATH=${RESOURCES_PATH}/weights
TECHNICAL_WEIGHTS_FILE_PATH=${WEIGHTS_PATH}/weights_mobilenet_technical_0.11.hdf5
AESTHETIC_WEIGHTS_FILE_PATH=${WEIGHTS_PATH}/weights_mobilenet_aesthetic_0.07.hdf5
OUTPUT_PATH=${SRC_PATH}/output
OUTPUT_IMAGES_PATH=${OUTPUT_PATH}/images

# run the server
export FLASK_APP=highlight-generator
export FLASK_ENV=development
export FLASK_DEBUG=1
python src/app.py \
--log-config-file-path ${LOG_CONFIG_FILE_PATH} \
--temp-videos-path ${TEMP_VIDEOS_PATH} \
--temp-images-path ${TEMP_IMAGES_PATH} \
--temp-predictions-path ${TEMP_PREDICTIONS_PATH} \
--technical-weights-file-path ${TECHNICAL_WEIGHTS_FILE_PATH} \
--aesthetic-weights-file-path ${AESTHETIC_WEIGHTS_FILE_PATH} \
--output-images-path ${OUTPUT_IMAGES_PATH}
