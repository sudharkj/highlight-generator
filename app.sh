#!/usr/local/bin/bash
set -e

install_cuda()
{
  # export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda/extras/CUPTI/lib64"

  # Add NVIDIA package repositories
  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.1.243-1_amd64.deb
  sudo dpkg -i cuda-repo-ubuntu1804_10.1.243-1_amd64.deb
  sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
  sudo apt-get update
  wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
  sudo apt install ./nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
  sudo apt-get update

  # Install NVIDIA driver
  sudo apt-get install --no-install-recommends nvidia-driver-430
  echo "Check that GPUs are visible using the command: nvidia-smi"

  # Install development and runtime libraries (~4GB)
  sudo apt-get install --no-install-recommends cuda-10-1 libcudnn7=7.6.4.38-1+cuda10.1 libcudnn7-dev=7.6.4.38-1+cuda10.1

  # Install TensorRT. Requires that libcudnn7 is installed above.
  sudo apt-get install -y --no-install-recommends libnvinfer6=6.0.1-1+cuda10.1 libnvinfer-dev=6.0.1-1+cuda10.1 libnvinfer-plugin6=6.0.1-1+cuda10.1
}

create_or_update_conda_env()
{
  # update conda only for create or update environments
  if [ "${IS_NEW}" = "0" ] && [ "${IS_NEW}" != "1" ]
  then
    conda update -n base -c defaults conda
  fi

  # create a new env after deleting it (if exists and is required)
  if [ "${IS_GPU}" = "1" ]
  then
    install_cuda
    ENV_FILE_NAME="environment-gpu.yml"
  else
    ENV_FILE_NAME="environment.yml"
  fi

  # create a new env after deleting it (if exists and is required)
  if [ "${IS_NEW}" = "1" ]
  then
    conda env remove -n "${ENV_NAME}"
    conda env create -n "${ENV_NAME}" -f "${ENV_FILE_NAME}"
  fi

  # activate the env if not activated
  if [ "${ENV_NAME}" != "" ]
  then
    eval "$(conda shell.bash hook)"
    conda activate "${ENV_NAME}"
  fi

  # update the env if required
  if [ "${IS_NEW}" = "0" ]
  then
    conda env update -f "${ENV_FILE_NAME}"
  fi
}

validate_mode()
{
  if [ "${MODE}" = "production" ]
  then
    IS_DEBUG=0
  fi
  if [ "${MODE}" != "production" ]
  then
    MODE=development
  fi
  if [ "${MODE}" = "development" ] && [ "${IS_DEBUG}" != "0" ]
  then
    IS_DEBUG=1
  fi
}

generate_args()
{
  ARGS=""
  for i in "${!ARG_MAP[@]}"
  do
    ARGS="${ARGS} ${i} ${ARG_MAP[i]}"
  done
}

usage()
{
  echo "usage: start[-local].sh [-h] [-g]"
  echo "                        [--create-conda-env CONDA_ENV_NAME]"
  echo "                        [--update-conda-env [CONDA_ENV_NAME]]"
  echo "                        [--use-conda-env CONDA_ENV_NAME]"
  echo "                        [-r] [--mode MODE] [-d] [*]"
  echo ""
  echo "optional arguments:"
  echo "  -h, --help                show this help message and exit"
  echo "conda environment arguments"
  echo "  --update-conda-env CONDA_ENV_NAME"
  echo "                            updates existing conda environment CONDA_ENV_NAME,"
  echo "                            uses the activated environment when CONDA_ENV_NAME is not provided"
  echo "  --create-conda-env CONDA_ENV_NAME"
  echo "                            creates a new conda environment CONDA_ENV_NAME"
  echo "                            when both create and update are sent, then tries to creates"
  echo "  -g, --gpu                 create or update conda environment with tensorflow gpu packages"
  echo "  --use-conda-env CONDA_ENV_NAME"
  echo "                            starts the app by activating conda environment CONDA_ENV_NAME,"
  echo "                            this option is ignored if either of --update-conda-env or --create-conda-env are also available"
  echo "run arguments"
  echo "  -r, --run                 run the application"
  echo "  --mode MODE               server environment mode, accepted values are [production, development]"
  echo "  -d, --debug               starts the server in debug mode if the mode is development"
  echo "  *                         arguments to app.py as shown below"
  echo ""
}

# parse arguments
IS_NEW=-1
IS_DEBUG=0
IS_GPU=0
RUN=0
declare -A ARG_MAP
while [ "$1" != "" ]; do
  case $1 in
    -h | --help )               HELP=1
                                shift
                                ;;
    --create-conda-env )        IS_NEW=1
                                ENV_NAME=$2
                                shift
                                shift
                                ;;
    --update-conda-env )        if [ "$2" != "" ]
                                then
                                  if [ ${IS_NEW} = -1 ]
                                  then
                                    ENV_NAME=$2
                                  fi
                                  shift
                                fi
                                if [ ${IS_NEW} = -1 ]
                                then
                                  IS_NEW=0
                                fi
                                shift
                                ;;
    -g | --gpu )                IS_GPU=1
                                shift
                                ;;
    --use-conda-env )           if [ "${ENV_NAME}" = "" ]
                                then
                                  ENV_NAME=$2
                                fi
                                shift
                                shift
                                ;;
    -r | --run )                RUN=1
                                shift
                                ;;
    --mode )                    MODE=$2
                                shift
                                shift
                                ;;
    -d | --debug )              IS_DEBUG=1
                                shift
                                ;;
    * )                         ARG_MAP["$1"]=$2
                                shift
                                ;;
  esac
done

# create or update conda environment
if { [ "${IS_NEW}" = "0" ] || [ "${IS_NEW}" = "1" ] || [ "${ENV_NAME}" != "" ]; } && [ "${HELP}" != "1" ]
then
  create_or_update_conda_env
  exit 0
fi

# set environment variables
validate_mode
export FLASK_APP=highlight-generator
export FLASK_ENV=${MODE}
export FLASK_DEBUG=${IS_DEBUG}
export MODEL_INPUT_WIDTH=64
export MODEL_INPUT_HEIGHT=64
export MODEL_INPUT_CHANNELS=3
export MODEL_BATCH_SIZE=1

# call help if run is not enabled
if [ "${RUN}" = "0" ]
then
  HELP=1
fi
# generate args
if [ "${HELP}" = "1" ]
then
  ARGS="-h"
  usage
else
  generate_args
fi
# run the server
if [ "${ARGS}" != "" ]
then
  python src/app.py "${ARGS}"
else
  python src/app.py
fi
