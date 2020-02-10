#!/bin/bash
set -e

create_or_update_conda_env()
{
  # update conda only for create or update environments
  if [ "${IS_NEW}" = "0" ] && [ "${IS_NEW}" != "1" ]
  then
    conda update -n base -c defaults conda
  fi

  # create a new env after deleting it (if exists and is required)
  if [ "${IS_NEW}" = "1" ]
  then
    conda env remove -n "${ENV_NAME}"
    conda env create -n "${ENV_NAME}" -f environment.yml
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
    conda env update -f environment.yml
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
  echo "usage: start[-local].sh [-h] [--debug] [*] [--mode MODE]"
  echo "                        [--create-conda-env CONDA_ENV_NAME]"
  echo "                        [--update-conda-env [CONDA_ENV_NAME]]"
  echo "                        [--use-conda-env CONDA_ENV_NAME]"
  echo ""
  echo "optional arguments:"
  echo "  -h, --help                show this help message and exit"
  echo "  --mode MODE               server environment mode, accepted values are [production, development]"
  echo "  --debug                   starts the server in debug mode if the mode is development"
  echo "  --update-conda-env CONDA_ENV_NAME"
  echo "                            updates existing conda environment CONDA_ENV_NAME,"
  echo "                            uses the activated environment when CONDA_ENV_NAME is not provided"
  echo "  --create-conda-env CONDA_ENV_NAME"
  echo "                            creates a new conda environment CONDA_ENV_NAME"
  echo "                            when both create and update are sent, then tries to creates"
  echo "  --use-conda-env CONDA_ENV_NAME"
  echo "                            starts the app by activating conda environment CONDA_ENV_NAME,"
  echo "                            this option is ignored if either of --update-conda-env or --create-conda-env are also available"
  echo "  *                         arguments to app.py as shown below"
  echo ""
}

# parse arguments
IS_NEW=-1
IS_DEBUG=0
declare -A ARG_MAP
while [ "$1" != "" ]; do
  case $1 in
    --mode )                    MODE=$2
                                shift
                                shift
                                ;;
    --debug )                   IS_DEBUG=1
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
    --use-conda-env )           if [ "${ENV_NAME}" = "" ]
                                then
                                  ENV_NAME=$2
                                fi
                                shift
                                shift
                                ;;
    -h | --help )               usage
                                HELP=1
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
fi

# set environment variables
validate_mode
export FLASK_APP=highlight-generator
export FLASK_ENV=${MODE}
export FLASK_DEBUG=${IS_DEBUG}

# generate args
if [ "${HELP}" = "1" ]
then
  ARGS="-h"
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
