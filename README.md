## Highlight Generator

#### Project Structure

* This project is designed to run in conda environments instead of pip virtual environments
and so a project structure different from a typical flask application.
* Values required to start the application are sent as command line arguments to bash script instead of a config file.

#### Requirements

* Conda package management system 
([>=v4.8.2](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html))
* Shell should be properly configured to use `conda activate` 
(ref: [conda-init](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html#using-with-fish-shell))
* Bash ([>=v4](https://www.tldp.org/LDP/abs/html/bashver4.html))

#### Getting Started

###### Manually start the server (how *start.sh* works)

* Create or update conda environment using *environment.yml* file and activate it 
(ref: [managing-envs](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)). 
* set the environment variables

  ```shell
  export FLASK_APP=highlight-generator
  export FLASK_ENV=[MODE]
  export FLASK_DEBUG=[IS_DEBUG]
  ```

  | Key | Value |
  | --- | --- |
  | `MODE` | server environment mode, accepted values are [`production`, `development`] |
  | `IS_DEBUG` | starts the server in debug mode, accepted values are [`1`, `0`] |

* Application entry-point is `src/app.py`

  ```shell
  python app.py [--log-config-path LOG_CONFIG_PATH] [--temp-path TEMP_PATH] [--output-path OUTPUT_PATH]
  ```

  Optional Arguments:
  
  | Argument | Default Value | Description |
  | --- | --- | --- |
  | `--log-config-path` | `./resources/log.json` | logging configuration file |
  | `--temp-path` | `./temp` | folder for temporary use |
  | `--output-path` | `./output` | folder for output |

###### Start the server with script

* Make a copy of *start.sh*, say *start-local.sh*.

  ```shell
  cp start.sh start-local.sh
  ```

* Run the start script

  ```shell
  start-local.sh [-h] [--debug] [*] [--mode MODE] [--create-conda-env CONDA_ENV_NAME] [--update-conda-env [CONDA_ENV_NAME]] [--use-conda-env CONDA_ENV_NAME]
  ```

  Optional Arguments:
  
  | Argument | Description |
  | --- | --- |
  | `--mode MODE` | server environment mode, accepted values are [`production`, `development`] |
  | `--debug` | starts the server in debug mode if the mode is development |
  | `--update-conda-env CONDA_ENV_NAME` | updates existing conda environment CONDA_ENV_NAME, uses the activated environment when CONDA_ENV_NAME is not provided |
  | `--create-conda-env CONDA_ENV_NAME` | creates a new conda environment CONDA_ENV_NAME when both create and update are sent, then tries to creates arguments to app.py as shown below |
  | `--use-conda-env CONDA_ENV_NAME` | starts the app by activating conda environment CONDA_ENV_NAME, this option is ignored if either of --update-conda-env or --create-conda-env are also available |
  | * | any other arguments required by *src/app.py* as shown in the above section |

Troubleshooting: 

* *`--update-conda-env` option will remain in solving environment*: This happens when the python version is not `3.7.6`. 
So, use `--create-conda-env` option for the first run if conda environment with `python=3.7.6` is not available. 
For all subsequent runs use `--use-conda-env` option if required.
* *Start script throws error `declare: -a: invalid option`*: Start script requires bash 4+. 
Replace the first line in the above script with the location of bash. 
For MacOS, it is `/usr/local/bin/bash` installed with [HomeBrew](https://brew.sh/).

###### Start the server with IDEs

* Create configuration for `src/app.py` in IDE.
* Add [`--log-config-path`, `--temp-path`, `--output-path`] to arguments and assign values to them.
* Add [`FLASK_ENV`, `FLASK_DEBUG`] to environment variables and assign values to them.
* See *start.sh*, *src/app.py*, or the above sections for sample values.
* Use the IDE buttons to run/debug the application.

#### URLs

###### GET /

Home page of the application

###### GET /highlights

Home page of highlights

###### GET /highlights/video-types

Response:

```json
{
  "videoTypes": [
    "mov",
    "mp4"
  ]
}
```

###### GET /highlights/image-types

Response:

```json
{
  "imageTypes": [
    "jpg"
  ]
}
```

###### POST /highlights/generate

Form input: File with name `video` of type as mentioned in *GET /highlights/video-types*

Optional form inputs:

| Input | Default Value | Description |
| --- | --- | --- |
| `clip_time` | 1 minute | time in minutes of each clip |
| `images_per_clip` | 1 | number of images per clip |
| `image_extension` | *jpg* | type of image among *GET /highlights/image-types* the client wants to see |

The API extracts highlights using [NIMA](https://github.com/idealo/image-quality-assessment) 
by mimicking a human eye view and scanning the input video to get best technical images from each clip.
It then generates an aestical score to all those top images and returns the information as an array of predictions.

Prediction:

```json
{
  "imageUrl": "/highlights/images/**/*.jpg",
  "meanScorePrediction": 5.678,
  "timestamp": 7462
}
```

#### Image Evaluation

Image evaluation based on aesthetic model is adapted from 
[idealo/image-quality-assessment](https://github.com/idealo/image-quality-assessment) and 
the author of this repository takes no credit for that part.
