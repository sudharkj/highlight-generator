## Highlight Generator

#### Getting Started

**NOTE** 
* A project structure different from traditional flask application is used 
to be able to run in conda environments instead of pip virtual environments.
* Avoiding the config file and sending them in bash script because of less number of values.

###### Start the server with script

Make a copy of `start` to `start.local`.

```shell script
cp start.sh start-local.sh
```

In `start-local.sh`, update the python environment name, directory paths, and logging configuration file path 
(if using a different log file or update `./resources/log-dev.json` to match the needs). 
Make any other required changes like you do not need to install packages after first run 
and so they can be commented. 

So, `start-local.sh` is the entrypoint to the application. Now run the script that will handle everything else for you.

```shell script
./start-local.sh
```

###### Manually start the server

* Install all the required packages.
* Entrypoint is `src/app.py` that takes optional arguement

  | Argument | Default Value | Description |
  | --- | --- | --- |
  | --log-config-file-path | ./resources/log-dev.json | specify the path to logging configuration |
  | --temp-videos-path | ./temp/videos | specify the path to store the downloaded videos |
  | --temp-images-path | ./temp/images | specify the path to store the extracted images |
  | --temp-predictions-path | ./temp/predictions | specify the path to store the generated predictions |
  | --technical-weights-file-path | ./resources/weights/weights_mobilenet_technical_0.11.hdf5 | specify the file path of technical weights |
  | --aesthetic-weights-file-path | ./resources/weights/weights_mobilenet_aesthetic_0.07.hdf5 | specify the file path of aesthetic weights |
  | --output-images-path | ./output/images | specify the path to store the output images |

#### Image Evaluation

Image evaluation based on aesthetic model is adapted from 
[idealo/image-quality-assessment](https://github.com/idealo/image-quality-assessment) and 
the author of this repository takes no credit for that part.
