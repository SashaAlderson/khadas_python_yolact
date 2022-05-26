# khadas_python_yolact
## Getting started
build library for yolact
```
chmod u+x build-cv4.sh
./build-cv4.sh
```
You can download [model](https://drive.google.com/file/d/11depzZYc2pchDJYWFhc7yR4uaHUUSltZ/view?usp=sharing) and run yolact.py to inference on camera.
```
python3 yolact.py
```
## Demo
This results is from [c_version](https://github.com/SashaAlderson/khadas_python_yolact/tree/c_version), to get them modify python example or run 
```
git clone -b c_version https://github.com/SashaAlderson/khadas_python_yolact
cd khadas_python_yolact
chmod u+x build-cv4.sh
chmod u+x run.sh
./run.sh
```

![yolact_out](https://user-images.githubusercontent.com/84590713/170486754-21fb2593-5328-4ad4-bd35-eeef432ba1b6.jpg)

## TODO
Maybe someday i'll increase FPS and use 2 models in parallel.
