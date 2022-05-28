# khadas_python_yolact
This repository contains feature that shows only masked objects on the screen.
## Getting started
Build library for yolact
```
chmod u+x build-cv4.sh
./build-cv4.sh
```
Download [model](https://drive.google.com/file/d/11depzZYc2pchDJYWFhc7yR4uaHUUSltZ/view?usp=sharing) and run yolact.py to inference on camera.
```
python3 yolact.py
```
## Demo
To run demo use
```
python3 demo.py
```
![cropped_image](https://user-images.githubusercontent.com/84590713/170768312-c8cbaca9-4b84-41ba-8730-9248e7a1365f.jpg)

## TODO
Maybe someday i'll increase FPS and use 2 models in parallel.
