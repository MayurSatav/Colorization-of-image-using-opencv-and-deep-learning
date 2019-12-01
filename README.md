# Image-Colorizatio-With-GUI

### Overview
This repository contains
- bw_to_color_gui.exe
- Inference_images_dataset folder
- Output-Screenshots folder

### Requirements
- Python 3.7.4
- Tkinter
- PIL
- Numpy
- Open-Cv

### Excecution Steps
#### Step 1 - Download models

colorization_deploy_v2.prototxt [here](https://github.com/richzhang/colorization/blob/master/models/colorization_deploy_v2.prototxt)

colorization_release_v2.caffemodel [here](https://github.com/richzhang/colorization/blob/master/models/fetch_release_models.sh)

pts_in_hull.npy [here](https://github.com/richzhang/colorization/blob/master/resources/pts_in_hull.npy)

#### Step 2 - Create model folder and Add all these models

#### Step 3 - Run ```bw_to_color_gui.py```

### Screenshots
![bw_to_color_gui](https://github.com/MayurSatav/Colorization-of-image-using-opencv-and-deep-learning/blob/master/Output-Screenshots/Screenshot.png)

#### References
[Richard Zhang](https://github.com/richzhang/colorization)

#### For more details on the image colorization refer to the official publication of Zhang
http://richzhang.github.io/colorization/

