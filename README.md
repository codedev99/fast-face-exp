# Fast Face Expressions (FFE)

Rapid, accurate, face expression inference. Transfer learning, PyTorch, OpenCV

##### **Note** - The previous release of this repo was built with a different objective, but for the same purpose, to perform emotion detection. The latest version however, although using the same API, uses a different DL model.

# 

![](https://github.com/codedev99/fast-face-exp/releases/download/v0.3/detect.jpg)

This repository allows you to run face expression detection locally on your system; either on a library of images, or even a video stream from your webcam, in real time. This repo is a PyTorch implementation of the EmotionNet model described in the paper [EmotionNet: ResNeXt Inspired CNN Architecture for Emotion Analysis on Raspberry Pi](https://ieeexplore.ieee.org/document/9573569).

The FFE system detects the following five facial expressions: Neutral, Happy, Surprised, Sad, Angry.

## Requirements
+ PyTorch
+ OpenCV
+ Numpy

## How to Run
The FFE (fast face expression) system can be run in two different modes:

1. Run inference on a library of images.
2. Real-time inference from your local webcam feed.
3. Video file inference (currently unimplemented).

### 1. Files Inference
This mode allows you to specify the path to a specific folder containing images. The FFE system then runs inference on these images, marking the detected face expression with a bounding box and type of expression. This marked image is saved in locally in a new folder.

1. Firstly, clone the repo to your local machine, using the following command.
```
git clone https://github.com/codedev99/fast-face-exp.git
```
2. To specify the folder containing input images, open `config.json`, and change the path as desired under the headings *`["inference"]["files"]["data_folder"]`*.

3. Navigate to the repo directory and run the following command to start inference:
```
python inference.py -m 1
```
4. After the inference is complete, the resulting marked images are saved in `./data/inference` folder.

### 2. Camfeed Inference
This mode allows you to run the FFE system in real-time mode using your local camera feed. It open a window displaying the real-time video feed, with detected faces marked by bounding boxes and detected face expressions displayed on top of each boxes.

1. Firstly, clone the repo to your local machine, using the following command.
```
git clone https://github.com/codedev99/fast-face-exp.git
```
2. Ensure that a webcam is attached and in working condition.
3. Navigate to the repo directory and run the following command to start inference:
```
python inference.py -m 2
```
4. This will open a new window with the marked video-feed of the type described above. To exit, press the **ESC** key.