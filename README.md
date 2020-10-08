# [Traffic signs detection and classification in real time](https://kashish121.github.io/Traffic-Signal-Detection-System/)

## This project is a traffic sign :no_mobile_phones: detection and classification system using OpenCV:round_pushpin:

This project uses the technology *Convolution Neural Network(CNN)*. Because of its high recognition rate and fast execution, CNN is highly preferred in areas where it is required to recognize and classify real world objects.<br>

## Pre-requisites :rotating_light:

- **Python** `>= v3.4.0`
    - Install Python from [here](https://www.python.org/).
- **OpenCV** `= v3.4.x`
    - Install OpenCV from [here](https://opencv.org/releases/).
- **pip** `>= v19.0.1`
    - Install pip from [here](https://pip.pypa.io/en/stable/installing/).
- **Imutils, matplotlib, skimage**
    - Install module using pip.

## How to run? :rocket:

- **Open terminal** inside project folder location.
Powershell / command prompt,
1. Install OpenCV,

```ps
pip3 install opencv-contrib-python==3.4.11.43
```
> :warning: *Make sure your pip version in in association with python3 and not python2.*<br> 
> :warning: *Make sure you install OpenCV v3.x and not OpenCV v4.x to avoid unsupported formats and errors.*

2. Install matplotlib,

```ps
pip3 install matplotlib
```

3. Install skimage,

```ps
pip3 install -U scikit-image
```

4. Install imutils,

```ps
pip3 install imutils
```

5. Run `main.py`

```ps
python main.py
```
*Voila! :clap: You can now check out the output.txt and the output video generated.*

## Directory structure :open_file_folder:
- `main.py` : *The main program to execute which produces output.*
- `classification.py` : *SVM Model to classify traffic signs.*
- `common.py` : *Functions for defining SVM Model.*
- `data_svm.dat` : *Saved SVM model after training.*
- `dataset` : *contains images for training SVM models.*
- `dataset/1-12` : *contains cropped images of traffic signs, each folder containing different class of traffic sign.*
- `dataset/0` : *contains non-traffic-sign cropped images which can be recognized as traffic signs in the detection phase* 
- `images` : *contains original samples of images of environment(grass, path, road, ..etc) and the supported traffic signs in detection phase.*

This project is divided in two phases, namely **detection phase** and **classification phase**. 

### :one: Detection phase

The detection phase uses Image Processing techniques that creates contours on each video frame and finds all ellipses or circles among those contours. They are marked as candidates for traffic signs.

**Detection strategy:**
- Increase the contrast and dynamic range of the video frame
- Remove unnecessary colors like green with HSV Color range
- Use Laplacian of Gaussian to display border of objects
- Make contours by Binarization.
- Detect ellipse-like and circle-like contours

### :two: Classification phase

In the classification phase, a list of images will be created by cropping from the original frame based on candidates' coordinate. A pre-trained SVM model classifies these images to find out which type of traffic sign they are.

## System architecture: How does this work? :wrench:

For better understanding, flow of the project is as follows:

1. Determine the data set Understanding
2. Load the data
3. Analyse the data
4. Data pre-processing
5. Define the Convolution network
6. Model the data
7. Compile the model
8. Train the model
9. Model evaluation of the test data set
10. Generating the classification result

If a traffic sign is detected, it will be tracked until it disappears or there is another bigger sign in the frame. The tracking method is Dense Optical Flow.
> The `main.py` trains the dataset every time it executes. But, the `data_smv.dat` already contains a trained dataset if we want to skip the traning of SMV model every time we execute the detection system.

## Thanks for visiting! :cherry_blossom: 

[![](https://sourcerer.io/fame/Kashish121/Kashish121/Traffic-Signal-Detection-System/images/0)](https://sourcerer.io/fame/Kashish121/Kashish121/Traffic-Signal-Detection-System/links/0)
[![](https://sourcerer.io/fame/Kashish121/Kashish121/Traffic-Signal-Detection-System/images/1)](https://sourcerer.io/fame/Kashish121/Kashish121/Traffic-Signal-Detection-System/links/1)
[![](https://sourcerer.io/fame/Kashish121/Kashish121/Traffic-Signal-Detection-System/images/2)](https://sourcerer.io/fame/Kashish121/Kashish121/Traffic-Signal-Detection-System/links/2)
[![](https://sourcerer.io/fame/Kashish121/Kashish121/Traffic-Signal-Detection-System/images/3)](https://sourcerer.io/fame/Kashish121/Kashish121/Traffic-Signal-Detection-System/links/3)
[![](https://sourcerer.io/fame/Kashish121/Kashish121/Traffic-Signal-Detection-System/images/4)](https://sourcerer.io/fame/Kashish121/Kashish121/Traffic-Signal-Detection-System/links/4)
[![](https://sourcerer.io/fame/Kashish121/Kashish121/Traffic-Signal-Detection-System/images/5)](https://sourcerer.io/fame/Kashish121/Kashish121/Traffic-Signal-Detection-System/links/5)
[![](https://sourcerer.io/fame/Kashish121/Kashish121/Traffic-Signal-Detection-System/images/6)](https://sourcerer.io/fame/Kashish121/Kashish121/Traffic-Signal-Detection-System/links/6)
[![](https://sourcerer.io/fame/Kashish121/Kashish121/Traffic-Signal-Detection-System/images/7)](https://sourcerer.io/fame/Kashish121/Kashish121/Traffic-Signal-Detection-System/links/7)

Feel free to contact me on socials! 
[@Kashish121](https://github.com/Kashish121)

<!-- Using CNN, python based.
Tools and technologies used:
1. Python
2. CNN
3. opencv
4. machine learning
5. image processing -->
