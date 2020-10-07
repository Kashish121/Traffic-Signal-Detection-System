# Traffic signs detection and classification in real time

## This project is a traffic signs detection and classification system on videos using OpenCV.

This project uses the technology *Convolution Neural Network(CNN)*. Because of its high recognition rate and fast execution, CNN is highly preferred in areas where it is required to recognize and classify real world objects.<br>
This project is divided in two phases, namely **detection phase** and **classification phase**. 

### Detection phase

The detection phase uses Image Processing techniques that creates contours on each video frame and finds all ellipses or circles among those contours. They are marked as candidates for traffic signs.

**Detection strategy:**
- Increase the contrast and dynamic range of the video frame
- Remove unnecessary colors like green with HSV Color range
- Use Laplacian of Gaussian to display border of objects
- Make contours by Binarization.
- Detect ellipse-like and circle-like contours

### Classification phase

In the classification phase, a list of images will be created by cropping from the original frame based on candidates' coordinate. A pre-trained SVM model classifies these images to find out which type of traffic sign they are.

### System architecture: How does this work?

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

<!-- Using CNN, python based.
Tools and technologies used:
1. Python
2. CNN
3. opencv
4. machine learning
5. image processing -->
