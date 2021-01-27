# Driver Drowsiness Detection System with OpenCV & Keras
### Drowsiness detection is a safety technology that can prevent accidents that are caused by drivers who fell asleep while driving.
* Built a drowsy driver alert system that can be implement in numerous ways.
* Used OpenCV to detect faces and eyes using a haar cascade classifier.
* Used a Convolutional Neural Network model to predict the status.

## Objective
* To build a drowsiness detection system that will detect that a person’s eyes are closed for a few seconds. This system will alert the driver when drowsiness is detected.

## Code and Resources Used
**Python Version:** 3.7
**Packages:** OpenCV(face and eye detection), TensorFlow(keras uses TensorFlow as backend), Keras(to build the classification model), Pygame(to play alarm sound).

## Dataset 
The data comprises around 7000 images of people’s eyes under different lighting conditions. After training the model on dataset, the final weights and model architecture file “models/cnnCat2.h5” is not uploaded as github cannot except file above 25 MB.

## The Model Architecture
The model used is built with Keras using Convolutional Neural Networks (CNN). A convolutional neural network is a special type of deep neural network which performs extremely well for image classification purposes. A CNN basically consists of an input layer, an output layer and a hidden layer which can have multiple numbers of layers. A convolution operation is performed on these layers using a filter that performs 2D matrix multiplication on the layer and filter.

The CNN model architecture consists of the following layers:
* Convolutional layer; 32 nodes, kernel size 3
* Convolutional layer; 32 nodes, kernel size 3
* Convolutional layer; 64 nodes, kernel size 3
* Fully connected layer; 128 nodes
* The final layer is also a fully connected layer with 2 nodes. In all the layers, a Relu activation function is used except the output layer in which the Softmax activation function is used.

## Face & eyes detection using OpenCV
The 'haar cascade files' which works with OpenCV to classify the face,lefteye and righteye is been used.
The models folder contains my model file “cnnCat2.h5” which was trained on convolutional neural networks, so the best weights are already odtained for the model.
An audio clip “alarm.wav” which is played when the person is feeling drowsy.
“Model.py” file contains the program through which the classification model is built by training on dataset. The implementation of convolutional neural network can be seen in this file.

## Methodology
OpenCV is been used for gathering the images from webcam and feed them into a Deep Learning model which will classify whether the person’s eyes are ‘Open’ or ‘Closed’. The steps involved are as follows:

* **Step 1:** Take image as input from a camera and read it using OpenCV.
* **Step 2:** Detect the face in the image and create a Region of Interest (ROI).
* **Step 3:** Detect the eyes from ROI and feed it to the CNN classifier.
* **Step 4:** The CNN classifier will categorize whether eyes are open or closed.
* **Step 5:** Calculate score to check whether the person is drowsy(If the eyes are close for above 20 seconds the alarm will ring).

## Results

* Prediction: Eyes Open, No alarm

![alt text](https://github.com/vikasbhadoria69/drowsiness_detection_MachineLearning_ComputerVision/blob/master/Images/Screenshot%202021-01-27%20030835.png)

* Prediction: Eyes Closed, if the score is greater than 20. Rings an alarm

![alt text](https://github.com/vikasbhadoria69/drowsiness_detection_MachineLearning_ComputerVision/blob/master/Images/Screenshot%202021-01-27%20031004.png)

## Codes: 
"Mpdel.py" contains the code used to build the CNN classifier model.
“Drowsiness detection.py” is the main file of this project. To start the detection procedure, one have to run this file.

