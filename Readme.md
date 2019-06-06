

## ***Behavrioal Cloning Project***



The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./imgs/normal.jpg "Normal Image"
[image2]: ./imgs/normal_flip.jpg "Flipped Image"
[image3]: ./imgs/recovery.jpg "Recovery Image"
[image4]: ./imgs/recovery1.jpg "Recovery Image"
[image5]: ./imgs/recovery2.jpg "Recovery Image"
[image6]: ./imgs/recovery3.jpg "Recovery Image"


---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submssion includes functional code
Using the Udacity provided beta simulator and my drive.py file, the car can be driven autonomously around the track by executing 
`
python drive.py model.json
`
However the code has been modified to reduce the input size of a image by 40%, since the model was trained on 64x128 px images.The speed has been reduced to 10 mph.  

#### 3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model arcthiecture has been employed

My model consists of a convolution neural network with range of filters from 8x8 to 1x1 sizes and depths between 16 and 128 

The model includes RELU layers to introduce nonlinearity , and the data is normalized in the model using a Keras lambda layer 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning
Before usage of adam optimizer I tried different learning rate parameters, and concluded that the default adam optimizer parameters are acceptable.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. It's size has been reduced to 40% because of my hardware limits.  

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to record serveral runs on the track and additional recovery driving data.

My first step was to use a convolution neural network model similar to the Nvidia's. I thought this model might be appropriate because my testing between the comma.ai and nvidia gave me better results with the Nvidia's. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 

I added additonal convolution layer, and dropouts after flattening and last fully conected one.


The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. to improve the driving behavior in these cases, I recorded additional recovery data on a difficult parts of the track.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes:
* 5 convolutional layers (sizes of: 16, 34, 48, 64, 64) with maxpooling afer each one
* 3 fully conected layers (sizes of: 128, 64, 32) with dropout after flattening, and after the last one



#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded 4-5 laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover in difficult areas and sharp turns on the track for 2-3 laps. These images show what a recovery looks like:

![alt text][image3]
![alt text][image4]
![alt text][image5]![alt text][image6]

To augment the data sat, I also flipped images and angles thinking that this would help with better training.  For example, here is an image that has then been flipped:

![alt text][image2]



After the collection process, I had ~23k number of data points. I then preprocessed this data by reducing the size of the picture by 40% (64x128px) because of my hardware limitations, and generated additional data with by flipping the data points witch ammounted to final ~46k data points.


I finally randomly shuffled the data set and put 30% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 20 and 1000 batches to reach convergence. I used an adam optimizer so that manually training the learning rate wasn't necessary.


```python

```
