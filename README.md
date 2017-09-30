# **Behavioral Cloning** 

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[pp_left]: ./examples/pp_left.png "pp_left"
[pp_right]: ./examples/pp_right.png "pp_right"
[pp_center]: ./examples/pp_center.png "pp_center"
[hist_original]: ./examples/hist_original.png "hist_original"
[hist_augmented]: ./examples/hist_augmented.png "hist_augmented"
[loss]: ./examples/loss.png "loss"
[model]: ./examples/model.png "model"
[nvidia]: ./examples/nvidia.png "nvidia"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* this README.md report summarizing the results
* video.mp4 recording of model driving the car around the track

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```python drive.py```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy Quick Overview

#### 1. An appropriate model architecture has been employed

Modified Nvidia End to End architecture was used. (model.py lines 183-210)

The model consists of following layers
 
* cropping top 50 and bottom 20 pixel rows of the image
* resizing the image to 50% in both dimensions
* normalization layer
* 5 convolution layers with various filter sizes, strides and depths to extract features
* 4 fully connected layers to predict the steering angle 

The model includes ELU activation layers to introduce nonlinearity.


#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on data set from both track 1 and track 2 to ensure that the model was not overfitting. 

Evolution of loss function on both training and validation sets over training was recorded and final number of epochs was chosen to prevent overfitting of the model.

The model was also tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 213).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used following data:

* 2 laps of center lane driving counter-clockwise on track 1
* 1 lap of center lane driving clockwise on track 1
* recovering from the left and right sides of the road on track 1
* 1 lap of center lane driving on track 2
* recovering from the left and right sides of the road on track 2

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy Details

#### 1. Model Architecture

I followed the tips mentioned in the [quide](https://slack-files.com/T2HQV035L-F50B85JSX-7d8737aeeb) by Paul Heraty.

My first step was to use a [Nvidia's End to End convolution neural network](https://arxiv.org/abs/1604.07316). I thought this model might be appropriate, because it was used for the same purpose - to predict steering angles based on center mounted front camera. Here is the visualization of the Nvidia architecture:

![alt text][nvidia]


Several changes were made to tailor the model for my use:

* The expected input is 160x320x3 instead of 3x66x220.
* Three preprocessing layers are introduced:
	* Cropping layer to focus the model on the important part of the image. The layer crop top 50 and bottom 20 pixel rows.
	* Rescaling layer with factor 0.5 to effectively cut down the number of pixels to a fourth.
	* Normalization layer to improve the performance of gradient descent during training.
* The stride in the first convolutional layer had to be changed from 2 to 1, because of the rescaling of the images.
* ELU activation functions were included in between the convolutional and fully connected layers.
* The output layer consists of only one node - the steering angle prediction.


#### 2. Final Model Architecture

The final model architecture (model.py lines 183-210) looks like this:

![alt text][model]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving in counter-clockwise direction and one lap driving clockwise. Here are two  example images of center lane driving and recovery before and after preprocessing in the model (cropped and scaled down):

![alt text][pp_center]
![alt text][pp_left]

I then recorded the vehicle recovering from the left side and right sides of the road of track one back to center so that the vehicle would learn how to get back to the center if it steers to close to the edge of the road. 

I also recorded one lap and some recovery driving on track two in order to help the model to generalize. Here is an example of that before and after preprocessing:

![alt text][pp_right]


At this point my data set had 13720 unique samples. Then I investigated the steering angle histogram in this data set:
![alt text][hist_original]

I noticed that the straight driving is greatly overrepresanted and might make the model more prone to drive in staright line. To combat this I decide to randomly discard half of the straight driving samples (model.py lines 140-143).
The number of samples after this dropped to 11230.


To expand the data set, I also used the images from the left and right mounted camera with steering correction factor of 0.2, resp. -0.2. Suddenly, the number of samples tripled to 33690 and the histogram looked more evenly distributed.

![alt text][hist_augmented]

To feed the data into the training process a python generator was defined with option to also include vertically flipped images and negative angle thinking that this would balance the steering angle distribution out. (model.py lines 32-58)

I used the ```train_test_split``` from sklearn to split the data into test and validation sets with ratio 80-20. During the training the training data were generated with the flipped augmentation, the validation set wasn't.

The resulting data set size:

* Training set: 53904 samples
* Validation set: 6738 samples


I used this training data for training the model. I used an adam optimizer for *mean squared error* loss function so that manually tuning the learning rate wasn't necessary.

I ran the training for 10 epochs with enabled Keras checkpoint callback to save model after every epoch.

This is the plot of the loss function on both training and validation sets over the epochs:

![alt text][loss]

You can see that the loss function keeps on decreasing on both sets, but the score is lower on test set than on validation set from epoch 5. This might suggest overfitting of the model. To prevent this, the final model included with this submission is the model trained for 5 epochs, when the loss fucntion was rougly the same for both data sets.

**Problems encountered**

* In the start, I had pretty good results on both training and validation sets, but the performance in the simulator was quite bad. I tried adding more samples and training for more epochs, but the behavior in the simulator got even worse! After checking the code I found the root cause - my model used opecv function ```imread``` to read the image files for training and validation, but this function returns images in BGR format. On the other hand, the Udacity simulator returns images in RGB. After fixing this issue, the performace drastically improved.
* I had to rewrite the ```drive.py```, because the original version was raising GPU memory allocation and CuDNN errors.
* The desired speed in ```drive.py``` affects the ability of the car to stay on the road. Set to 10 the vehicle drives reliably on the first track, but at higher speeds (20-30) it can get into trouble.

**Future improvements**

* Try more agressive cropping.
* Try the limit for downscaling.
* Add more data augmentation - translation, rotation, brightness shift, etc.
* Would dropout layers improve the performace on the second track?
* Record more data to make the model drive more confidently on the second test track.
