# Behavioral Cloning Project

Overview
---
This repository contains files for the Behavioral Cloning Project. A detailed writeup of the project is given below.

The goals / steps of this project were to:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[imageNormal]: ./examples/steering_angle_0p2.png "Example of center lane driving (steering angle 0.2)."
[imageCropped]: ./examples/cropped_image.png "Image after cropping out irrelevant portions (e.g., trees and the car hood)."
[imageOverfitting]: ./examples/training_and_validation_loss_without_dropout.png "Without dropout, training loss is lower than validation loss, indicating overfitting."
[imageNotOverfitting]: ./examples/training_and_validation_loss.png "With dropout, both training and validation loss continue to decrease."
[imageReflected]: ./examples/reflected_image.png "An image reflected about the vertical axis (reflected steering angle -0.2)."
[imageRecoveryTurn]: ./examples/recovery_image.png "An image collected during a recovery turn (steering angle -0.53)."
[imageDrivingGIF]: ./examples/video.gif "Performance on the challenge video"
[imageUnsuccessful]: ./examples/unsuccessful_model.PNG "A simpler model with only two convolutional layers and maxpooling could not successfully complete the track."
[imageSuccessful]: ./examples/successful_model.PNG "A convolutional neural network based on NVIDIA's PilotNet successfully completed the track."

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* [model.py](https://github.com/marcbadger/CarND-Behavioral-Cloning-P3/blob/master/model.py) containing the script to create and train the model
* [drive.py](https://github.com/marcbadger/CarND-Behavioral-Cloning-P3/blob/master/drive.py) for driving the car in autonomous mode
* [model.h5](https://github.com/marcbadger/CarND-Behavioral-Cloning-P3/blob/master/model.h5) containing a trained convolution neural network 
* [README.md, this document](https://github.com/marcbadger/CarND-Behavioral-Cloning-P3/blob/master/README.md), a writeup report summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. Final Model Architecture

I started with NVIDIA's PilotNet, as suggested in project resources. PilotNet is a convolution neural network with 5 convolutional layers (each followed by a RELU layer) and 3 fully conected layers (model.py lines 111-121).  The first 3 convolutional layers have 24, 36, and 48 channels, 5x5 filter sizes, and 2x2 stride, which reduces the feature size.  The last 2 convolutional layers have 64 channels, 3x3 filter sizes, and are not strided.  The three fully connected layers in my model have 120, 84, and 1 neurons, respectively.  Note that the ouput is only 1 dimensional (rather than 10 dimensional) because all we are controlling in this project is steering angle. See [Bojarski et al. 2017](https://arxiv.org/pdf/1704.07911.pdf) for further details.

In addition to the layers used by Bojarski et al., I also normalized data in the model using a Keras lambda layer (code line 109), included dropout layers (code lines 118, 120), and cropped the input to limit input to relevant regions (cropping out trees and the car hood, e.g.) of the image (code line 110, see images below).

* Original:
![alt text][imageNormal]

* Cropped:
![alt text][imageCropped]

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 118, 120). 

The model was trained and validated on different data sets to measure whether the model was overfitting (code line 32, 101-102, 142-146). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 139).

#### 4. Appropriate training data

Collecting appropriate training data was the most difficult part of this project. What ultimately worked was two laps of VERY careful center lane driving at 9 mph, along with 10-20 segments recorded while recovering from various angles of approach at the difficult turns. The final dataset I trained on consisted of:
* 5457 frames drawn from two laps at 9 mph. Giving ~32.7k samples after including left, center, and right views and their reflections.
* 4253 frames drawn from two laps at 9 mph after eliminating 55% of angles between -0.03 and 0.03. Giving 25.5k samples.
* 408 frames drawn from recovery maneuvers. Giving 2.4k samples (note that turn-biased sampling was not performed on these recovery segments).

Data were split into training (80%) and validation (20%) sets using sklearn. Data were also randomly shuffled before being input to the generator and each batch returned by the generator was also shuffled.

Because the dataset was too large to load into memory all at once, I used a Python generator to load samples only when they were needed. The generator handles: 1) loading all three views (code lines 53-63), 2) adding steering correction for the right and left views (code lines 64-65), and 3) augmenting the data by reflecting the images and steering angles (code lines 68-80).

For details about how I collected the training data, see the next section. 

### Approach Towards a Successful Solution

#### 1. Initial Architecture

I started with NVIDIA PilotNet architecture, as suggested in project resources. I trained for 5 epochs on two forward loops and one backward loop (using 3 camera views and flipping images and steering commands) for a total of 3*2*3347 ~ 20k samples). The resulting behavior seemed to do fairly well and made it to the bridge.  But then it ran into the side of the bridge. Because the car seemed to not be turning enough, I hypothesized that all the samples of straight driving were biasing the model to go straight when it shouldn't have been. To aleviate this problem, I i) increased the turning parameter for side images form 0.1 to 0.2 and ii) altered my training dataset so that samples with straight steering commands would be undersampled.  To do so, I kept all samples with turning angles less than -0.03 or more than 0.03 and randomly kept only 45% of the samples with steering angles in between. Although this approach did allow the car to get past the bridge, it had the unfortunate consequence of causing the car to pinball slightly around the lane.

When assessing the training and validation error, I noticed that my model was overfitting (see image below).

![alt text][imageOverfitting]

Once I included dropout layers after each fully connected layer, the training error was not lower than the validation error, indicating that the model was no longer overfitting. In general, training for 3-5 epochs seemed to work well.  Even with dropout layers after the first two fully connected layers, after 5 epochs the validation loss usually stopped decreasing even though the training loss continued to decrease further.

![alt text][imageNotOverfitting]

#### 2. Creation of the Training Set
I started using left, center, and right views with reflections of these images about the vertical axis (steering angles were also reflected). Reflections were an easy way to double the training data. An example of a reflected image is below

![alt text][imageReflected]

After getting the architecture and code running, I found that the car then ran off the road in the dirt area after the bridge.  So I collected three more samples of that turn and re-trained. Collecting more samples didn't end up helping much despite collecting several more recovery segments and two more entire loops of data (for a total of 4453*3*2 ~ 26.7k samples, including reflections, in total).

Nothing I tried was working and it keept running off the road.  I finally realized that some of my training data must have been bad - something was sneaking in from my original, somewhat careless, laps around the track.

Next, I borrowed a nice computer mouse and recorded two more laps driving at 9 mph instead of 30 mph.  An example image of center lane driving is below (turning angle 0.2).

![alt text][imageNormal]

 I also completed 10-20 recovery sequences at the locations of difficult turns. An example of one of these recovery turns is below (turning angle -0.53).

![alt text][imageRecoveryTurn]

 I saved the new laps and recovery data in a different folder so I could do an ablation study.  It took a while to get a good technique down where I only started recording data once the recovery turn was already underway.

After retraining for 5 epochs on the new data it worked and the car successfully drove around the track without going off the paved surface!  Successful driving behavior is shown below

![alt text][imageDrivingGIF]

#### 3. Ablation Studies to Evaluate Contributions of the Data and Model Components
Based on a few ablation studies, data augmentation (in this case, reflection about the vertical axis), the recovery data, and eliminating about 55% of the samples with steering angles between -0.03 and 0.03 were all necessary for the car to successfully navigate the track (models trained in these configurations are included as model_noRecoveryData.h6 and model_noTurningBias.h5).

I also tried a smaller model (shown below) with only two convolutional layers and maxpooling, but it definitely gave worse results when trained using the same data. Using the smaller model, the car exhibited more pinballing behavior and it ran off the track before the first turn (model_smallerModel.h5).

Smaller model that didn't work:

![alt text][imageUnsuccessful]

Larger NVIDIA PilotNet model that worked:

![alt text][imageSuccessful]

### Conclusion
I achieved the goal of this project, which was to train a neural network to successfully drive a car around a simple track, but my model failed miserably on the harder track.  My experience with project really highlights the importance of having appropriate training data, and saving data in a way that allows you to trace unwanted behaviors back to specific samples in your training set.  It would be interesting to see if the model has the capacity to generalize to the harder track given enough good training data.
