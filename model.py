# Marc Badger
# 9.14.17
# Behavioral Cloning Project

import csv
import cv2
import numpy as np
import sklearn

augmentImages = True
useRecoveryData = True
modelSaveName = 'model.h5'
side_image_correction = 0.2

samples = []
with open('data/data2/turning_biased_driving_log.csv') as csvfile:
	reader=csv.reader(csvfile)
	for line in reader:
		samples.append(line)

if useRecoveryData:
	with open('data/recovery_data/driving_log.csv') as csvfile:
		reader=csv.reader(csvfile)
		for line in reader:
			samples.append(line)


#######################

# We first split the data into training (80%) and validation (20%) sets using sklearn
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# Because the dataset is too large to load into memory all at once, use
# a Python generator to load samples only when they are needed.
# Note: the generator handles:
# 1. loading all three views
# 2. adding steering correction for the right and left views
# 3. augmenting the data by reflecting the images and steering angles
# If all three images and reflections are used, the generator will
# return 6 times the samples (batch_size) it was asked for.
# Code outline based on code shown in the project walkthrough.
def generator(samples, batch_size=8):
	num_samples = len(samples)
	while 1: # Loop forever so the generator never terminates
		sklearn.utils.shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]

			images = []
			angles = []
			for batch_sample in batch_samples:
				for i in range(3):
					source_path = batch_sample[i]

					# Use these if the images are not in the location they were originally saved:
					#filename = source_path.split('\\')[-1] 
					#current_path = 'data/data/IMG/' + filename

					# My images are still where the simulator saved them:
					current_path = source_path
					image = cv2.imread(current_path)
					images.append(image)
				angle = float(batch_sample[3])
				angles.extend([angle, angle+side_image_correction, angle-side_image_correction])

			# if desired, mirror the images
			if augmentImages:
				augmented_images, augmented_angles = [], []
				for image, angle in zip(images, angles):
					augmented_images.append(image)
					augmented_angles.append(angle)
					augmented_images.append(cv2.flip(image,1)) # or np.fliplr(image)
					augmented_angles.append(-1.0*angle) # or -angle

				X_train = np.array(augmented_images) # or augmented_images
				y_train = np.array(augmented_angles) # or augmented_images
			else:
				X_train = np.array(images) # or augmented_images
				y_train = np.array(angles) # or augmented_images

			yield sklearn.utils.shuffle(X_train, y_train)

print("driving log opened!")

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D

# The generator returns a different number of images per batch
# depending on if augmenting images or not, so
# limit the batch size to keep my computer from crashing:
if augmentImages:
	batchSize = 6
	sampMult = 2*3
else:
	batchSize = 12
	sampMult = 1*3

# Tell the generator to use training set for training and validation set for validation
train_generator = generator(train_samples, batch_size=batchSize)
validation_generator = generator(validation_samples, batch_size=batchSize)

# Just the basic NVIDIA model shown in the lesson, but also with:
# 1. Normalization via a Lambda() layer
# 2. Cropping via a Cropping2D() layer to limit input to the relevant area
# 3. Dropout via Dropout() layers after the fully connected layers to prevent overfitting
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(120))
model.add(Dropout(0.5))
model.add(Dense(84))
model.add(Dropout(0.5))
model.add(Dense(1))

# A smaller model with only two comvolutional layers was unable
# to complete the track given the same training data:

# model = Sequential()
# model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
# model.add(Cropping2D(cropping=((70,25),(0,0))))
# model.add(Convolution2D(6,5,5,activation="relu"))
# model.add(MaxPooling2D())
# model.add(Convolution2D(6,5,5,activation="relu"))
# model.add(MaxPooling2D())
# model.add(Flatten())
# model.add(Dense(120))
# model.add(Dense(84))
# model.add(Dense(1))

# Compile, and use an adam optimizer for easy training
model.compile(loss='mse', optimizer='adam')
print("model compiled!")
# Training for 5 epochs seems to be the right amount
model.fit_generator(train_generator, 
	samples_per_epoch=sampMult*len(train_samples),
	validation_data=validation_generator,
	nb_val_samples=len(validation_samples), 
	nb_epoch=5)

model.save(modelSaveName)
print("model saved!")
exit()