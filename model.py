### Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import time
import cv2
import math

from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers import Cropping2D
from keras.layers import Lambda
from keras.layers.advanced_activations import ELU
from keras.callbacks import ModelCheckpoint

# Supress pandas warning on chained assignments
pd.options.mode.chained_assignment = None  # default='warn'


### Tunable parameters
steering_correction = 0.2 # Steering correction applied to left and right images
prob_to_exclude = 0.5 # Probablitiy of excluding low steering angle sample
sanity_check = False # Whether to train only on sanity data set
epochs = 5 # number of epochs for training the model


### Generator of training data
def data_generator(images, angles, batch_size=128, flip=True):
    """
    Inputs are list of image filepaths and list of steering angles.
    Optionally, the batch size and flag, whether flipped images shall be added.
    Outputs training data.
    """
    X,y = ([],[]) # Initialize the output tuple
    # Iterate through image paths
    while True:       
        for i in range(len(angles)):
            # Read image and convert it from BRG to RGB
            img = cv2.imread(images[i]) 
            b,g,r = cv2.split(img)
            img = cv2.merge([r,g,b])
            # Read the asociated angle
            angle = angles[i]
            # Append to output tuple
            X.append(img)
            y.append(angle)
            # If flip flag was used, also append flipped image with negative angle
            if flip:
                X.append(cv2.flip(np.copy(img),1))
                y.append(-angle)
            # If the batch is full, yield it and reset the output tuple for next batch 
            if len(X) == batch_size:
                yield (np.array(X), np.array(y))
                X, y = ([],[])


### Read in csv files with training data
track1_normal = pd.read_csv('training_data\\track1_normal\\driving_log.csv', 
                 header=None, 
                 names = ['Center image',
                          'Left image',
                          'Right image',
                          'Steering angle',
                          'Throttle',
                          'Brake',
                          'Speed',
                          ])
track1_normal2 = pd.read_csv('training_data\\track1_normal2\\driving_log.csv', 
                 header=None, 
                 names = ['Center image',
                          'Left image',
                          'Right image',
                          'Steering angle',
                          'Throttle',
                          'Brake',
                          'Speed',
                          ])
track1_opp = pd.read_csv('training_data\\track1_opp\\driving_log.csv', 
                 header=None, 
                 names = ['Center image',
                          'Left image',
                          'Right image',
                          'Steering angle',
                          'Throttle',
                          'Brake',
                          'Speed',
                          ])  
track1_recovery = pd.read_csv('training_data\\track1_recovery\\driving_log.csv', 
                 header=None, 
                 names = ['Center image',
                          'Left image',
                          'Right image',
                          'Steering angle',
                          'Throttle',
                          'Brake',
                          'Speed',
                          ])
    
track2_normal = pd.read_csv('training_data\\track2_normal\\driving_log.csv', 
                 header=None, 
                 names = ['Center image',
                          'Left image',
                          'Right image',
                          'Steering angle',
                          'Throttle',
                          'Brake',
                          'Speed',
                          ])
track2_recovery = pd.read_csv('training_data\\track2_recovery\\driving_log.csv', 
                 header=None, 
                 names = ['Center image',
                          'Left image',
                          'Right image',
                          'Steering angle',
                          'Throttle',
                          'Brake',
                          'Speed',
                          ])

### Merge all those data into one dataframe
training_data = pd.concat([track1_normal,
                           track1_normal2,
                           track1_opp,
                           track1_recovery,
                           track2_normal,
                           track2_recovery,
                           ],ignore_index=True)


## This piece of code randomly drops samples with small steering angles
## to compensate for the overrepresantation of those samples. The probability
## of dropping is defined by tunable parameter: prob_to_exclude
prob = ((abs(training_data['Steering angle'])<0.05)*prob_to_exclude)
random = np.random.random(prob.shape[0])
mask = prob<random
training_data = training_data[mask]



# Extract the paths and angles for center images
center_images = training_data[['Center image', 'Steering angle']]
center_images.columns = ['Path', 'Steering angle']

# Extract the paths and angles for left images and apply the steering angle
# correction
left_images = training_data[['Left image', 'Steering angle']]
left_images.columns = ['Path', 'Steering angle']
left_images['Steering angle'] += steering_correction

# Extract the paths and angles for right images and apply the steering angle
# correction
right_images = training_data[['Right image', 'Steering angle']]
right_images.columns = ['Path', 'Steering angle']
right_images['Steering angle'] -= steering_correction

# Merge center, left and right images into final data set
data_set = pd.concat([center_images,
                      left_images,
                      right_images,
                      ], ignore_index=True)
    
print('Number of training samples: %d' %data_set.shape[0])

# Plot distribution of the samples in the augmented data set
plt.figure()
data_set['Steering angle'].plot.hist(bins = 101)  
plt.show()
    
# Define sanity check data set - 3 images 
sanity_data_set = pd.concat([data_set[abs(data_set['Steering angle'])<0.05][:1],
                             data_set[data_set['Steering angle']>0.7][:1],
                             data_set[data_set['Steering angle']<-0.7][:1],
                             ], ignore_index=True)


### Model architecture inspired by NVidia Neural Network
model = Sequential()
# Preprocessing - cropping, rescaling, normalization
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: tf.image.resize_images(x, (45, 160))))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))

# Convolutional layers to extract features
model.add(Conv2D(24, (5, 5), strides=(1, 1)))
model.add(ELU())
model.add(Conv2D(36, (5, 5), strides=(2, 2)))
model.add(ELU())
model.add(Conv2D(48, (5, 5), strides=(2, 2)))
model.add(ELU())
model.add(Conv2D(64, (3, 3)))
model.add(ELU())
model.add(Conv2D(64, (3, 3)))
model.add(ELU())

# Fully connected layers to get the prediction
model.add(Flatten())
model.add(Dense(100))
model.add(ELU())
model.add(Dense(50))
model.add(ELU())
model.add(Dense(10))
model.add(ELU())
model.add(Dense(1))

# Compile the model with adam optimizer and Mean sqared error as loss function
model.compile('adam', 'mse')

# Print model architecture graph
from keras.utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=False)



### Train only on sanity data set if the sanity_check flag is raised
if sanity_check:
    # Prepare the training data
    X,y = ([],[])
    for i in range(len(sanity_data_set['Steering angle'])):
        img = cv2.imread(sanity_data_set['Path'][i])
        b,g,r = cv2.split(img)
        img = cv2.merge([r,g,b])
        angle = sanity_data_set['Steering angle'][i]
        plt.figure()
        plt.subplot(211)
        plt.imshow(img)
        plt.gca().add_patch(Rectangle((0, 50), 320, 90, color='b', lw=3, fill=False))
        plot_img = img[50:-20, :, :]
        plot_img = cv2.resize(plot_img,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_AREA)
        plt.subplot(212)
        plt.imshow(plot_img)
        plt.show()
        X.append(img)
        y.append(angle)          
    
    X = np.array(X)
    y = np.array(y)

        
    # Train the model for 20 epochs
    t = time.time() # start clock
    history = model.fit(X, y, epochs=20, verbose = 1)
    print("Time to train the model %6d s" %(time.time()-t))
    
    # Print out actual angles and model prediction
    print(['%0.2f' %i for i in y])
    print(['%0.2f' %j for j in [float(i) for i in model.predict(X)]])
    
    # PLot loss over epochs
    plt.figure()
    plt.title('Loss function')
    plt.xlabel('Epoch')
    plt.ylabel('Mean squared error')
    xticks = np.arange(len(history.history['loss']))+1
    xticks = range(min(xticks), math.ceil(max(xticks))+1)
    plt.plot(xticks, history.history['loss'])
    plt.xticks(xticks)
    plt.show()
    
### Otherwise train with the full dataset
else:
    #### Train model
    t = time.time() # start clock
    
    # Randomly split the data into train and test sets
    images_train, images_val, angles_train, angles_val = train_test_split(data_set['Path'].tolist(), 
                                                           data_set['Steering angle'].tolist(), 
                                                           test_size=0.2, 
                                                           random_state=42)
    
    batch_size=64 # batch size chosen with respect to my system resources and model
    # Use the generator to get batch from training and testing data
    # Augment the training data with vertical flipping, let the test set be
    data_train = data_generator(images_train, angles_train, batch_size=batch_size, flip=True)
    data_val = data_generator(images_val, angles_val, batch_size=batch_size, flip=False)
    
    # Define checkpoints for saving the model after each epoch
    checkpoint = ModelCheckpoint('model{epoch:02d}.h5')
    
    # Train the model for x epochs (parameter)
    history = model.fit_generator(data_train, 
                                  validation_data=data_val, 
                                  epochs=epochs,
                                  steps_per_epoch=len(images_train)//batch_size + 1,
                                  validation_steps=len(images_val)//batch_size + 1,
                                  verbose=1, 
                                  callbacks=[checkpoint])
    
   
    model.save('model.h5')  # Save the model
    print("Time to train the model %6d s" %(time.time()-t))
    
    # Plot loss over epochs
    plt.figure()
    plt.title('Loss function')
    plt.xlabel('Epoch')
    plt.ylabel('Mean squared error')
    xticks = np.arange(len(history.history['loss']))+1
    xticks = range(min(xticks), math.ceil(max(xticks))+1)
    plt.plot(xticks, history.history['loss'])
    plt.xticks(xticks)
    plt.show()
