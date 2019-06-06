import pickle
import numpy as np
import math
import os
import matplotlib.image as mpimg
import tensorflow as tf
tf.python.control_flow_ops = tf
from sklearn.utils import shuffle
from keras import callbacks
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, ELU
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l2
from keras.optimizers import Adam
import pandas as pd


#load the csv data file.
data = pd.read_csv('data2/driving_log.csv', delimiter=',',skipinitialspace=True)

#remove unnececary rows
data.drop(['left','right','brake','throttle','speed'],axis=1,inplace=True)

#load the images in an array
images = np.array([mpimg.imread('%s%s' % ('data2/', fname)) for fname in data['center']])

#load the steering data sa labes array
labels = np.array([y for y in data.xs('steering', axis=1)])


#flip imgs and labels
X_flipped = np.array([np.fliplr(i) for i in images])
Y_flipped = np.array([-i for i in labels])
images = np.concatenate([images, X_flipped])
labels = np.concatenate([labels, Y_flipped])

#shuffle the data
X_train, y_train = shuffle(images, labels) 



### Network model
model = Sequential()
#added normalizer 
model.add(Lambda(lambda x: x/127.5 -1., input_shape=(64, 128, 3), output_shape=(64, 128, 3)))

#conv. 1 
model.add(Convolution2D(16, 8, 8))
model.add(MaxPooling2D((1, 2)))
model.add(Activation('relu'))

#conv. 2 
model.add(Convolution2D(32, 5, 5))
model.add(MaxPooling2D((2, 2)))
model.add(Activation('relu'))

#conv. 3 
model.add(Convolution2D(48, 5, 5))
model.add(MaxPooling2D((2, 2)))
model.add(Activation('relu'))

#conv 4 
model.add(Convolution2D(64, 3, 3))
model.add(MaxPooling2D((2, 2)))
model.add(Activation('relu'))

#con 5
model.add(Convolution2D(64, 1, 1))
model.add(MaxPooling2D((1, 2)))
model.add(Activation('relu'))

#flatten & dropout
model.add(Flatten())
model.add(Dropout(0.2))

#fc1
model.add(Dense(128))
model.add(Activation('relu'))

#fc2
model.add(Dense(64))
model.add(Activation('relu'))

#fc3 & dropout
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.5))

#output
model.add(Dense(1))


model.summary()

#Experimenting with learning rate but found the default adam parameters work best
adam=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

#compiling model with adam optimizer and mean squared error metrics
model.compile(adam, 'mse')


#stop training if the validation loss doesn't improve for 5 consecutive epochs.
early_stop = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')

callbacks_list = [early_stop]

#training the model
model.fit(X_train, y_train, batch_size=1000, nb_epoch=20, validation_split=0.3, callbacks=callbacks_list)


#saving model to the disk
model_json = model.to_json()
with open("model.json","w") as json_file:
    json_file.write(model_json)

model.save_weights("model.h5")
print("Saved model to disk")
