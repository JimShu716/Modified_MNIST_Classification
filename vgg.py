from __future__ import print_function
import keras
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import pandas as pd
import numpy as np
from keras.applications.vgg16 import VGG16
from keras_preprocessing.image import ImageDataGenerator

batch_size = 32
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 128, 128

# the data, split between train and test sets
#(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = pd.read_pickle('/Users/tylerliu/GitHub/Proj3_source/train_max_x')
x_test = pd.read_pickle('/Users/tylerliu/GitHub/Proj3_source/test_max_x')

print('x_test shape:',x_test.shape)
# Binarize
# for i, img in enumerate(x_train): x_train[i] = cv2.threshold(img, 225, 255, cv2.THRESH_BINARY)


x_validation = x_train

y_train = pd.read_csv('./data/train_max_y.csv',header=0)['Label'].values
y_validation = y_train


# if K.image_data_format() == 'channels_first':
#     x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
#     x_validation = x_validation.reshape(x_validation.shape[0], 1, img_rows, img_cols)
#     input_shape = (1, img_rows, img_cols)
# else:
#     x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
#     x_validation = x_validation.reshape(x_validation.shape[0], img_rows, img_cols, 1)
#     input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_validation = x_validation.astype('float32')
x_train /= 255
x_validation /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_validation.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_validation = keras.utils.to_categorical(y_validation, num_classes)

print('y_train shape:', y_train.shape)
print(y_train.shape[0], 'train samples')
print(y_validation.shape[0], 'test samples')

x_train_rgb = np.zeros((x_train.shape[0],img_rows,img_cols,3), dtype='float32')

for i in range(10):
    for j in range(img_cols):
        for k in range(img_rows):
            x_train_rgb[i][j][k][0:3] = x_train[i][j][k]

img_shape = (img_rows, img_cols, 3)

vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=img_shape)

model = Sequential()
# Add the vgg convolutional base model
model.add(vgg_conv)
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
# Compile the model

for layer in model.layers:
    layer.trainable = True
model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.SGD(lr=1e-4),
              metrics=['accuracy'])

history = model.fit(x_train_rgb, y_train,batch_size=batch_size, epochs=12, verbose=1)

score = model.evaluate(x_train_rgb, y_validation, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save('model.h5')