import tensorflow as tf
import keras
from keras import backend as K
import pandas as pd
import numpy as np

from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten

from keras.applications.vgg16 import VGG16
from keras.applications.densenet import DenseNet169
from keras.applications.resnet import ResNet152, ResNet101, ResNet50

# Hyperparameter setting
batch_size = 100
num_classes = 10
epochs = 12

# Data loading
x_train = pd.read_pickle('/content/drive/My Drive/551_A3/train_max_x')
x_test = pd.read_pickle('/content/drive/My Drive/551_A3/test_max_x')
y_train = pd.read_csv('/content/drive/My Drive/551_A3/train_max_y.csv',header=0)['Label'].values

for i in range(x_train.shape[0]):
  x_train[i] = (x_train[i]>200).astype('int32')*255

# We use method from torch to add one dimension [50000,128,128] to [50000,128,128,1]
# Method from keras for dimension expansion has memory overflow issue, so we use method from torch to do this.
import torch
# x_train = pd.read_pickle('/content/drive/My Drive/551_A3/train_max_x')
x_train = torch.Tensor(x_train)
x_train = torch.unsqueeze(x_train, dim=3)/255.

# Since the model we use has input of 3 channels, we deal with this issue using 2 strategies: 
# 1.We repeat this forth layer by 3 times. 2.We add an Conv2D layer as the first layer in our model.
# Here for the uncommented part, we adopt the strategy 1.
x_train = x_train.repeat(1,1,1,3)
x_train = x_train.numpy()
x_train = x_train.astype('float32')

# To evaluate our model's performance and know at which time step it achieves the minimum loss and highest accuracy on the unseen data,
# We use first 49500 data for training and last 500 as validation dataset.
x_valid = x_train[49500:50000]
x_train = x_train[:49500]
print('x_train shape:', x_train.shape)
print('x_valid shape:', x_valid.shape)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_valid = y_train[49500:50000]
y_train = y_train[:49500]
print('y_train shape:', y_train.shape)
print('y_valid shape:', y_valid.shape)


# Model Setup

model = Sequential()
# The trans layer is added if we adopt strategy 2, which we use an additional layer to convert our input's channel from 1 to 3.
# trans = keras.layers.Conv2D(filters=3,kernel_size=5,padding='same')
vgg16 =  VGG16(weights='imagenet', include_top=False, input_shape=(128,128,3))
# res = ResNet152(weights='imagenet', include_top=False, input_shape=(128,128,3))
# des = DenseNet169(weights='imagenet', include_top=False, input_shape=(128,128,3))

# model.add(trans)
model.add(vgg16)
# model.add(des)
# model.add(res)

# Here we have two strategies for the final output generation. 
# 1.We directly use Flatten() and Dense() Method to compress the 1000 units to 10 units
# 2.We add an extra layer of 1000 hidden units and compress it "softly and smoothly"
# Here for the uncommented part, we use strategy 1.

# model.add(Flatten())
# model.add(Dense(10, activation='softmax'))

model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(lr=0.0001),
              metrics=['accuracy'])

# For every trail(epoch), we print out the performane of our model and save it.
for i in range(16):
  print('Trail: ',i)
  history = model.fit(x_train, y_train, batch_size=batch_size, epochs=1, verbose=1)
  score = model.evaluate(x_valid, y_valid, verbose=1)
  print('Test loss:', score[0])
  print('Test accuracy:', score[1])
  model.save('Res152_{}'.format(i))

# Since we supervise the performance and save our model after every trial, we obeserve the loss and accuracy is shaky and goes up and down.
# Therefore, we load the model which gives a good loss and accuracy and go back to that time. We reduce the learning rate from 1e-4 to 1e-6
# in order to help our model goes to the optimal state in which the minimum loss and highest accuracy is achieved.
model = Sequential()
model = keras.models.load_model('/content/drive/My Drive/Vgg16Liu2/Vgg16_2_10**')
K.set_value(model.optimizer.lr, 0.000001)
for i in range(5):
  print('Trail: ',i)
  history = model.fit(x_train, y_train, batch_size=batch_size, epochs=1, verbose=1)
  score = model.evaluate(x_valid, y_valid, verbose=1)
  print('Test loss:', score[0])
  print('Test accuracy:', score[1])
  model.save('VGG16_2_SP_{}'.format(i))



# Now training with reduced learning rate is finished, we use the trained model for the prediction and save the result.
model = Sequential()
model = keras.models.load_model('/content/drive/My Drive/Three_Color/VGG16_1_Super_15')
x_test = pd.read_pickle('/content/drive/My Drive/551_A3/test_max_x')

for i in range(x_test.shape[0]):
  x_test[i] = (x_test[i]>200).astype('int32')*255
x_test = torch.Tensor(x_test)
x_test = torch.unsqueeze(x_test, dim=3)/255.
x_test = x_test.repeat(1,1,1,3)
x_test = x_test.numpy()
x_test = x_test.astype('float32')

pred = model.predict(x_test,verbose=1)
pred = np.argmax(pred,axis=1)
df = pd.DataFrame({'Label': pred})
df.to_csv(r'my_pred.csv')