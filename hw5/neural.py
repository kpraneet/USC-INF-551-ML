import numpy as np
from PIL import Image
from numpy import array
import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

# define baseline model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(100, input_dim=num_pixels, init='normal', activation='sigmoid'))
	model.add(Dense(num_classes, init='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
	return model

with open("downgesture_train.list", 'rb') as f:
    my_list = [line.rstrip('\n') for line in f]
image_list = []
y_list = []
for list in my_list:
    if 'down' in list:
        y_list.append(1)
    else:
        y_list.append(0)
    img = Image.open(list)
    arr = array(img)
    image_list.append(arr)

with open("downgesture_test.list", 'rb') as f:
    my_list_1 = [line.rstrip('\n') for line in f]
image_list_test = []
y_list_test = []
for list in my_list_1:
    if 'down' in list:
        y_list_test.append(1)
    else:
        y_list_test.append(0)
    img = Image.open(list)
    arr = array(img)
    image_list_test.append(arr)
# print(y_list_test)
# print(image_list_test[0])

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# print(X_train)
# print(X_train[0])
# flatten 28*28 images to a 784 vector for each image
X_train = np.array(image_list)
y_train = np.array(y_list)
X_test = np.array(image_list_test)
y_test = np.array(y_list_test)
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

print(X_train[0])
print(y_train)
# print(y_train)

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# print(y_train[0])

# build the model
model = baseline_model()
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=1000, batch_size=1, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

