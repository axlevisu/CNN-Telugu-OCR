# Devanagri Keras
# Data from AML course
import numpy as np
from PIL import Image, ImageFilter
import glob
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.utils import np_utils
from keras import backend as K
from skimage.morphology import square,dilation
from skimage.transform import rescale
from skimage import io
from scipy.ndimage.filters import gaussian_filter
from scipy import misc
K.set_image_dim_ordering('th')

X = []
y = []
X_test = []
y_test = []
max_size =32
train_data_dir  = '../data/devanagri-characters/train/'
test_data_dir  = '../data/devanagri-characters/test/'
# Read Lables
with open(train_data_dir + 'labels.txt','r') as f:
	for line in f:
		y.append(int(line.strip()))

y = np.array(y)
N = y.shape[0]

with open(test_data_dir + 'labels.txt','r') as f:
	for line in f:
		y_test.append(int(line.strip()))

y_test = np.array(y_test)
N_test = y_test.shape[0]
# Read Images
for i in xrange(N):
	image = io.imread(train_data_dir + str(i) + ".png")
	image = np.invert(image)
	image = rescale(dilation(image,square(7)),float(max_size)/320)
	X.append(image)

for i in xrange(N_test):
	image = io.imread(test_data_dir + str(i) + ".png")
	image = np.invert(image)
	image = rescale(dilation(image,square(7)),float(max_size)/320)
	X_test.append(image)

y = np_utils.to_categorical(y)
y_test = np_utils.to_categorical(y_test)
X = np.array(X)
X = X.reshape(X.shape[0], 1, max_size, max_size).astype('float32')
X_test = np.array(X_test)
X_test = X_test.reshape(X_test.shape[0], 1, max_size, max_size).astype('float32')
num_classes = y.shape[1]

model = Sequential()
model.add(Conv2D(1, (7, 7), padding='SAME', input_shape=(1,max_size,max_size), activation='relu'))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (5, 5), padding='SAME',  activation='relu'))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), padding='SAME',  activation='relu'))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (3, 3), padding='SAME',  activation='relu'))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))	
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, validation_data=(X_test, y_test), epochs=10, batch_size=100)
model.save('devanagri4.h5')

scores = model.evaluate(X_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
