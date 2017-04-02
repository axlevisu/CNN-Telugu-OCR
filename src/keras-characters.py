# keras-characters
import numpy as np
# from imread import imread
from PIL import Image, ImageFilter
import glob
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')


X = []
y =[]
# Restrict maximum width/height to 128 due to memory constraints
max_size =64.0
char_dir = '../data/characters/'
for i in xrange(147):
	folder = char_dir + 'usr_' + str(i) + '/'
	for imagefile in glob.glob(folder + '*r.tiff'):
		im = Image.open(imagefile)
		image_name = imagefile[-12:-9]

		y.append(int(image_name))		
		X.append(np.array(im)*1)

t = int(round(len(X)*0.8))		
X = np.array(X)
y = np.array(y)
X_train = X[0:t]
y_train = y[0:t]

X_test = X[t:]
y_test = y[t:]
# Convert to vectors
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

model = Sequential()
# model.add(ZeroPadding2D((1,1),input_shape=(1,None,None)))
model.add(Conv2D(50, (5, 5), input_shape=(1,None,None), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(20, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))	
model.add(Dense(50, activation='relu'))
# model.add(Dense(pixels, input_shape = shape, kernel_initializer='normal',  activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)