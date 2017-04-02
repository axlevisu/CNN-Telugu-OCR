# keras-nnumerical
import numpy as np
# from imread import imread
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
numbers_dir = "../data/numbers/"
X = []
y =[]
for i in xrange(3000):
	y.append(i%10)
	image = Image.open(numbers_dir +"img" + str(i+1)+'.bmp')
	image = np.array(image)[:,:,0].flatten()
	X.append(np.array(image)/255)

X = np.array(X)
y = np.array(y)
shape = X[0].shape
X = X.reshape(X.shape[0], 1, 32, 32).astype('float32')


X_train = X[0:2400]
y_train = y[0:2400]

X_test = X[2400:]
y_test = y[2400:]
# Convert to vectors
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)



num_classes = y_test.shape[1]
pixels = len(X[0])

model = Sequential()
model.add(Conv2D(30, (5, 5), input_shape=(1, 32, 32), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(15, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))	
model.add(Dense(50, activation='relu'))
# model.add(Dense(pixels, input_shape = shape, kernel_initializer='normal',  activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=100)

scores = model.evaluate(X_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))


