import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from imread import imread
# from PIL import Image
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
numbers_dir = "../data/numbers/"
X = []
y =[]
for i in xrange(3000):
	y.append(i)
	image = imread(numbers_dir +"img" + str(i+1)+'.bmp')
	image = image[:,:,0].flatten()
	X.append(np.array(X)/255)

X_train = X[0:2400]
y_train = X[0:2400]

X_test = X[2400:]
y_train = y[2400:]
pixels = len(X[0])
model = Sequential()
model.add(Dense(pixels, input_dim=pixels, kernel_initializer='normal', activation='relu'))
model.add(Dense(10, kernel_initializer='normal', activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size = 60, verbose =2)

scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))


