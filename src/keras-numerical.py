import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from imread import imread
# from PIL import Image
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
numbers_dir = "../data/"
X = []
y =[]
for i in xrange(3000):
	y.append(1)
	image = imread(numbers_dir + str(i+1)+'.bmp')
	image = image[:,:,0].flatten()
	X.append(np.array(X))




