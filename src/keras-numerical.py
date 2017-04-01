import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from PIL import Image
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
numbers_dir = "../data/"
X = []
y =[]
for i in xrange(1,3001):
	bmpfile = 