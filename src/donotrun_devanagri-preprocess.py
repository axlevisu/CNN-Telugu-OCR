#One time run
# Data from AML course
import numpy as np
from PIL import Image, ImageFilter
import glob
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
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
