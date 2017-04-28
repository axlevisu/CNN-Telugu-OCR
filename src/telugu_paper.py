# paper-implementation, zoneing
# keras-characters
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
y =[]
# Restrict maximum width/height to 128 due to memory constraints
max_size = 64
size = 32
char_dir = '../data/characters/'
for i in range(24)+range(25,147):
	folder = char_dir + 'usr_' + str(i) + '/'
	for imagefile in glob.glob(folder + '*r.tiff'):
		im = rescale(dilation(np.invert(io.imread(imagefile)),square(3)),float(size)/max_size)
		image_name = imagefile[-12:-9]
		y.append(int(image_name))
		b = np.zeros((size, size))
		b[:im.shape[0],:im.shape[1]] = im
		X.append(b.flatten()) 		

t = int(round(len(X)*0.8))		
X = np.array(X)
y = np.array(y)