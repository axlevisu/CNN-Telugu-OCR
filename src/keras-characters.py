# keras-characters
import numpy as np
# from imread import imread
from PIL import Image
import glob
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import Dropout
# from keras.layers import Dropout
# from keras.layers import Flatten
# from keras.layers.convolutional import Conv2D
# from keras.layers.convolutional import MaxPooling2D
# from keras.utils import np_utils
# from keras import backend as K
# K.set_image_dim_ordering('th')


X = []
y =[]
# Restrict maximum width/height to 128 due to memory constraints
max_size =128.0
char_dir = '../data/characters/'
for i in xrange(147):
	folder = char_dir + 'usr_' + str(i) + '/'
	for imagefile in glob.glob(folder + '*.tiff'):
		im = Image.open(imagefile)
		try:
			y.append(int(imagefile[-11:-8]))
		except:
			print imagefile
		f = max_size/max(im.size)
		im = im.resize((int(round(f*im.size[0])),int(round((f*im.size[1])))),Image.ANTIALIAS)
		X.append(np.array(im)*1)
		# max_size = max(max_size,max(im.size))

print max_size
print len(X), len(y)