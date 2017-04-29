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
from sklearn.naive_bayes import GaussianNB
K.set_image_dim_ordering('th')


X = []
y =[]
# Restrict maximum width/height to 128 due to memory constraints
size = 32
max_size = 64
char_dir = '../data/64characters/'

#reading the dataset from the relevant directories and preprocessing it

for imagefile in glob.glob(char_dir + '*.tiff'):
	im = rescale(dilation(np.invert(io.imread(imagefile)),square(5)),float(size)/max_size)
	image_name = imagefile[-11:-8]
	y.append(int(image_name))
	X.append(im) 	

#partitioning the data as train and test dataset

t = int(round(len(X)*0.8))		
X = np.array(X)
y = np.array(y)
X = X.reshape(X.shape[0], size, size).astype('float32')

X_train = X[0:t]
y_train = y[0:t]

X_test = X[t:]
y_test = y[t:]
# Convert to vectors
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print X_train.shape
print y_train.shape

#flattening the input of train data set

a=[]
for i in range(X_train.shape[0]):
    a.append(X_train[i].flatten())

a=np.asarray(a)

#flatenning the input of test dataset

b=[]
for i in range(X_test.shape[0]):
	b.append(X_test[i].flatten())

b=np.asarray(b)

#flattening the output of train dataset

c=[]
for i in range(y_train.shape[0]):
	k=0
	for j in y_train[i]:
		if(j==1):
			c.append(k)
		else:
			k=k+1
c=np.asarray(c)

#flattening the output of test dataset

d=[]
for i in range(y_test.shape[0]):
	k=0
	for j in y_test[i]:
		if(j==1):
			d.append(k)
		else:
			k=k+1
d=np.asarray(d)

#implementing a naive bayesian classifier

model=GaussianNB()
model.fit(a,c)
predicted=model.predict(b)

print(model.score(b,d))



for i in range(10):
    print str(predicted[i])+"    "+str(y_test[i])