# paper-implementation, zoneing
# keras-characters
import numpy as np
import glob
from skimage.morphology import square,dilation
from skimage.transform import rescale
from skimage import io
from scipy.ndimage.filters import gaussian_filter
from scipy import misc
from sklearn import svm, linear_model, neighbors

X = []
y =[]
# Restrict maximum width/height to 128 due to memory constraints
max_size = 64
char_dir = '../data/64characters/'
for imagefile in glob.glob(char_dir + '*.tiff'):
	im  = io.imread(imagefile)
	im = dialtion(np.invert(im),square(3))
	features =[]
	for i in xrange(8):
		for j in xrange(8):
			features.append(sum(sum(im[4*i:4 + 4*i,4*j:4+4*j])))
	image_name = imagefile[-11:-8]
	y.append(int(image_name))
	X.append(features) 		

print "done reading"
t = int(round(len(X)*0.8))		
X = np.array(X)
y = np.array(y)
Z = np.c_[X,y]
np.random.shuffle(Z)
X = Z[:,:-1]
y = Z[:,-1]

X_train = X[0:t]
y_train = y[0:t]

X_test = X[t:]
y_test = y[t:]

clf = neighbors.KNeighborsClassifier(7)
clf.fit(X_train,y_train)
# y_pred = clf.predict(X_test)
print "Score:", clf.score(X_test,y_test)