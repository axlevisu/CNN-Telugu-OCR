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
max_size = 50
char_dir = '../data/original-characters/'
for i in range(24)+range(25,147):
	folder = char_dir + 'usr_' + str(i) + '/'
	for imagefile in glob.glob(folder + '*.tiff'):
		im  = io.imread(imagefile)
		f = float(max_size)/max(im.shape)
		im = rescale(im,f)
		b = np.zeros((max_size, max_size))
		b[:im.shape[0],:im.shape[1]] = im
		features =[]
		for i in xrange(10):
			for j in xrange(10):
				features.append(sum(sum(b[5*i:5 + 5*i,5*j:5+5*j])))
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