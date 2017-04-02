# preprocess-characters
import numpy as np
from PIL import Image, ImageFilter
import glob

# Restrict maximum width/height to 128 due to memory constraints
max_size =64.0
char_dir = '../data/characters/'
for i in xrange(147):
	folder = char_dir + 'usr_' + str(i) + '/'
	for imagefile in glob.glob(folder + '*.tiff'):
		im = Image.open(imagefile)
		# Dilating Image before resize
		im = im.filter(ImageFilter.MinFilter(7))
		im = im.convert('1')
		# Resizing Image
		f = max_size/max(im.size)
		im = im.resize((int(round(f*im.size[0])),int(round((f*im.size[1])))),Image.ANTIALIAS)
		im.save( imagefile[:-11] + '000' + imagefile[-11:-5] + 'r.tiff' )

