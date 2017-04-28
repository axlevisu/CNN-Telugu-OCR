# preprocess-characters
import numpy as np
from PIL import Image, ImageFilter
import glob
import os
# Restrict maximum width/height to 128 due to memory constraints


def scale(image, max_size, method=Image.ANTIALIAS):
    im_aspect = float(image.size[0])/float(image.size[1])
    out_aspect = float(max_size[0])/float(max_size[1])
    if im_aspect >= out_aspect:
        scaled = image.resize((max_size[0], int((float(max_size[0])/im_aspect) + 0.5)), method)
    else:
        scaled = image.resize((int((float(max_size[1])*im_aspect) + 0.5), max_size[1]), method)
 
    offset = (((max_size[0] - scaled.size[0]) / 2), ((max_size[1] - scaled.size[1]) / 2))
    back = Image.new("1", max_size, "white")
    back.paste(scaled, offset)
    return back


max_size =64
char_dir = '../data/original-characters/'
save_dir = '../data/64characters/'	
for i in range(24) + range(25,147):
	folder = char_dir + 'usr_' + str(i) + '/'
	for imagefile in glob.glob(folder + '*.tiff'):
		im = Image.open(imagefile)
		im = scale(im,(64,64))
		im.save( save_dir + str(i) + imagefile[-11:-5] + '.tiff' )


# for i in xrange(147):
# 	folder =  'usr_' + str(i) + '/'
# 	for imagefile in glob.glob(folder + '*.tiff'):
# 		if imagefile[-6] !='r':
# 			os.remove(imagefile)
