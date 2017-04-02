from PIL import Image

images=map(Image.open, ['../data/characters/usr_1/000012t01r.tiff','../data/characters/usr_4/000012t01r.tiff','../data/characters/usr_3/000012t01r.tiff','../data/characters/usr_6/000012t01r.tiff'])
j=0
for im in images:
	print j
	im=im.resize((64,64),Image.ANTIALIAS)
	im.save(str(j)+'.tiff')
	j=j+1

images=map(Image.open, ['0.tiff','1.tiff','2.tiff','3.tiff'])

widths,heights=zip(*(i.size for i in images))
total_width=sum(widths)
max_height=max(heights)

new_im=Image.new('L',(total_width,max_height))
x_offset=0
for im in images:
	new_im.paste(im,(x_offset,0))
	x_offset+=im.size[0]

new_im.save('b.tiff')

