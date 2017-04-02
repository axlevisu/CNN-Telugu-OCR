from PIL import Image

images=map(Image.open, ['../data/numbers/img157.bmp','../data/numbers/img177.bmp','../data/numbers/img160.bmp','../data/numbers/img187.bmp'])
j=0
for im in images:
	print j
	im=im.resize((64,64),Image.ANTIALIAS)
	im.save(str(j)+'.bmp')
	j=j+1

images=map(Image.open, ['0.bmp','1.bmp','2.bmp','3.bmp'])

widths,heights=zip(*(i.size for i in images))
total_width=sum(widths)
max_height=max(heights)

new_im=Image.new('L',(total_width,max_height))
x_offset=0
for im in images:
	new_im.paste(im,(x_offset,0))
	x_offset+=im.size[0]

new_im.save('d.bmp')

