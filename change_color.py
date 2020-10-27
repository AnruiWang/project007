from PIL import Image
import numpy as np
import cv2
file = "G:/dlc/test/test-eight/video_picture_10/original_labeled/"

for i in range(0, 55):
	print(i)
	img = Image.open(file + str(i + 1) + '.bmp')
	img1 = img.convert('RGBA')
	pixdata = img1.load()
	for y in range(400, 512):
		for x in range(0, 512):
			pixdata[x, y] = (0, 0, 0)

	img1.save(file + str(i + 1) + '.bmp')