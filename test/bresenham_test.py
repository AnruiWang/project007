from bresenham import bresenham
from PIL import Image
import numpy as np
import cv2
import math


file = "G:/dlc/test/test-eight/video_picture_4"


def change_list(b):
	for c in b:
		b[b.index(c)] = list(c)
	return b


def make_line(points):
	bresenham_test = np.array([[0, 0]], dtype=int)
	for x in range(0, len(points)-1):
		x0, y0 = math.floor(points[x][0]), math.floor(points[x][1])
		x1, y1 = math.floor(points[x+1][0]), math.floor(points[x+1][1])
		medium = change_list(list(bresenham(x0, y0, x1, y1)))
		for i in range(0, len(medium)):
			medium_test = np.array([[medium[i][0], medium[i][1]]], dtype=int)
			bresenham_test = np.concatenate((bresenham_test, medium_test), axis=0)
	medium = np.delete(bresenham_test, 0, axis=0)
	return medium


def make_pure_color(z, points):
	img = np.zeros((512, 512, 3), np.uint8)
	img.fill(0)
	cv2.imwrite(file + "/process/change_color/" + str(z) + ".jpg", img)
	img = Image.open(file + "/process/change_color/" + str(z) + ".jpg")
	medium = make_line(points)

	for i in range(0, len(medium)):
		img.putpixel((medium[i][1], medium[i][0]), (255, 255, 255))

	img = img.convert("RGB")
	img.save(file + "/process/change_color/" + str(z) + ".jpg")
	img = cv2.imread(file + "/process/change_color/" + str(z) + ".jpg")
	kernel = np.ones((3, 3), np.uint8)

	erosion = cv2.dilate(img, kernel)

	cv2.imshow("erosion", erosion)

	cv2.imwrite(file + "/process/change_color/" + str(z) + ".jpg", erosion)


def pixel_contrast(z, points):
	make_pure_color(z, points)
	img = Image.open(file + "/process/change_color_1/" + str(z) + ".jpg")
	img_contrast = Image.open(file + "/process/change_color/" + str(z) + ".jpg")

	width = img_contrast.size[0]
	height = img_contrast.size[1]

	for i in range(0, width):
		for j in range(0, height):
			data = (img_contrast.getpixel((i, j)))
			if data[0] > 100 and data[1] > 100 and data[2] > 100:
				print("图像信息：", img.getpixel((i, j)))
				img.putpixel((i, j), (0, 0, 0))

	img = img.convert("RGB")
	img.save(file + "/process/change_color_1/" + str(z) + ".jpg")


