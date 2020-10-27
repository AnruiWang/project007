import cv2
import numpy as np
from matplotlib import pyplot as plt


def create_image(z, snake, file):
	img = np.zeros([512, 512, 3], np.uint8)
	print(img.shape)
	fig, ax = plt.subplots()
	print("dpi", fig.dpi)
	ax.imshow(img, cmap='gray')
	ax.set_axis_off()
	ax.plot(snake[:, 1], snake[:, 0], '-w', lw=1)
	#plt.rcParams['savefig.dpi'] = 512
	#plt.rcParams['figure.dpi'] = 512
	plt.savefig(file + '/area/' + str(z + 1) + '.jpg', bbox_inches='tight',dpi=(512*100)/369, pad_inches=0.0)

	draw_contour(z, file)
	# draw_contour_original(z, file)


def draw_contour(z, file):
	# imgfile = "G:/dlc/test/test-eight/video_picture_8_1/area/49.jpg"
	imgfile = file + '/area/' + str(z + 1) + '.jpg'
	img = cv2.imread(imgfile)
	h, w, _ = img.shape

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	ret, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

	# find contour
	_, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

	# 需要给一个list
	c_max = []
	for i in range(len(contours)):
		cnt = contours[i]
		area = cv2.contourArea(cnt)

		if (area < (h / 100000000 * w / 100000000)):
			c_min = []
			c_min.append(cnt)
			cv2.drawContours(img, c_min, -1, (0, 0, 0), thickness=-1)
			continue

		c_max.append(cnt)

	cv2.drawContours(img, c_max, 0, (255, 255, 255), thickness=-1)

	cv2.imwrite(file + "/area/con" + str(z+1) + ".jpg", img)
