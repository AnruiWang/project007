import cv2


file = 'G:/dlc/test/test-eight/video_picture_10'

for z in range(10, 19):
	print(z)
	#imgfile = "G:/dlc/test/test-eight/video_picture_8_1/area/49.jpg"
	imgfile = file + '/GVF_snake/' + str(z + 1) + '.bmp'
	#imgfile = file + '/original_labeled/' + str(z + 1) + '.bmp'
	img = cv2.imread(imgfile)
	h, w, _ = img.shape

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# cv2.imshow('mask', gray)

	ret, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

	# find contour
	_, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

	#需要给一个list
	c_max = []
	for i in range(len(contours)):
		cnt = contours[i]
		area = cv2.contourArea(cnt)

		if(area < (h/1000000*w/1000000)):
			c_min = []
			c_min.append(cnt)
			cv2.drawContours(img, c_min, -1, (0, 0, 0), thickness=-1)
			continue

		c_max.append(cnt)

	cv2.drawContours(img, c_max, 0, (255, 255, 255), thickness=-1)

	cv2.imwrite(file + '/GVF_snake/con' + str(z + 1) + '.bmp', img)
	# cv2.imshow('mask', img)
