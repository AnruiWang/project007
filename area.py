import cv2

img = cv2.imread("G:/dlc/test/test-eight/video_picture_8_1/original_labeled/45.bmp")
# img = cv2.imread("G:/dlc/train/data/batch_1/kuoda/quandongguang/label/QUANDONGGUANG_Axial+066.00-000.bmp")
h, w, _ = img.shape

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(gray, 127, 255, 4)

#find contour
_, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

c_max = []
for i in range(len(contours)):
	cnt = contours[i]
	area = cv2.contourArea(cnt)

	if (area < (h/10*w/10)):
		c_min = []
		c_min.append(cnt)
		cv2.drawContours(img, c_min, -1, (0, 0, 0), thickness=-1)
		continue

	c_max.append(cnt)

cv2.drawContours(img, c_max, -1, (255, 255, 255), thickness=-1)

cv2.imwrite('mask.png', img)
cv2.imshow('mask', img)
cv2.waitKey(0)
