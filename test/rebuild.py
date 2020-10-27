import cv2
import numpy as np
from skimage import morphology
import matplotlib.pyplot as plt
from skimage import measure, color


img = cv2.imread("G:/dlc/test/test-eight/video_picture_4_1/original/100.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 选择灰度值大于180
retval, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
cv2.imshow("thresh", thresh)
# 开运算
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
cv2.imshow("opened", opened)
# 寻找联通区域
image, contours, hierarch = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
for i in range(len(contours)):
	area = cv2.contourArea(contours[i])
	if area < 700:
		cv2.drawContours(image, [contours[i]], 0, 0, -1)
# 图片相减
bitwiseXor = cv2.bitwise_xor(opened, current_bw)
cv2.imshow("bitwise", bitwiseXor)
# 闭运算
# open = cv2.morphologyEx(bitwiseXor, cv2.MORPH_OPEN, kernel)
closed = cv2.morphologyEx(bitwiseXor, cv2.MORPH_CLOSE, kernel)

cv2.imshow("closed", closed)

