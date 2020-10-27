import cv2
import numpy as np

for i in range(157, 163):
	print(i)
	img = cv2.imread('G:/dlc/test/test-eight/video_picture_4_1/process_1/' + str(i) + '.jpg', 0)
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

	opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
	cv2.imshow("open", opened)

	cv2.imwrite('C:/Users/wangx/Desktop/opened/' + str(i) + '.jpg', opened)

	cv2.destroyAllWindows()