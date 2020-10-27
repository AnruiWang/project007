import cv2
import numpy as np


def main():
	# 1.导入图片
	img_src = cv2.imread("G:/dlc/train/data/batch_1/kuoda/quandongguang/label/QUANDONGGUANG_Axial+073.50-000.bmp")

	# 2.灰度化与二值化
	img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
	ret, img_bin = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

	# 3.连通域分析
	img_contour, contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	# 4.轮廓面积打印
	img_contours = []
	print(contours[0])
	print(len(contours))
	for i in range(len(contours)):
		area = cv2.contourArea(contours[i])
		print("轮廓 %d 的面积是:%d" % (i, area))

		img_temp = np.zeros(img_src.shape, np.uint8)
		img_contours.append(img_temp)

		cv2.drawContours(img_contours[i], contours, i, (255, 255, 255), -1)
		cv2.imshow("%d" % i, img_contours[i])

	# 5.显示结果
	cv2.imshow("img_bin", img_bin)
	cv2.imshow("img_src", img_src)
	cv2.waitKey()
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()