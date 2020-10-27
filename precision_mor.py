import cv2
import numpy as np

def jaccard_similarity(list1, list2):
	inter = 0
	union = 0
	for i in range(0, 512):
		for j in range(0, 512):
			if list1[i][j] == 1 and list2[i][j] == 1:
				inter = inter + 1
			if list1[i][j] == 1 or list2[i][j] == 1:
				union = union + 1
	print(inter)
	print(union)
	if union == 0:
		pre = 1
	else:
		pre = inter / union
	return pre


file = "G:/dlc/test/test-eight/video_picture_8_1"

def precision_mor(i, array):
	imgA = cv2.imread(file + "/original_labeled/con" + str(i + 1) + ".bmp", cv2.IMREAD_GRAYSCALE)
	initA = np.zeros((512, 512), dtype=np.int8)
	for i in range(0, 512):
		for j in range(0, 512):
			if(imgA[i][j] > 0):
				initA[i, j] = 1

	pre = jaccard_similarity(initA, array)

	f = open('G:/dlc/test/test-eight/video_picture_8_1/precision_mor.txt', 'a')
	f.write(str(pre) + '\n')
	print(pre)