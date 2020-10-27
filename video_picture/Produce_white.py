import cv2 as cv
import numpy as np

img = np.zeros([512, 512, 3], np.uint8)
img[:, :, 0] = np.zeros([512, 512]) + 0
img[:, :, 1] = np.zeros([512, 512]) + 0
img[:, :, 2] = np.zeros([512, 512]) + 0
cv.imshow('image', img)
cv.imwrite('G:/dlc/test/test-eight/test/black.bmp', img)
cv.waitKey(0)