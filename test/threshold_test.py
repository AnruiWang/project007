import cv2
import numpy as np
from matplotlib import pyplot as plt


image = cv2.imread("G:/dlc/test/test-eight/video_picture_4_1/process_1/158.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.subplot(131)
plt.imshow(image, "gray")
plt.title("source image")
plt.xticks([])
plt.yticks([])

plt.subplot(132)
plt.hist(image.ravel(), 256)
plt.title("Histogram")
plt.xticks([])
plt.yticks([])

ret1, th1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

plt.subplot(133)
plt.imshow(th1, "gray")
plt.title("OTSU, threshold is " + str(ret1))
plt.xticks([])
plt.yticks([])
plt.show()

