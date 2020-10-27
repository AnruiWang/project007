import matplotlib.pyplot as plt
from skimage import measure, data, color
from PIL import Image
import os
from skimage.io import imread

img = imread("G:/dlc/test/test-eight/video_picture_4/process/change_color_1/121.jpg")
img=color.rgb2gray(img)

contours = measure.find_contours(img, 0.5)

fig, axes = plt.subplots(1, 2, figsize=(8, 8))
ax0, ax1 = axes.ravel()
ax0.imshow(img,plt.cm.gray)
ax0.set_title('original image')

rows,cols=img.shape
ax1.axis([0,rows,cols,0])
for n, contour in enumerate(contours):
    ax1.plot(contour[:, 1], contour[:, 0], linewidth=2)
ax1.axis('image')
ax1.set_title('contours')
plt.show()