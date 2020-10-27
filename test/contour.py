import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import data
from skimage.io import imread
from skimage.filters import gaussian
from skimage.segmentation import active_contour

file = 'G:/dlc/test/test-eight/video_picture_4/change_color/'
#img = Image.open('G:/dlc/test/test-eight/video_picture_1/compound/88.jpg')
img = imread('G:/dlc/test/test-eight/video_picture_4/change_color/124.jpg')
img = rgb2gray(img)

s = np.linspace(0, 2*np.pi, 300)
r = 320 + 70*np.sin(s)
c = 250 + 50*np.cos(s)
init = np.array([r, c]).T
print(init)

snake = active_contour(gaussian(img, 3),
                       init, alpha=0.015, beta=10, gamma=0.007,
                       coordinates='rc')

print(snake)
fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(img, cmap=plt.cm.gray)
ax.plot(init[:, 1], init[:, 0], '--b', lw=3)
ax.plot(snake[:, 1], snake[:, 0], '-r', lw=3)
ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, img.shape[1], img.shape[0], 0])

plt.show()