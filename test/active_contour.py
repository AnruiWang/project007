import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread
import skimage.data as data
from skimage.filters import gaussian
import skimage.segmentation as seg
import skimage.filters as filters
import skimage.color as color
from PIL import Image

img = Image.open('G:/dlc/test/test-eight/video_picture_1/original/95.jpg')
img = imread('G:/dlc/test/test-eight/video_picture_1/original/95.jpg')

def image_show(image, nrows=1, ncols=1, cmap='gray'):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 14))
    ax.imshow(image, cmap='gray')
    ax.axis('off')
    return fig, ax


def circle_points(resolution, center, radius):
    radians = np.linspace(0, 2 * np.pi, resolution)
    c = center[1] + radius * np.cos(radians)
    r = center[0] + radius * np.sin(radians)

    return np.array([c, r]).T

image_gray = color.rgb2gray(img)
points = circle_points(100, [260,200], 40)[:-1]

fig, ax = image_show(img)
ax.plot(points[:, 0], points[:, 1], '--r', lw=3)

#snake = seg.active_contour(image_gray, points, coordinates='rc')
snake = seg.active_contour(gaussian(img, 3),
                       points, alpha=0.015, beta=10, gamma=0.01,
                       coordinates='rc')
fig, ax = image_show(img)
ax.plot(points[:, 0], points[:, 1], '--r', lw=3)
ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
plt.show()