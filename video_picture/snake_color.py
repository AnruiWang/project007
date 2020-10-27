import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import data
from skimage.io import imread
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from test.bresenham_test import pixel_contrast
from video_picture.change_color_bresenham import change_color_bresenham

file = 'G:/dlc/test/test-eight/video_picture_4'
#img = Image.open('G:/dlc/test/test-eight/video_picture_1/compound/88.jpg')


def find_snake(z):
    img = imread(file + '/process/change_color_1/' + str(z) + '.jpg')
    img = rgb2gray(img)

    s = np.linspace(0, 2*np.pi, 300)
    r = 260 + 180*np.sin(s)
    c = 290 + 180*np.cos(s)
    init = np.array([r, c]).T

    snake = active_contour(gaussian(img, 3),
                       init, alpha=0.015, beta=10, gamma=0.007,
                       coordinates='rc')

    pixel_contrast(z, snake)
    """
    bresenham = make_line(snake)
    print(bresenham)

    change_color_bresenham(z, bresenham)
    """
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img, cmap=plt.cm.gray)
    # ax.plot(init[:, 1], init[:, 0], '--b', lw=3)
    ax.plot(bresenham[:, 1], bresenham[:, 0], '-r', lw=10)
    ax.set_xticks([]), ax.set_yticks([])
    ax.axis([0, img.shape[1], img.shape[0], 0])
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)

    plt.savefig(file + '/change_picture_2/' + str(z) + '.jpg', dpi=73.2, pad_inches=0.0)
    plt.show()
    """


def main():
    for z in range(71, 168):
        print(z)
        find_snake(z)


main()
