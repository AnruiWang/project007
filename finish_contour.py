import numpy as np
import matplotlib.pyplot as plt
import skimage.color
import cv2
from skimage.color import rgb2gray
from skimage.io import imread
from skimage.filters import gaussian
from skimage.segmentation import active_contour, inverse_gaussian_gradient
from scipy.spatial import ConvexHull
from skimage.segmentation import morphological_chan_vese

points = np.loadtxt('./data/hull.txt')
file = 'G:/dlc/test/test-eight/video_picture_4_1'


def point_intersect(a, b, z):
    if (points[b][2] - points[a][2]) == 0:
        x = np.array([0], dtype=float)
        y = np.array([0], dtype=float)
        return x, y
    else:
        x = points[a][0] + [(z - points[a][2]) * (points[b][0] - points[a][0]) / (points[b][2] - points[a][2])]
        y = points[a][1] + [(z - points[a][2]) * (points[b][1] - points[a][1]) / (points[b][2] - points[a][2])]
        return x, y


#获取交点
def judge_intersect(z):
    inter = np.array([[0, 0]], dtype=float)
    hull = ConvexHull(points)
    for s in hull.simplices:
        if (points[s[0]][2] - z)*(points[s[1]][2] - z) <= 0:
            x, y = point_intersect(s[0], s[1], z)
            add = np.array([[x[0], y[0]]], dtype=float)
            if add[0][0] != 0 and add[0][1] != 0:
                inter = np.concatenate((inter, add))
        if (points[s[0]][2] - z)*(points[s[2]][2] - z) <= 0:
            x, y = point_intersect(s[0], s[2], z)
            add = np.array([[x[0], y[0]]], dtype=float)
            if add[0][0] != 0 and add[0][1] != 0:
                inter = np.concatenate((inter, add))
        if (points[s[1]][2] - z)*(points[s[2]][2] - z) <= 0:
            x, y = point_intersect(s[1], s[2], z)
            add = np.array([[x[0], y[0]]], dtype=float)
            if add[0][0] != 0 and add[0][1] != 0:
                inter = np.concatenate((inter, add))
    if len(inter) > 2:
        inter = np.delete(inter, 0, axis=0)
        hull = ConvexHull(inter)
        return inter, hull.vertices


def number_points(i):
    inter, s = judge_intersect(i)
    print(inter)
    init = np.array([[0, 0]], dtype=float)
    n = 500/(len(s) - 1)
    for k in range(len(s) - 1):
        x1, y1 = inter[s[k]][0], inter[s[k]][1]
        x2, y2 = inter[s[k+1]][0], inter[s[k+1]][1]
        init = np.concatenate((init, [[x1, y1]]))
        for m in range(1, int(n)):
            t = 1/(int(n))
            init = np.concatenate((init, [[x1+t*m*(x2-x1), y1+t*m*(y2-y1)]]))
    init = np.delete(init, 0, axis=0)
    return init


def store_evolution_in(lst):
    """Returns a callback function to store the evolution of the level sets in
    the given list.
    """

    def _store(x):
        lst.append(np.copy(x))

    return _store


for i in range(94, 157):
    print(i)
    """img_original = imread(file + '/process_1/' + str(i) + '.jpg')
    img = rgb2gray(img_original)"""

    image = cv2.imread(file + '/process_1/' + str(i) + '.jpg')
    img = skimage.color.rgb2gray(image)

    gimage = inverse_gaussian_gradient(img)
    init = number_points(i)


    print(len(init))

    """snake = active_contour(gaussian(img, 3),
                           init, alpha=0.015, beta=10, gamma=0.007,
                           coordinates='rc')"""

    evolution = []
    callback = store_evolution_in(evolution)
    snake = morphological_chan_vese(img, 35, init_level_set=init, smoothing=3,
                                 iter_callback=callback)

    fix, ax = plt.subplots()
    ax.imshow(img, cmap=plt.cm.gray)
    ax.plot(init[:, 0], init[:, 1], '--r', lw=3)
    ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
    ax.set(xlim=[0, 512], ylim=[0, 512], xlabel='x', ylabel='y')

    plt.savefig(file + '/mor_snake/' + str(i) + '.jpg')
    plt.show()