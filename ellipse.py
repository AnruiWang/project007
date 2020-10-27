"""generate ellipse"""
from _datetime import datetime
import os
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import skimage
from PIL import Image
from scipy.spatial import ConvexHull
from skimage.io import imread
from skimage.filters import gaussian
from skimage.color import rgb2gray
from skimage.segmentation import active_contour, morphological_chan_vese, morphological_geodesic_active_contour, inverse_gaussian_gradient, checkerboard_level_set
from matplotlib.patches import Ellipse
from test.bresenham_test import make_line
from video_picture.change_color_bresenham import change_color_bresenham
import cv2
import pandas as pd
import openpyxl
from scipy import misc
import imageio
from Draw_picture import create_image
from precision_mor import precision_mor

points = np.loadtxt('./data/hull.txt')
file = 'G:/dlc/test/test-eight/video_picture_8_1'

def mvee(points, tol=0.0001):
    N, d = points.shape
    Q = np.column_stack((points, np.ones(N))).T
    err = tol + 1.0
    u = np.ones(N)/N
    while err > tol:
        X = np.dot(np.dot(Q, np.diag(u)), Q.T)
        M = np.diag(np.dot(np.dot(Q.T, la.inv(X)), Q))
        jdx = np.argmax(M)
        step_size = (M[jdx]-d-1.0)/((d+1)*(M[jdx]-1.0))
        new_u = (1-step_size)*u
        new_u[jdx] += step_size
        err = la.norm(new_u-u)
        u = new_u
    c = np.dot(u, points)
    A = la.inv(np.dot(np.dot(points.T, np.diag(u)), points)
               - np.multiply.outer(c, c))/d
    return A, c


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
        make_ellipse(z, inter)
    else:
        picture(z)


def ellipse_points(centroid, a, b, angle):
    print(a, b)
    print("angle", angle)
    s = np.linspace(0, 2 * np.pi, 200)
    r = a * np.cos(s)
    c = b * np.sin(s)
    points_c = np.array([c, r]).T

    point_a = np.array([[0, 0]], dtype=float)
    for i in range(len(points_c)):
        x = points_c[i][0] * np.cos(angle*np.pi/180) - points_c[i][1] * np.sin(angle*np.pi/180) + centroid[0]
        y = points_c[i][0] * np.sin(angle*np.pi/180) + points_c[i][1] * np.cos(angle*np.pi/180) + centroid[1]
        medium = np.array([[y, x]], dtype=float)
        point_a = np.concatenate((point_a, medium), axis=0)
    points_a = np.delete(point_a, 0, axis=0)

    return points_a


def make_ellipse_range(centroid, a, b, angle):
    a = a + 50
    b += 50
    init_ls = np.zeros((512, 512), dtype=np.int8)
    for z in range(0, 512):
        Y = z - centroid[0]
        A = np.square(a) * np.square(np.cos(angle*np.pi/180)) + np.square(b) * np.square(np.sin(angle*np.pi/180))
        B = 2 * np.cos(angle*np.pi/180) * np.sin(angle*np.pi/180) * (np.square(a) - np.square(b)) * Y
        C = (np.square(b) * np.square(np.cos(angle*np.pi/180)) + np.square(a) * np.square(np.sin(angle*np.pi/180))) * np.square(Y) - np.square(a * b)
        triangle = np.square(B) - 4 * A * C
        # print(triangle)
        if (triangle <= 0):
            continue
        else:
            X1 = (-B + np.sqrt(triangle)) / (2 * A) + centroid[1]
            X2 = (-B - np.sqrt(triangle)) / (2 * A) + centroid[1]
            for u in range(0, 512):
                if ((u >= X2) & (u <= X1)) :
                    init_ls[u, z] = 1

    return init_ls


def store_evolution_in(lst):
    """Returns a callback function to store the evolution of the level sets in
    the given list.
    """

    def _store(x):
        lst.append(np.copy(x))

    return _store


def make_ellipse(z, inter):
    # img = Image.open(os.path.join('compound', file + '/original/' + str(z + 1) + '.jpg'))

    img = cv2.imread(file + '/process_1/' + str(z + 1) + '.jpg')
    img = skimage.color.rgb2gray(img)
    gimage = inverse_gaussian_gradient(img)

    A, centroid = mvee(inter)
    U, D, V = la.svd(A)
    rx, ry = 1./np.sqrt(D)
    a, b = rx, ry
    arcsin = -1. * np.rad2deg(np.arcsin(V[0][0]))
    arccos = np.rad2deg(np.arccos(V[0][1]))

    alpha = arccos if arcsin > 0. else -1. * arccos
    print("angle", alpha)
    """
    plt.cla()
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')"""

    # points_a = ellipse_points(centroid, a, b, angle=alpha)


    # Initial level set

    init_ls = make_ellipse_range(centroid, a, b, angle=alpha)

    #init_ls = np.zeros(img.shape, dtype=np.int8)
    #init_ls[150:-150, 150:-150] = 1
    """
    init_ls = np.zeros(img.shape, dtype=np.int8)
    init = cv2.imread('C:/Users/wangx/Desktop/124.png')
    init = cv2.cvtColor(init, cv2.COLOR_RGB2GRAY)

    for i in range(0, 512):
        for j in range(0, 512):
            if init[i][j] > 100 :
                init_ls[i][j] = 1
            else:
                init_ls[i][j] = 0
    
    data = pd.DataFrame(init_ls)
    writer = pd.ExcelWriter('C:/Users/wangx/Desktop/numpy.xlsx')
    data.to_excel(writer, 'Sheet1', float_format='%.5f')
    writer.save()
    writer.close()
    #np.savetxt('C:/Users/wangx/Desktop/numpy.txt', init_ls)
    #init_ls = np.array(init_ls)
    #print(init_ls)
    #init_ls = skimage.color.rgb2gray(init_ls)
    """
    evolution = []
    callback = store_evolution_in(evolution)

    #snake = morphological_chan_vese(img, 35, init_level_set=init_ls, smoothing=3, iter_callback=callback)


    snake = morphological_geodesic_active_contour(gimage, 100, init_ls,
                                               smoothing=4, balloon=-1,
                                               threshold=0.69,
                                               iter_callback=callback)
    #np.savetxt('G:/dlc/test/test-eight/video_picture_8_1/mor_snake.txt', snake)

    #imageio.imwrite('G:/dlc/test/test-eight/video_picture_8_1/mor_snake_10/'+ str(z + 1) + '.png', snake)

    # create_image(z=z, snake=snake, file=file)
    precision_mor(z, snake)
    """
    snake = active_contour(gimage,
                           points_a, alpha=0.015, beta=5, gamma=0.007,
                           coordinates='rc')"""

    fig, ax = plt.subplots()
    ax.imshow(img, cmap=plt.cm.gray)
    ax.set_axis_off()
    #ax.imshow(init, cmap=plt.cm.gray)
    # plt.scatter(points_c[:, 0], points_c[:, 1], s=5, zorder=1)
    # ax.plot(inter[:, 0], inter[:, 1], '--b', zorder=1)
    # ax.plot(points_a[:, 1], points_a[:, 0], '--b', lw=3)
    #ax.plot(snake[:, 1], snake[:, 0], '-r', lw=3)

    ax.contour(snake, [0.5], colors='r', lw=2)
    plt.savefig(file + '/mor_snake_sqrt_100/' + str(z + 1) + '.jpg', bbox_inches='tight', dpi=fig.dpi, pad_inches=0.0)


def picture(z):
    img = cv2.imread(file + '/process_1/' + str(z + 1) + '.jpg')
    plt.cla()
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.imshow(img, cmap=plt.cm.gray)
    ax.set_axis_off()
    plt.savefig(file + '/mor_snake_after/' + str(z + 1) + '.jpg', bbox_inches='tight', dpi=fig.dpi, pad_inches=0.0)


def main():
    a = datetime.now()
    for z in range(32, 71):
        print(z)
        judge_intersect(z)
    b = datetime.now()
    print((b-a).seconds)

if __name__ == "__main__":
    main()