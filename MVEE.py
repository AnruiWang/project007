import os
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from PIL import Image
import skimage.color
import cv2
from mpl_toolkits.mplot3d import Axes3D
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.segmentation import active_contour, morphological_geodesic_active_contour
from skimage.segmentation import morphological_chan_vese
from skimage.segmentation import inverse_gaussian_gradient
from skimage.segmentation import checkerboard_level_set

pi = np.pi
sin = np.sin
cos = np.cos

points = np.loadtxt('./data/hull.txt')
file = 'G:/dlc/test/test-eight/video_picture_4_1'


def mvee(points, tol = 0.001):
    N, d = points.shape
    Q = np.column_stack((points, np.ones(N))).T
    err = tol+1.0
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


def intersect_function(z, A, centroid):
    U, D, V = la.svd(A)  # V 是椭球体方向的旋转矩阵
    rx, ry, rz = 1./np.sqrt(D)  # rx, ry, rz 是半径
    u, v = np.mgrid[0:2*pi:20j, -pi/2:pi/2:10j]

    Radius_function = np.diag(np.array([(1/rx**2), (1/ry**2), (1/rz**2)]))
    Z = np.dot(np.dot(V, Radius_function), V.T)  # 椭球中心点的方程Z
    P11 = np.array([[Z[0][0], Z[0][1]],
                    [Z[1][0], Z[1][1]]])
    S1 = np.array([[centroid[0]],
                  [centroid[1]]])
    P12 = np.array([[Z[0][2]], [Z[1][2]]])
    zs2 = float(z) - centroid[2]
    P12_after = np.array([[zs2 * P12[0][0]],
                          [zs2 * P12[1][0]]])

    C_ellipse = np.dot(np.linalg.inv(P11), (np.dot(P11, S1) - P12_after))  # 截面的椭圆的中心点

    # 椭圆的方程
    Q22 = np.dot(np.dot(S1.T, P11), S1) - 2 * zs2 * np.dot(P12.T, S1) + np.array([(zs2 ** 2) * Z[2][2] - 1])
    r = 1/(np.dot(np.dot(C_ellipse.T, P11), C_ellipse) - Q22)
    Q = np.array([[r[0][0]*P11[0][0], r[0][0]*P11[0][1]],
                  [r[0][0]*P11[1][0], r[0][0]*P11[1][1]]])
    # 椭圆的特征值及特征向量
    value, vector = np.linalg.eig(Q)

    print("r:", r)
    if r[0][0] < 0:
        non_intersect(z)
    else:
        intersect(z, Q, C_ellipse)

"""
    def ellipse(u, v):
        x = rx*cos(u)*cos(v)
        y = ry*sin(u)*cos(v)
        z = rz*sin(v)
        return x, y, z


    E = np.dstack(ellipse(u, v))
    # 旋转之后得到的新的椭圆的位置
    E = np.dot(E, V) + centroid
    x, y, z = np.rollaxis(E, axis=-1)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(x, y, z, cstride=2, rstride=2, alpha=0.5)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.set_xlim3d(0, 512)
    ax.set_ylim3d(0, 512)
    ax.set_zlim3d(0, 180)

    plt.show()
"""

def store_evolution_in(lst):
    """Returns a callback function to store the evolution of the level sets in
    the given list.
    """

    def _store(x):
        lst.append(np.copy(x))

    return _store



def intersect(z, Q, C_ellipse):
    print("1")
    # img = Image.open(os.path.join('compound', file + '/original/' + str(z + 1) + '.jpg'))
    img = imread( file + '/process_1/' + str(z + 1) + '.jpg')
    img = rgb2gray(img)
    gimage = inverse_gaussian_gradient(img)
    # 椭圆的特征值及特征向量
    value, vector = np.linalg.eig(Q)
    # 椭圆的长短轴
    a = 1 / np.sqrt(value[0])
    b = 1 / np.sqrt(value[1])

    """
    points_c = np.array([[0, 0]], dtype=float)
    for i in range(0, 200):
        t = 0 + i * (2*pi)/200
        x = a * cos(t)
        y = b * sin(t)
        print("t", t)
        print("x, y:", x, y)
        medium = np.array([[x, y]], dtype=float)
        points_c = np.concatenate((points_c, medium), axis=0)
    points_c = np.delete(points_c, 0, axis=0)
    """

    s = np.linspace(0, 2 * np.pi, 1000)
    r = a * np.cos(s)
    c = b * np.sin(s)
    points_c = np.array([r, c]).T


    points_a = np.array([[0, 0]], dtype=float)
    for i in range(0, 1000):
        U_inv = np.linalg.inv(vector)
        Y = np.array([[points_c[i][0], points_c[i][1]]])
        medium_T = np.dot(U_inv, Y.T) + C_ellipse
        points_a = np.concatenate((points_a, medium_T.T), axis=0)
    points_a = np.delete(points_a, 0, axis=0)

    for i in range(0, 1000):
        medium = points_a[i][0]
        points_a[i][0] = points_a[i][1]
        points_a[i][1] = medium

    #init_ls = checkerboard_level_set(img.shape, 6)

    init_ls = np.zeros(img.shape, dtype=np.int8)
    init_ls[10:-10, 10:-10] = 1
    """for i in range(0, len(points_a)):
        init_ls[int(points_a[i][0])][int(points_a[i][1])] = 1"""


    print(init_ls.shape)

    evolution = []
    callback = store_evolution_in(evolution)
    snake = morphological_geodesic_active_contour(gimage, 500, init_ls, smoothing=1, balloon=-1,
                                           threshold=0.69,
                                           iter_callback=callback)

    snake = active_contour(gaussian(img, 3),
                           points_a, alpha=0.015, beta=10, gamma=0.007,
                           coordinates='rc')

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img, cmap="gray")
    ax.set_axis_off()
    ax.contour(snake, [0.5], colors='r')
    plt.savefig(file + '/mor_snake/' + str(z + 1) + '.jpg')
    plt.show()


def non_intersect(z):
    print("2")
    img = Image.open(os.path.join('compound', file + '/process_1/' +
                                  str(z + 1) + '.jpg'))
    plt.imshow(img)

    # plt.plot(inter[0, 0], inter[0, 1], 'bo', markersize=1)
    plt.xlabel('x')
    plt.ylabel('y')

    plt.xlim(0, 512)
    plt.ylim(0, 512)
    plt.savefig(file + '/mor_snake/' + str(z + 1) + '.jpg')


def main():
    A, centroid = mvee(points)
    for z in range(40, 180):
        print(z)
        intersect_function(z, A, centroid)


main()