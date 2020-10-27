import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
from PIL import Image
import os


points = np.loadtxt('./data/hull.txt')
file = 'G:/dlc/test/test-eight/video_picture_8'


#判断每一张多少点
def judge_points(z):
    inter = np.array([[0, 0]], dtype=float)
    for i in range(len(points)):
        if points[i][2] == z:
            add = np.array([[points[i][0], points[i][1]]], dtype=float)
            inter = np.concatenate((inter, add))

    if len(inter) > 2:
        inter = np.delete(inter, 0, axis=0)
        make_picture(inter, z)
    else:
        make_original(z)


def make_picture(inter, z):
    print('ok')
    img = Image.open(os.path.join('compound', file + '/original/' +
                                  str(z + 1) + '.jpg'))
    plt.imshow(img)

    hull_2d = ConvexHull(inter)
    plt.plot(inter[:, 0], inter[:, 1], 'bo', markersize=1)
    for s in hull_2d.simplices:
        s = np.append(s, s[0])
        plt.plot(inter[s, 0], inter[s, 1], 'r-')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.xlim(0, 512)
    plt.ylim(0, 512)
    plt.savefig(file + '/after_compound/' + str(z+1) + '.jpg')
    plt.show()


def make_original(z):
    img = Image.open(os.path.join('compound', file + '/original/' +
                                  str(z + 1) + '.jpg'))
    plt.imshow(img)

    # plt.plot(inter[0, 0], inter[0, 1], 'bo', markersize=1)
    plt.xlabel('x')
    plt.ylabel('y')

    plt.xlim(0, 512)
    plt.ylim(0, 512)
    plt.savefig(file + '/after_compound/' + str(z + 1) + '.jpg')
    plt.show()


def main():
    for z in range(109):
        print(z)
        judge_points(z)


main()
