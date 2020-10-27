#对三维凸包进行截面并与原图合并

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

points = np.loadtxt('./data/hull.txt')
n = 0
file = 'G:/dlc/test/test-eight/video_picture_9'


#找交点
def point_intersect(a, b, z):
    if (points[b][2] - points[a][2]) == 0:
        x = np.array([0], dtype=float)
        y = np.array([0], dtype=float)
        return x, y
    else:
        x = points[a][0] + [(z - points[a][2]) * (points[b][0] - points[a][0]) / (points[b][2] - points[a][2])]
        y = points[a][1] + [(z - points[a][2]) * (points[b][1] - points[a][1]) / (points[b][2] - points[a][2])]
        return x, y

#判断线段与该平面是否相交
def judge_intersect(z):
    inter = np.array([[0, 0]], dtype=float) #第一行不可用，注意省略
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
    if len(inter) > 3:
        inter = np.delete(inter, 0, axis=0)
        print(inter)
        hull_2d_f(inter, z)
    else:
        picture(z)


def hull_2d_f(inter, z):

    img = Image.open(os.path.join('compound', file +'/original/' +
                                  str(z+1) + '.jpg'))
    plt.cla()
    plt.imshow(img)

    hull_2d = ConvexHull(inter)
    plt.plot(inter[:, 0], inter[:, 1], 'bo', markersize=1)
    for s in hull_2d.simplices:
        s = np.append(s, s[0])
        plt.plot(inter[s, 0], inter[s, 1], 'r-')
    #plt.xlabel('x')
    #plt.ylabel('y')

    #plt.xlim(0, 512)
    #plt.ylim(0, 512)
    plt.savefig(file + '/compound/' + str(z+1) + '.jpg')
    plt.show()


#平面与凸包无交点时生成的图片
def picture(z):

    img = Image.open(os.path.join('compound', file + '/original/' +
                                  str(z + 1) + '.jpg'))
    plt.cla()
    plt.imshow(img)

    #plt.plot(inter[0, 0], inter[0, 1], 'bo', markersize=1)
    #plt.xlabel('x')
    #plt.ylabel('y')

    #plt.xlim(0, 512)
    #plt.ylim(0, 512)
    plt.savefig(file + '/compound/' + str(z+1) + '.jpg')
    plt.show()

def main():
    for z in range(200):
        print(z)
        judge_intersect(z)

main()

