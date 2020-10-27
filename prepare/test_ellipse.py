import numpy as np
from scipy.spatial import ConvexHull
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import matplotlib.pyplot as plt
from prepare import lowner_ellipse as le


points = np.loadtxt('../data/hull.txt')


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
        return inter


def main():
	for i in range(41, 42):
		inter = judge_intersect(i)
		print(inter)
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.scatter(inter[:, 0], inter[:, 1], s=0.5)

		ax.set_xlim(0, 512)
		ax.set_ylim(0, 512)

		enclosing_ellipse = le.welzl(points)
		print(enclosing_ellipse)

		le.plot_ellipse(enclosing_ellipse, str='k--')
		plt.show()