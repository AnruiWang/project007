import numpy as np
from scipy.spatial import ConvexHull
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import matplotlib.pyplot as plt

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


#生成两列数组
def make_array(inter):
	x = np.array([0], dtype=float)
	y = np.array([0], dtype=float)
	for i in range(len(inter)):
		x = np.append(x, [inter[i][0]], axis=0)
		y = np.append(y, [inter[i][1]], axis=0)
	x = np.delete(x, 0, axis=0)
	y = np.delete(y, 0, axis=0)
	return x, y


#根据两列数组生成椭圆
def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
                      width=ell_radius_x * 2,
                      height=ell_radius_y * 2,
                      facecolor=facecolor,
                      **kwargs)

    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def main():
	mu = 2, 4
	for i in range(41, 42):
		inter = judge_intersect(i)
		print(inter)
		x, y = make_array(inter)
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.scatter(x, y, s=0.5)

		ax.axvline(c='gray', lw=1)
		ax.axhline(c='gray', lw=1)
		ax.set_xlim(0, 512)
		ax.set_ylim(0, 512)

		confidence_ellipse(x, y, ax, edgecolor='red')

		ax.scatter(mu[0], mu[1], c='red', s=3)
		plt.show()

main()

