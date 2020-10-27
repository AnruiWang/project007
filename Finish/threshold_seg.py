import os
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from PIL import Image
from scipy.spatial import ConvexHull
from skimage.io import imread
from skimage.filters import gaussian
from skimage.color import rgb2gray
from skimage.segmentation import active_contour
from matplotlib.patches import Ellipse
from test.bresenham_test import make_line
from video_picture.change_color_bresenham import change_color_bresenham
import cv2
from Draw_picture import create_image

points = np.loadtxt('../data/hull.txt')
file = 'G:/dlc/test/test-eight/video_picture_10'


def mvee(points, tol=0.0001):
	N, d = points.shape
	Q = np.column_stack((points, np.ones(N))).T
	err = tol + 1.0
	u = np.ones(N) / N
	while err > tol:
		X = np.dot(np.dot(Q, np.diag(u)), Q.T)
		M = np.diag(np.dot(np.dot(Q.T, la.inv(X)), Q))
		jdx = np.argmax(M)
		step_size = (M[jdx] - d - 1.0) / ((d + 1) * (M[jdx] - 1.0))
		new_u = (1 - step_size) * u
		new_u[jdx] += step_size
		err = la.norm(new_u - u)
		u = new_u
	c = np.dot(u, points)
	A = la.inv(np.dot(np.dot(points.T, np.diag(u)), points)
			   - np.multiply.outer(c, c)) / d
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


# 获取交点
def judge_intersect(z):
	inter = np.array([[0, 0]], dtype=float)
	hull = ConvexHull(points)
	for s in hull.simplices:
		if (points[s[0]][2] - z) * (points[s[1]][2] - z) <= 0:
			x, y = point_intersect(s[0], s[1], z)
			add = np.array([[x[0], y[0]]], dtype=float)
			if add[0][0] != 0 and add[0][1] != 0:
				inter = np.concatenate((inter, add))
		if (points[s[0]][2] - z) * (points[s[2]][2] - z) <= 0:
			x, y = point_intersect(s[0], s[2], z)
			add = np.array([[x[0], y[0]]], dtype=float)
			if add[0][0] != 0 and add[0][1] != 0:
				inter = np.concatenate((inter, add))
		if (points[s[1]][2] - z) * (points[s[2]][2] - z) <= 0:
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
		x = points_c[i][0] * np.cos(angle * np.pi / 180) - points_c[i][1] * np.sin(angle * np.pi / 180) + centroid[0]
		y = points_c[i][0] * np.sin(angle * np.pi / 180) + points_c[i][1] * np.cos(angle * np.pi / 180) + centroid[1]
		medium = np.array([[y, x]], dtype=float)
		point_a = np.concatenate((point_a, medium), axis=0)
	points_a = np.delete(point_a, 0, axis=0)

	return points_a


def make_ellipse(z, inter):
	# img = Image.open(os.path.join('compound', file + '/original/' + str(z + 1) + '.jpg'))

	img = cv2.imread(file + '/process_1/' + str(z + 1) + '.jpg')
	# img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

	A, centroid = mvee(inter)
	U, D, V = la.svd(A)
	rx, ry = 1. / np.sqrt(D)
	a, b = rx, ry
	arcsin = -1. * np.rad2deg(np.arcsin(V[0][0]))
	arccos = np.rad2deg(np.arccos(V[0][1]))

	alpha = arccos if arcsin > 0. else -1. * arccos

	points_a = ellipse_points(centroid, a, b, angle=alpha)

	thre_img = threshold(z)
	snake_thre = active_contour(gaussian(thre_img, 3), points_a, alpha=0.015, beta=5, gamma=0.007, coordinates='rc')
	create_image(z=z, snake=snake_thre, file=file)
	"""snake = active_contour(gaussian(img, 3),
						   snake_thre, alpha=0.015, beta=5, gamma=0.007,
						   coordinates='rc')"""

	plt.cla()
	print("shape", img.shape)
	fig = plt.figure()
	ax = fig.add_subplot(111, aspect='equal')
	ax.imshow(img, cmap=plt.cm.gray)
	ax.set_axis_off()

	"""
    print(len(inter))
    hull_2d = ConvexHull(inter)
    for s in hull_2d.simplices:
        s = np.append(s, s[0])
        plt.plot(inter[s, 0], inter[s, 1], 'b-')
    """
	# plt.scatter(points_c[:, 0], points_c[:, 1], s=5, zorder=1)
	# ax.plot(inter[:, 0], inter[:, 1], '--b', zorder=1)
	# ax.plot(points_a[:, 1], points_a[:, 0], '--b', lw=3)
	# ax.plot(points_c[:, 0], points_c[:, 1], '--b', lw=3)
	ax.plot(snake_thre[:, 1], snake_thre[:, 0], '-r', lw=2)
	#ax.plot(snake[:, 1], snake[:, 0], '--r', lw=1)
	plt.savefig(file + '/threshold/' + str(z + 1) + '.jpg', bbox_inches='tight', dpi=fig.dpi, pad_inches=0.0)
	# plt.show()


def picture(z):
	img = cv2.imread(file + '/process_1/' + str(z + 1) + '.jpg')
	plt.cla()
	fig = plt.figure()
	ax = fig.add_subplot(111, aspect='equal')
	ax.imshow(img, cmap=plt.cm.gray)
	ax.set_axis_off()
	# plt.xlabel('x')
	# plt.ylabel('y')

	# plt.xlim(0, 512)
	# plt.ylim(0, 512)
	plt.savefig(file + '/threshold/' + str(z + 1) + '.jpg', bbox_inches='tight', dpi=fig.dpi, pad_inches=0.0)
	# plt.show()


#只保留血栓块
def threshold(z):
	image = cv2.imread("G:/dlc/test/test-eight/video_picture_10/process_1/" + str(z + 1) + ".jpg")
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	ret1, th1 = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	closed = cv2.morphologyEx(th1, cv2.MORPH_CLOSE, kernel, iterations=3)
	opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=3)
	# plt.xticks([])
	# plt.yticks([])
	plt.cla()
	fig = plt.figure()
	ax = fig.add_subplot(111, aspect='equal')
	ax.imshow(opened, cmap=plt.cm.gray)
	ax.set_axis_off()
	plt.savefig(file + '/threshold/th' + str(z + 1) + '.jpg', bbox_inches='tight', dpi=fig.dpi, pad_inches=0.0)
	return opened
	# plt.show()

def main():
    for z in range(0, 55):
        print(z)
        judge_intersect(z)


if __name__ == "__main__":
    main()