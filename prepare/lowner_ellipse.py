import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull


points = np.loadtxt('../data/hull.txt')


def sample_ellipse(ellipse, num_pts, endpoint=True):
	# extract ellipse parameters
	c, a, b, t = ellipse

	# rotation matrix
	rot_mat = np.array([[np.cos(t), -np.sin(t)],
						[np.sin(t), np.cos(t)]])

	# array of angles uniformly chosen between rotation and shift
	theta = np.linspace(0, 2 * np.pi, num_pts, endpoint=endpoint)

	# points on an ellipse with axis a, b before rotation and shift
	z = np.column_stack((a * np.cos(theta), b * np.sin(theta)))

	# rotation points by angle t and shift to center c
	x = rot_mat.dot(z.T).T + c

	return x


def plot_ellipse(ellipse, num_pts=100, str='-'):
	# if ellipse is empty, do nothing
	if ellipse is None:
		return

	# sample points on ellipse
	x = sample_ellipse(ellipse, num_pts)

	# plot ellipse
	plt.plot(x[:, 0], x[:, 1], str)


def center_from_to_geometric(F, c):
	# extract a, b, and t from F by finding eigenvalues and eigenvectors
	w, V = np.linalg.eigh(F)

	# the eigenvalues are 1/a**2 and 1/b**2
	# the eigenvectors form the rotation matrix with angle t

	# if one the eigenvalues is not positive, the ellipse is degenerate
	if w[0] <= 0 or w[1] <= 0:
		return None

	# we assume without loss of generality 0 < t < pi.
	# V[1, 0] = sin(t), therefore it must be non-negative:
	if V[1, 0] < 0:
		V[:, 0] = -V[:, 0]

	# find t
	t = np.arccos(V[0, 0]) # V[0, 0] = cos(t)

	print("c", c)
	print("major", 1 / np.sqrt(w[0]))
	print("minor", 1 / np.sqrt(w[1]))
	print("t", t)

	return c, 1 / np.sqrt(w[0]), 1 / np.sqrt(w[1]), t


def is_in_ellipse(point, ellipse):
	if ellipse is None:
		return False

	c, a, b, t = ellipse
	v = point - c
	rot_mat = np.array([[np.cos(t), np.sin(t)],
						[-np.sin(t), np.cos(t)]])
	F = rot_mat.T.dot(
		np.diag(1 / np.array([a, b], dtype=np.float) ** 2)).dot(rot_mat)

	return v.T.dot(F.dot(v)) <= 1


def ellipse_from_boundary5(S):
	print("5")
	x = S[:, 0]
	y = S[:, 1]
	A = np.column_stack((x ** 2, y ** 2, 2 * x * y, x, y))
	if np.linalg.cond(A) >= 1 / np.finfo(float).eps:
		return None

	sol = np.linalg.solve(A, -np.ones(S.shape[0]))
	c = np.linalg.solve(-2 * np.array([[sol[0], sol[2]],
									   [sol[2], sol[1]]]), sol[3:5])

	A = np.vstack([np.hstack([np.eye(3),
							  -np.array([[sol[0], sol[2], sol[1]]]).T]),
				   np.array([c[0] ** 2, 2 * c[0] * c[1], c[1] ** 2, -1])])
	s = np.linalg.solve(A, np.array([0, 0, 0, 1]))
	F = np.array([[s[0], s[1]], [s[1], s[2]]])

	return center_from_to_geometric(F, c)


def ellipse_from_boundary4(S):
	print("4")
	Sc = S - np.mean(S, axis=0)
	angles = np.arctan2(Sc[:, 1], Sc[:, 0])
	S = S[np.argsort(-angles), :]

	A = np.column_stack([S[2, :] - S[0, :], S[1, :] - S[3, :]])
	b = S[1, :] - S[0, :]
	s = np.linalg.solve(A, b)
	diag_intersect = S[0, :] + s[0] * (S[2, :] - S[0, :])

	S = S - diag_intersect

	AC = S[2, :] - S[0, :]
	theta = np.arctan2(AC[1], AC[0])
	rot_mat = np.array([[np.cos(theta), np.sin(theta)],
						[-np.sin(theta), np.cos(theta)]])
	S = rot_mat.dot(S.T).T

	m = (S[1, 0] - S[3, 0]) / (S[3, 1] - S[1, 1])
	shear_mat = np.array([[1, m], [0, 1]], dtype=np.float)
	S = shear_mat.dot(S.T).T

	b = np.linalg.norm(S, axis=1)
	if (b[2] * b[0]) == 0:
		after_b2 = b[2] + 1
		after_b0 = b[0] + 1
		d = b[1] * b[3] / (after_b2 * after_b0)
		print("d1:", d)
	else:
		if (b[1] * b[3]) == 0:
			after_b1 = b[1] + 1
			after_b3 = b[3] + 1
			d = after_b1 * after_b3 / (b[2] * b[0])
			print("d2:", d)
		else:
			d = b[1] * b[3] / (b[2] * b[0])
			print("d3", d)

	if d == 0:
		d = d + 1
		stretch_mat = np.diag(np.array([d ** .25, d ** -.25], dtype=np.float))
	else:
		stretch_mat = np.diag(np.array([d ** .25, d ** -.25], dtype=np.float))
	print(stretch_mat)
	S = stretch_mat.dot(S.T).T

	a = np.linalg.norm(S, axis=1)
	if a[0] == 0:
		a[0] += 1
	elif a[1] == 0:
		a[1] += 1
	elif a[2] == 0:
		a[2] += 1
	elif a[3] == 0:
		a[3] += 1
	print(a)
	coeff = np.zeros(4)
	coeff[0] = -4 * a[1] ** 2 * a[2] * a[0]
	coeff[1] = -4 * a[1] * (a[2] - a[0]) * (a[1] ** 2 - a[2] * a[0])
	coeff[2] = 3 * a[1] ** 2 * (a[1] ** 2 + a[2] ** 2)\
		- 8 * a[1] ** 2 * a[2] * a[0] + 3 * (a[1] ** 2 + a[2] **2) * a[0] ** 2
	coeff[3] = coeff[1] / 2.
	print(coeff)
	rts = np.roots(coeff)
	print(rts)
	rts = rts[(-1 < rts) & (rts < 1)]
	theta = np.arcsin(np.real(rts[0]))

	D_mat = np.array([[np.cos(theta) ** -.5,
					   np.sin(theta) * np.cos(theta) ** -.5],
					  [0, np.cos(theta) ** .5]])
	print("D_mat:", D_mat)
	S = D_mat.dot(S.T).T

	boundary = S[:-1, :]
	print("b:", boundary)
	A = np.vstack([-2 * boundary.T, np.ones(boundary.shape[0])]).T
	b = -np.sum(boundary ** 2, axis=1)
	print("A", A)
	print("b:", b)
	s = np.linalg.solve(A, b)

	circle_c = s[:2]
	circle_r = np.sqrt(np.sum(circle_c ** 2) - s[2])

	T_mat = D_mat.dot(stretch_mat).dot(shear_mat).dot(rot_mat)

	ellipse_c = np.linalg.solve(T_mat, circle_c) + diag_intersect
	ellipse_F = T_mat.T.dot(T_mat) / circle_r ** 2

	print("ec:", ellipse_c)
	print("ef:", ellipse_F)

	return center_from_to_geometric(ellipse_F, ellipse_c)


def ellipse_from_boundary3(S):
	print("3")
	c = np.mean(S, axis=0)
	Sc = S - c
	F = 1.5 * np.linalg.inv(Sc.T.dot(Sc))

	return center_from_to_geometric(F, c)


def welzl(interior, boundary=np.zeros((0, 2))):
	if interior.shape[0] == 0 or boundary.shape[0] >= 5:
		if boundary.shape[0] <= 2:
			return None

		elif boundary.shape[0] == 3:
			return ellipse_from_boundary3(boundary)
		elif boundary.shape[0] == 4:
			return ellipse_from_boundary4(boundary)
		else:
			return ellipse_from_boundary5(boundary)

	i = np.random.randint(interior.shape[0])
	p = interior[i, :]

	interior_wo_p = np.delete(interior, i, 0)

	ellipse = welzl(interior_wo_p, boundary)

	if is_in_ellipse(p, ellipse):
		return ellipse

	else:
		return welzl(interior_wo_p, np.vstack([boundary, p]))


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

"""
def main():
	plt.figure()

	points = np.random.randn(50, 2)
	print(points)
	plt.plot(points[:, 0], points[:, 1], '.')

	enclosing_ellipse = welzl(points)
	print("A",enclosing_ellipse)

	plot_ellipse(enclosing_ellipse, str='k--')
	plt.show()


"""
def main():
	for i in range(41, 42):
		inter = judge_intersect(i)
		print(inter)
		plt.figure()
		plt.plot(inter[:, 0], inter[:, 1], '.')
		plt.xlim(0, 512)
		plt.ylim(0, 512)

		enclosing_ellipse = welzl(inter)
		print(enclosing_ellipse)

		plot_ellipse(enclosing_ellipse, str='k--')
		plt.show()

main()