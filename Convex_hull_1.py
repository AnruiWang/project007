#Correct function
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull

points = np.loadtxt('./data/hull.txt')

def tetrahedron_volume(a, b, c, d):
    return np.abs(np.einsum('ij,ij->i', a - d, np.cross(b - d, c - d))) / 6

def convex_hull_volume_bis(pts):
    ch = ConvexHull(pts)
    simplices = np.column_stack((np.repeat(ch.vertices[0], ch.nsimplex),
                                 ch.simplices))
    tets = ch.points[simplices]
    return np.sum(tetrahedron_volume(tets[:, 0], tets[:, 1],
                                     tets[:, 2], tets[:, 3]))


print(convex_hull_volume_bis(points))

hull = ConvexHull(points)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#Plot defining corner points
#ax.plot(points.T[0], points.T[1], points.T[2], "bo", markersize=2)

#print(hull.simplices)
#print(hull.neighbors)

for s in hull.simplices:
    #print(points[s[0]][0])
    s = np.append(s, s[0])
    ax.plot(points[s, 0], points[s, 1], points[s, 2], 'r-', markersize=0.1)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax.set_xlim3d(0, 512)
ax.set_ylim3d(0, 512)
ax.set_zlim3d(0, 109)

ax.grid(False)

plt.show()