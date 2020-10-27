import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN

data = np.loadtxt('../data/test.txt')
len_z = 180

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter3D(data[:, 0], data[:, 1], data[:, 2])
ax.view_init(azim=200)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_xlim(0, 512)
ax.set_ylim(0, 512)
ax.set_zlim(0, len_z)
plt.show()

model = DBSCAN(eps=20, min_samples=5)
model.fit_predict(data)
pred = model.fit_predict(data)

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter3D(data[:, 0], data[:, 1], data[:, 2], c=model.labels_)
ax.view_init(azim=200)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_xlim(0, 512)
ax.set_ylim(0, 512)
ax.set_zlim(0, len_z)
ax.grid(False)
# plt.savefig('C:/Users/wangx/Desktop/paper/素材/DBSCAN19.jpg')
plt.show()

print("number of cluster found: {}".format(len(set(model.labels_))))
print('cluster for each point: ', model.labels_)

np.savetxt('../data/result.txt', model.labels_)

#with open("./data/result.txt", "wb") as f:
#    for i in range(model.labels_):
#        f.write(i)