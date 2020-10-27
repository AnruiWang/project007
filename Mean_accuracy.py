import numpy as np

file = "G:/dlc/test/test-eight/video_picture_10"

def mean_accuracy(x, file_txt):
	sum = 0.0
	b = np.array([0], dtype=float)
	for i in range(0, len(x)):
		if (abs(x[i]) > 1e-2):
			c = np.array([x[i]], dtype=float)
			b = np.append(b, c)
			sum = sum + x[i]

	mean = sum / (len(b) - 1)

	f = open(file_txt, 'a')
	f.write("平均精度为：" + str(mean) + '\n')

k = np.loadtxt(file + "/precision.txt")
k_txt = file + "/precision.txt"
m = np.loadtxt(file + "/precision_mor.txt")
m_txt = file + "/precision_mor.txt"
n = np.loadtxt(file + "/precision_GVF.txt")
n_txt = file + "/precision_GVF.txt"

mean_accuracy(k, k_txt)
mean_accuracy(m, m_txt)
mean_accuracy(n, n_txt)
