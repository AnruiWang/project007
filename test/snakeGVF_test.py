import numpy as np
import matplotlib.pyplot as plt
import test.snake as sn
import test.GVF_test as GVF
import sys
import cv2
import skimage.color
import skimage.transform
try:
	import skimage.filter as skimage_filter
except:
	import skimage.filters as skimage_filter


def gradient_vector_flow(fx, fy, mu, dx=1.0, dy=1.0, verbose=True):
	u'''calc gradient vector flow of input gradient field fx, fy.'''
	b = fx**2.0 + fy**2.0
	c1, c2 = b*fx, b*fy
	#calc dt from scaling parameter r.
	r = 0.25 # (17) r < 1/4 required for convergence.
	dt = dx*dy/(r*mu)
	# max iteration
	N = int(max(1, np.sqrt(img.shape[0]*img.shape[1])))
	# initialize u(x, y), v(x, y) by the input.
	curr_u = fx
	curr_v = fy
	def laplacian(m):
		return np.hstack([m[:, 0:1], m[:, :-1]]) + np.hstack([m[:, 1:], m[:, -2:-1]]) \
			+ np.vstack([m[0:1, :], m[:-1, :]]) + np.vstack([m[1:, :], m[-2:-1, :]]) \
			- 4 * m
	for i in range(N):
		next_u = (1.0 - b*dt)*curr_u + r*laplacian(curr_u) + c1*dt
		next_v = (1.0 - b*dt)*curr_v + r*laplacian(curr_v) + c2*dt
		curr_u, curr_v = next_u, next_v
		if verbose:
			sys.stdout.write('.')
			sys.stdout.flush()
	if verbose:
		sys.stdout.write('\n')
	return curr_u, curr_v


def edge_map(img, sigma):
	blur = skimage_filter.gaussian(img, sigma)
	return skimage_filter.sobel(blur)


def gradient_field(im):
	im = skimage_filter.gaussian(im, 1.0)
	gradx = np.hstack([im[:, 1:], im[:, -2:-1]]) - np.hstack([im[:, 0:1], im[:, :-1]])
	grady = np.vstack([im[1:, :], im[-2:-1, :]]) - np.vstack([im[0:1, :], im[:-1, :]])
	return gradx, grady


def add_border(img, width):
	h, w = img.shape
	val = img[:, 0].mean() + img[:, -1].mean() + img[0, :].mean() + img[-1, :].mean()
	res = np.zeros((h + width*2, w + width*2), dtype=img.dtype) + val
	res[width:h + width, width:w + width] = img
	res[:width, :] = res[width, :][np.newaxis, :]
	res[:, :width] = res[:, width][:, np.newaxis]
	res[h + width:, :] = res[h + width - 1, :][np.newaxis, :]
	res[:, w + width:] = res[:, w + width - 1][:, np.newaxis]
	return res

"""
#load image and preprocess
if len(sys.argv) > 1:
	fn = sys.argv[1]
	img = skimage.color.rgb2gray('G:/dlc/test/test-eight/video_picture_8_1/process_1/59.jpg')
else:
	img = cv2.imread('G:/dlc/test/test-eight/video_picture_8_1/process_1/59.jpg')
	img = skimage.color.rgb2gray(img)
	img = skimage.transform.resize(img, (512, 512))
	img = img.astype(np.float32) / 255.0
	img = add_border(img, 32)
	edge = edge_map(img, sigma=2)

#calc gvf
gx, gy = gradient_field(edge)
gx, gy = gradient_vector_flow(gx, gy, mu=1.0)
"""
img = cv2.imread('G:/dlc/test/test-eight/video_picture_8_1/process_1/59.jpg')
img = skimage.color.rgb2gray(img)
t = np.arange(0, 2*np.pi, 0.1)
x = 256+100*np.cos(t)
y = 256+100*np.sin(t)




alpha = 0.001
beta  = 0.4
gamma = 100
iterations = 50


# fx and fy are callable functions
fx, fy = sn.create_external_edge_force_gradients_from_img( img, sigma=10 )
print("fx", fx(x, y).shape)

snakes = sn.iterate_snake(
    x = x,
    y = y,
    a = alpha,
    b = beta,
    fx = fx,
    fy = fy,
    gamma = gamma,
    n_iters = iterations,
    return_all = True
)

fig = plt.figure()
ax  = fig.add_subplot(111)
ax.imshow(img, cmap=plt.cm.gray)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(0,img.shape[1])
ax.set_ylim(img.shape[0],0)
ax.plot(np.r_[x,x[0]], np.r_[y,y[0]], c=(0,1,0), lw=2)

for i, snake in enumerate(snakes):
    if i % 10 == 0:
        ax.plot(np.r_[snake[0], snake[0][0]], np.r_[snake[1], snake[1][0]], c=(0,0,1), lw=2)

# Plot the last one a different color.
ax.plot(np.r_[snakes[-1][0], snakes[-1][0][0]], np.r_[snakes[-1][1], snakes[-1][1][0]], c=(1,0,0), lw=2)

plt.show()