import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import skimage.io as io
import skimage.filters as flt
import scipy.ndimage.filters as flt
import warnings


def anisodiff(img, niter=1, kappa=50, gamma=0.1, step=(1.,1.), sigma=0, option=1, ploton=False):
	if img.ndim == 3:
		warnings.warn("Only grayscale images allowed, converting to 2D matrix")
		img = img.mean(2)

	img = img.astype('float32')
	imgout = img.copy()

	deltaS = np.zeros_like(imgout)
	deltaE = deltaS.copy()
	NS = deltaS.copy()
	EW = deltaS.copy()
	gS = np.ones_like(imgout)
	gE = gS.copy()

	if ploton:
		import pylab as pl
		from time import sleep

		fig = pl.figure(figsize=(20, 5, 5), num="Anisotropic diffusion")
		ax1, ax2 = fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)

		ax1.imshow(img, interpolation='nearest')
		ih = ax2.imshow(imgout, interpolation='nearest', animated=True)
		ax1.set_title("Original image")
		ax2.set_title("Iteration 0")

		fig.canvas.draw()

	for ii in np.arange(1, niter):
		deltaS[:-1, : ] = np.diff(imgout, axis=0)
		deltaE[: , :-1] = np.diff(imgout, axis=1)

		if 0<sigma:
			deltaSf = flt.gaussian_filter(deltaS, sigma);
			deltaEf = flt.gaussian_filter(deltaE, sigma);
		else:
			deltaSf=deltaS;
			deltaEf=deltaE;

		if option == 1:
			gS = np.exp(-(deltaSf/kappa)**2.)/step[0]
			gE = np.exp(-(deltaEf/kappa)**2.)/step[1]
		elif option == 2:
			gS = 1./(1.+(deltaSf/kappa)**2.)/step[0]
			gE = 1./(1.+(deltaEf/kappa)**2.)/step[1]

		E = gE*deltaE
		S = gS*deltaS

		NS[:] = S
		EW[:] = E
		NS[1:, :] -= S[:-1, :]
		EW[:, 1:] -= E[:, :-1]

		imgout += gamma*(NS+EW)

		if ploton:
			iterstring = "Iteration %1" %(ii+1)
			ih.set_data(imgout)
			ax2.set_title(iterstring)
			fig.canvas.draw()
	return imgout


def anisodiff3(stack,niter=1,kappa=50,gamma=0.1,step=(1.,1.,1.),option=1,ploton=False):
	if stack.ndim == 4:
		warnings.warn("Only grayscale stacks allowed, converting to 3D matrix")
		stack = stack.mean(3)

	# initialize output array
	stack = stack.astype('float32')
	stackout = stack.copy()

	# initialize some internal variables
	deltaS = np.zeros_like(stackout)
	deltaE = deltaS.copy()
	deltaD = deltaS.copy()
	NS = deltaS.copy()
	EW = deltaS.copy()
	UD = deltaS.copy()
	gS = np.ones_like(stackout)
	gE = gS.copy()
	gD = gS.copy()

	if ploton:
		import pylab as pl
		from time import sleep

		showplane = stack.shape[0] // 2

		fig = pl.figure(figsize=(20, 5.5), num="Anisotropic diffusion")
		ax1, ax2 = fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)

		ax1.imshow(stack[showplane, ...].squeeze(), interpolation='nearest')
		ih = ax2.imshow(stackout[showplane, ...].squeeze(), interpolation='nearest', animated=True)
		ax1.set_title("Original stack (Z = %i)" % showplane)
		ax2.set_title("Iteration 0")

		fig.canvas.draw()

	for ii in np.arange(1, niter):

		# calculate the diffs
		deltaD[:-1, :, :] = np.diff(stackout, axis=0)
		deltaS[:, :-1, :] = np.diff(stackout, axis=1)
		deltaE[:, :, :-1] = np.diff(stackout, axis=2)

		# conduction gradients (only need to compute one per dim!)
		if option == 1:
			gD = np.exp(-(deltaD / kappa) ** 2.) / step[0]
			gS = np.exp(-(deltaS / kappa) ** 2.) / step[1]
			gE = np.exp(-(deltaE / kappa) ** 2.) / step[2]
		elif option == 2:
			gD = 1. / (1. + (deltaD / kappa) ** 2.) / step[0]
			gS = 1. / (1. + (deltaS / kappa) ** 2.) / step[1]
			gE = 1. / (1. + (deltaE / kappa) ** 2.) / step[2]

		# update matrices
		D = gD * deltaD
		E = gE * deltaE
		S = gS * deltaS

		UD[:] = D
		NS[:] = S
		EW[:] = E
		UD[1:, :, :] -= D[:-1, :, :]
		NS[:, 1:, :] -= S[:, :-1, :]
		EW[:, :, 1:] -= E[:, :, :-1]

		stackout += gamma * (UD + NS + EW)

		if ploton:
			iterstring = "Iteration %i" % (ii + 1)
			ih.set_data(stackout[showplane, ...].squeeze())
			ax2.set_title(iterstring)
			fig.canvas.draw()


	return stackout


img=io.imread('G:/dlc/test/test-eight/video_picture_4/process/change_color_1/121.jpg')
img=img.astype(float)
img=img[300:600,300:600]
m=np.mean(img)
s=np.std(img)
nimg=(img-m)/s


plt.figure(figsize=(16,9))
fimg=anisodiff(nimg,100,80,0.075,(1,1),2.5,1)
plt.subplot(2,3,1)
plt.imshow(nimg)
plt.title('Original')
plt.subplot(2,3,2)
plt.imshow(fimg,vmin=-1,vmax=1)
#plt.imshow(fimg)
plt.title('Filtered')
plt.subplot(2,3,3)
plt.imshow(fimg-nimg)
plt.title('Difference')
plt.subplot(2,3,4)
h=np.histogram(nimg,100)
plt.plot(h[0])

plt.subplot(2,3,5)
h,ax=np.histogram(fimg,100)

plt.plot(ax[0:(np.size(h))],h)
