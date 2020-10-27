import numpy as np
import skimage.filters as filt


def create_A(a, b, N):
    row = np.r_[
        -2*a - 6*b,
        a + 4*b,
        -b,
        np.zeros(N-5),
        -b,
        a + 4*b
    ]
    A = np.zeros((N, N))
    for i in range(N):
        A[i] = np.roll(row, i)
    return A


def create_external_edge_force_gradients_from_img(img, sigma=30. ):
    smoothed = filt.gaussian((img-img.min()) / (img.max()-img.min()), sigma)
    giy, gix = np.gradient(smoothed)
    gmi = (gix**2 + giy**2)**(0.5)
    gmi = (gmi - gmi.min()) / (gmi.max() - gmi.min())

    ggmiy, ggmix = np.gradient(gmi)

    def fx(x, y):
        x[x < 0] = 0.
        y[y < 0] = 0.

        x[x > img.shape[1] - 1] = img.shape[1] - 1
        y[y > img.shape[0] - 1] = img.shape[0] - 1

        return ggmix[(y.round().astype(int), x.round().astype(int))]

    def fy(x, y):
        x[x < 0] = 0.
        y[y < 0] = 0.

        x[x > img.shape[1] - 1] = img.shape[1] - 1
        y[y > img.shape[0] - 1] = img.shape[0] - 1
        return ggmiy[(y.round().astype(int), x.round().astype(int))]

    return fx, fy


def iterate_snake(x, y, a, b, fx, fy, gamma=0.1, n_iters=10, return_all=True):
    A = create_A(a, b, x.shape[0])
    B = np.linalg.inv(np.eye(x.shape[0]) - gamma*A)
    if return_all:
        snakes = []
    print(x.shape)

    for i in range(n_iters):
        x_ = np.dot(B, x + gamma*fx(x, y))
        y_ = np.dot(B, y + gamma*fy(x, y))
        x, y = x_.copy(), y_.copy()
        if return_all:
            snakes.append((x_.copy(), y_.copy()))

    if return_all:
        return snakes
    else:
        return (x,y)