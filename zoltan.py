import numpy as np
import matplotlib.pyplot as plt
import tifffile
from scipy.fft import dctn, idctn

#X = dctn(img, type=2, norm="ortho")      # orthonormal 2D DCT-II
#img_rec = idctn(X, type=2, norm="ortho") # inverse

img = tifffile.imread("bilder/rice.tif")   # np.ndarray, shape (H, W)

x_length = img.shape[0]
y_length = img.shape[1]

block_size = 8
x_blocks = int(x_length / block_size)
y_blocks = int(y_length / block_size)

blocks = img.reshape(x_blocks, block_size, y_blocks, block_size).swapaxes(1, 2)
first_block = blocks[0, 0]

def compress_block(img):
    X = dctn(img, type=2, norm="ortho")

    k = 8
    flat = np.abs(X).ravel()
    idx = np.argpartition(flat, -k)[-k:]          # unordered top-k, O(N)
    idx = idx[np.argsort(flat[idx])[::-1]]        # sort descending
    k1, k2 = np.unravel_index(idx, X.shape)       # frequency indices
    coeffs = X[k1, k2]                            # signed values

    X_sparse = np.zeros_like(X)
    X_sparse[k1, k2] = X[k1, k2]
    img_approx = idctn(X_sparse, type=2, norm="ortho")

    return img_approx

approx_blocks = np.zeros(blocks.shape, dtype=np.float64)
for i in range(x_blocks):
    for j in range(y_blocks):
        approx_blocks[i, j] = compress_block(blocks[i, j])

img_approx = approx_blocks.swapaxes(1, 2).reshape(x_length, y_length)

img_approx_clipped = np.clip(img_approx, 0, 255)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(img, cmap="gray", vmin=0, vmax=255)
ax[0].set_title("Original")
ax[1].imshow(img_approx_clipped, cmap="gray", vmin=0, vmax=255)
ax[1].set_title("Top 8 of 64 coeffs per block")
for a in ax:
    a.axis("off")
plt.tight_layout()
plt.show()


                    

