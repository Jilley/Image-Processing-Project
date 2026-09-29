import numpy as np
import matplotlib.pyplot as plt
import tifffile
from scipy.fft import dctn, idctn

def pad_to_multiple(img, block_size):
    x_length, y_length = img.shape
    pad_x = (-x_length) % block_size
    pad_y = (-y_length) % block_size
    return np.pad(img, ((0, pad_x), (0, pad_y)), mode="edge")

def compress_block(img, k=8):
    X = dctn(img, type=2, norm="ortho")

    flat = np.abs(X).ravel()
    idx = np.argpartition(flat, -k)[-k:]
    k1, k2 = np.unravel_index(idx, X.shape)

    X_sparse = np.zeros_like(X)
    X_sparse[k1, k2] = X[k1, k2]
    img_approx = idctn(X_sparse, type=2, norm="ortho")

    return img_approx

def compress_image(file_path, block_size=8, k=8, plotting=False):
    img = tifffile.imread(file_path)
    x_length, y_length = img.shape

    img_padded = pad_to_multiple(img, block_size)
    x_padded, y_padded = img_padded.shape

    x_blocks = x_padded // block_size
    y_blocks = y_padded // block_size

    blocks = img_padded.reshape(x_blocks, block_size, y_blocks, block_size).swapaxes(1, 2)

    approx_blocks = np.zeros(blocks.shape, dtype=np.float64)
    for i in range(x_blocks):
        for j in range(y_blocks):
            approx_blocks[i, j] = compress_block(blocks[i, j], k=k)

    img_approx_padded = approx_blocks.swapaxes(1, 2).reshape(x_padded, y_padded)
    img_approx = img_approx_padded[:x_length, :y_length]   # crop back to original size

    img_approx_clipped = np.clip(img_approx, 0, 255)

    if plotting:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(img, cmap="gray", vmin=0, vmax=255)
        ax[0].set_title("Original")
        ax[1].imshow(img_approx_clipped, cmap="gray", vmin=0, vmax=255)
        ax[1].set_title(f"Top {k} of {block_size*block_size} coeffs per block")
        for a in ax:
            a.axis("off")
        plt.tight_layout()
        plt.show()

    return img_approx_clipped

compress_image("bilder/ngc4024l.tif", plotting=True)