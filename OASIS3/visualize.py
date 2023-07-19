import matplotlib.pyplot as plt
import numpy as np


def save_image(volume, outfile, size=128):
    # Extract the central three slices along each axis
    x_slices = volume[size//2, :, :]
    y_slices = volume[:, size//2, :]
    z_slices = volume[:, :, size//2]

    # Create the figure and subplots
    fig, axes = plt.subplots(1, 3)
    axes[0].imshow(x_slices, cmap='gray')
    axes[0].axis('off')
    axes[0].set_title('X-axis')

    axes[1].imshow(y_slices, cmap='gray')
    axes[1].axis('off')
    axes[1].set_title('Y-axis')

    axes[2].imshow(z_slices, cmap='gray')
    axes[2].axis('off')
    axes[2].set_title('Z-axis')

    # Adjust spacing between subplots
    plt.tight_layout()

    # Save the figure as an image file
    plt.savefig(outfile)


if __name__ == "__main__":
    x = np.load('x.npy')[0]
    y = np.load('y.npy')[0]

    print(x.shape, y.shape)
    save_image(x, 'x.png')
    save_image(y, 'y.png')

    x = np.load('x_orig.npy')
    y = np.load('y_orig.npy')

    print(x.shape, y.shape)
    save_image(x, 'x_orig.png', size=256)
    save_image(y, 'y_orig.png', size=256)