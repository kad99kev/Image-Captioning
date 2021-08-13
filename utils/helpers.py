import numpy as np
import matplotlib.pyplot as plt


def show_image(img, cap=None) -> None:
    """
    Displays an image. (To be used in a notebook environment).

    Parameters:
        img: Image to be displayed.
        cap: Caption of the image.
    """
    if cap:
        plt.title(cap)
    plt.imshow(img)


def convert_numpy(img) -> np.ndarray:
    """
    Converts a PyTorch image tensor into a `np.ndarray`.

    Parameters:
        img: The image to be converted into numpy.

    Returns:
        The image in plt form.
    """
    disp_img = np.squeeze(img.numpy())
    disp_img = (disp_img - np.min(disp_img)) / (np.max(disp_img) - np.min(disp_img))
    disp_img = np.transpose(disp_img, (1, 2, 0))
    return disp_img