import numpy as np
from scipy import ndimage

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def plot_qolo(position, velocity, ax):
    QOLO_LENGHT_X = 0.4
    qolo_image = mpimg.imread("media/Qolo_T_CB_top_bumper_low_qual.png")

    qolo_x_length = QOLO_LENGHT_X
    qolo_y_length = float(qolo_image.shape[0]) / qolo_image.shape[1] * qolo_x_length
    angle_rot = np.arctan2(velocity[1], velocity[0])

    try:
        qolo_rot = ndimage.rotate(qolo_image, angle_rot * 180.0 / np.pi, cval=255)
    except:
        breakpoint()
    lenght_x_rotated = (
        np.abs(np.cos(angle_rot)) * qolo_x_length
        + np.abs(np.sin(angle_rot)) * qolo_y_length
    )
    lenght_y_rotated = (
        np.abs(np.sin(angle_rot)) * qolo_x_length
        + np.abs(np.cos(angle_rot)) * qolo_y_length
    )

    ax.imshow(
        (qolo_rot * 255).astype("uint8"),
        extent=[
            position[0] - lenght_x_rotated / 2,
            position[0] + lenght_x_rotated / 2,
            position[1] - lenght_y_rotated / 2,
            position[1] + lenght_y_rotated / 2,
        ],
        zorder=3,
    )
