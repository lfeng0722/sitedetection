import cv2
import numpy as np


def steepness_detection(img_path):
    d_im = cv2.imread(img_path)
    d_im = cv2.cvtColor(d_im, cv2.COLOR_BGR2GRAY)
    d_im = d_im.astype("float64")


    zx = cv2.Sobel(d_im, cv2.CV_64F, 1, 0, ksize=5)
    zy = cv2.Sobel(d_im, cv2.CV_64F, 0, 1, ksize=5)

    normal = np.dstack((-zx, -zy, np.ones_like(d_im)))
    n = np.linalg.norm(normal, axis=2)
    normal[:, :, 0] /= n
    normal[:, :, 1] /= n
    normal[:, :, 2] /= n
    #
    # # offset and rescale values to be in 0-255
    normal += 1
    normal /= 2


    heatmapshow = None
    heatmapshow = cv2.normalize(normal[:, :, ::-1], heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                dtype=cv2.CV_8U)
    heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
    return heatmapshow