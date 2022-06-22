import cv2
import numpy as np
import math

def flatness_detection(img_path):

    img_gray = cv2.imread(img_path)

    flatness = np.zeros_like(img_gray)
    img_edges = cv2.Canny(img_gray, 1, 250)

    non_zero=[]
    non_zero_2 = np.argwhere(img_edges != 0)

    for i in range(len(img_edges)):
        for j in range(len(img_edges[i])):

            if img_edges[i][j]!=0:
                non_zero.append((i,j))

    ixs = np.ndindex(img_gray.shape[0],img_gray.shape[1])
    for ix in ixs:
        # print(ix)
        flatness[ix[0]][ix[1]] = np.linalg.norm(np.array(ix)-non_zero_2, axis=1).min()


    heatmapshow = None
    heatmapshow = cv2.normalize(flatness, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
    return heatmapshow