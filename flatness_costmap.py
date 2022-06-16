import cv2
import numpy as np
import math

def flatness_detection(img_path):

    img_gray = cv2.imread(img_path)

    flatness = np.zeros_like(img_gray)
    img_edges = cv2.Canny(img_gray, 1, 250)

    non_zero=[]
    for i in range(len(img_edges)):
        for j in range(len(img_edges[i])):

            if img_edges[i][j]!=0:
                non_zero.append((i,j))

    for m in range(np.shape(img_gray)[0]):
        for n in range(np.shape(img_gray)[1]):
            flat=[]
            for tar in non_zero:
                flat.append(math.sqrt(math.pow((m-tar[0]),2) + math.pow((n-tar[1]),2)))
                # print(flat)
            flatness[m][n]=min(flat)

    heatmapshow = None
    heatmapshow = cv2.normalize(flatness, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
    return heatmapshow