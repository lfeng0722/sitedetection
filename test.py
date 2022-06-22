import numpy as np
import cv2
import math

img_gray = cv2.imread('test_2.JPG')

flatness = np.zeros_like(img_gray)
img_edges = cv2.Canny(img_gray, 1, 250)

x , y = np.nonzero(img_edges)

for m in range(np.shape(img_gray)[0]):
    for n in range(np.shape(img_gray)[1]):
        flat = []
        for i in range(len(x)):
            flat.append(math.sqrt(math.pow((m - x[i]), 2) + math.pow((n - y[i]), 2)))
            # print(flat)
        flatness[m][n] = min(flat)

heatmapshow = None
heatmapshow = cv2.normalize(flatness, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
cv2.imshow('final',heatmapshow)
cv2.waitKey(0)