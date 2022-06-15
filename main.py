import math
import tqdm
import cv2
import numpy as np
from PIL import Image
from scipy.spatial.distance import pdist



# def euclidean(image1, image2):
#     '''欧氏距离'''
#     X = np.vstack([image1, image2])
#     return pdist(X, 'euclidean')[0]



def detect_edges(img_path):

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
    return flatness
    # print(img_edges.shape)

    # euclidean = []
    # for i in range(len(img_edges)):
    #    dist = abs(img_gray[i]-img_edges[i])
    #    if dist < 1:
    #        euclidean.append(dist)
    #    else:
    #        euclidean.append(1)
    # euclidean = np.array(euclidean).reshape(347,610)
    # return euclidean

    # cv2.imshow('original',img_gray)





if __name__ == '__main__':
    img_path = 'test_2.JPG'
    img = detect_edges(img_path)

    heatmapshow = None
    heatmapshow = cv2.normalize(img, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)

    retval = cv2.imwrite('/home/linfeng/PycharmProjects/sitedetection/heatmap_3.jpg', heatmapshow)
    # cv2.imshow("Heatmap", heatmapshow)
    # cv2.waitKey(50)

    # cv2.imshow('original',img)
    # cv2.waitKey(0)