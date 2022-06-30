
from steepness_costmap import *
from flatness_costmap import *

import numpy as np
from sklearn.cluster import KMeans
from collections import Counter




if __name__ == '__main__':
    img_path = 'test_2.JPG'
    img = cv2.imread(img_path)


    flatness_heat = flatness_detection(img_path)
    steepness_heat = steepness_detection(img_path)

    final_costmap = flatness_heat + steepness_heat
    # retval = cv2.imwrite('/home/linfeng/PycharmProjects/sitedetection/heatmap_3.jpg', heatmapshow)
    # cv2.imshow("Heatmap", heatmapshow)
    # cv2.waitKey(50)
    # cv2.imwrite('final_cost.jpg', final_costmap)
    # print(final_costmap)
    gray = cv2.cvtColor(final_costmap, cv2.COLOR_BGR2GRAY)
    thresholding = np.argwhere(gray>210)
    # print(len(thresholding))
    kmeans = KMeans(n_clusters=10).fit(thresholding)
    # print(len(kmeans.labels_))
    # print(kmeans.cluster_centers_)
    cluster = []
    for i in range(len(kmeans.cluster_centers_)):
        # print("Cluster", i)
        # print("Center:", kmeans.cluster_centers_[i])
        # print("Size:", sum(kmeans.labels_ == i))
        cluster.append(sum(kmeans.labels_ == i))
    # for coord in kmeans.cluster_centers_:
    #     coord = coord.astype(int)
    #     # print(coord)
    print(kmeans.cluster_centers_[np.argmax(cluster)],)
    cv2.drawMarker(gray, position=kmeans.cluster_centers_[np.argmax(cluster)].astype(int), color=(255, 0, 0), markerSize=50, markerType=cv2.MARKER_CROSS,
                       thickness=5)
    cv2.imshow('final',gray)
    cv2.waitKey(0)