
from steepness_costmap import *
from flatness_costmap import *






if __name__ == '__main__':
    img_path = 'test_2.JPG'
    flatness_heat = flatness_detection(img_path)
    steepness_heat = steepness_detection(img_path)

    final_costmap = flatness_heat + steepness_heat
    # retval = cv2.imwrite('/home/linfeng/PycharmProjects/sitedetection/heatmap_3.jpg', heatmapshow)
    # cv2.imshow("Heatmap", heatmapshow)
    # cv2.waitKey(50)
    cv2.imwrite('final_cost.jpg', final_costmap)
    # cv2.imshow('final',final_costmap)
    # cv2.waitKey(0)