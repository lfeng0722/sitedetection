import cv2
import numpy as np

d_im = cv2.imread("image_123986672.JPG")
d_im = cv2.cvtColor(d_im, cv2.COLOR_BGR2GRAY)
d_im = d_im.astype("float64")

# z = np.gradient(d_im)
# print(np.shape(z))
# You may also consider using Sobel to get a joint Gaussian smoothing and differentation
# to reduce noise
zx = cv2.Sobel(d_im, cv2.CV_64F, 1, 0, ksize=5)
zy = cv2.Sobel(d_im, cv2.CV_64F, 0, 1, ksize=5)
# print(np.shape(zy))
normal = np.dstack((-zx, -zy, np.ones_like(d_im)))
n = np.linalg.norm(normal, axis=2)
normal[:, :, 0] /= n
normal[:, :, 1] /= n
normal[:, :, 2] /= n
#
# # offset and rescale values to be in 0-255
normal += 1
normal /= 2
# normal *= 255
# print(normal)
# cv2.imwrite("normal.png", normal[:, :, ::-1])
# normals = np.array(d_im, dtype="float32")
# h,w,d = d_im.shape
# for i in range(1,w-1):
#   for j in range(1,h-1):
#     t = np.array([i,j-1,d_im[j-1,i,0]],dtype="float64")
#     f = np.array([i-1,j,d_im[j,i-1,0]],dtype="float64")
#     c = np.array([i,j,d_im[j,i,0]] , dtype = "float64")
#     d = np.cross(f-c,t-c)
#     n = d / np.sqrt((np.sum(d**2)))
#     normals[j,i,:] = n
# print(normals)
# gray = cv2.cvtColor(normals*255, cv2.COLOR_BGR2GRAY)

heatmapshow = None
heatmapshow = cv2.normalize(normal[:, :, ::-1], heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#
heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
# print(normal[:, :, ::-1])
# cv2.imshow('123',heatmapshow)
# cv2.waitKey(0)
#
# retval = cv2.imwrite('/home/linfeng/PycharmProjects/sitedetection/normals.jpg', heatmapshow)
cv2.imwrite("steepness_2.jpg",heatmapshow)