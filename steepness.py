import open3d as o3d
import numpy as np
import math
import cv2
if __name__ == "__main__":
    # bunny = o3d.data.BunnyMesh()
    dataset = o3d.data.PLYPointCloud()
    pcd = o3d.io.read_point_cloud(dataset.path)

    # dataset = o3d.data.PCDPointCloud()
    # pcd = o3d.io.read_point_cloud(dataset.path)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius = 0.1, max_nn = 30))
    print(np.shape(pcd))
    # steepness =[]
    # for pixle in np.asarray(pcd.normals):
    #     x = math.sqrt(np.square(pixle[0] )+ np.square(pixle[1] ))
    #     theta = np.arctan(x/pixle[2])
    #     steepness.append(theta)
    #     # print(theta)
    # # print(pcd.normals[0])
    # steepness.reshape()
    # gt_mesh.compute_vertex_normals()
    #
    # pcd = gt_mesh.sample_points_poisson_disk(5000)
    # # Invalidate existing normals.
    # pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))
    #
    # print("Displaying input pointcloud ...")
    # o3d.visualization.draw_geometries([pcd], point_show_normal=True)
    #
    # pcd.estimate_normals()
    print("Displaying pointcloud with normals ...")
    o3d.visualization.draw_geometries([pcd], point_show_normal=True)
    #
    print("Printing the normal vectors ...")
    print(np.asarray(pcd.normals))

# if __name__ == "__main__":
#     bunny = o3d.data.DemoPoseGraphOptimization()
#     gt_mesh = o3d.io.read_triangle_mesh(bunny.path)
#     gt_mesh.compute_vertex_normals()
#
#     pcd = gt_mesh.sample_points_poisson_disk(5000)
#     # Invalidate existing normals.
#     pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))
#
#     print("Displaying input pointcloud ...")
#     o3d.visualization.draw_geometries([pcd], point_show_normal=True)
#
#     pcd.estimate_normals()
#     print("Displaying pointcloud with normals ...")
#     o3d.visualization.draw_geometries([pcd], point_show_normal=True)
#
#     print("Printing the normal vectors ...")
#     print(np.asarray(pcd.normals))