import time
import numpy as np
# from artelib.euler import Euler
# from artelib.homogeneousmatrix import HomogeneousMatrix
import open3d as o3d
import copy
# from config import ICP_PARAMETERS


class LiDARScan():
    def __init__(self, directory, scan_time, parameters):
        # directory
        self.directory = directory
        self.scan_time = scan_time
        # the pointcloud
        self.pointcloud = None  # o3d.io.read_point_cloud(filename)
        # self.pointcloud_filtered = None
        # voxel sizes
        self.voxel_size = parameters.get('voxel_size', None)
        self.voxel_size_normals = parameters.get('voxel_size_normals', None)
        self.max_nn_estimate_normals = parameters.get('max_nn_estimate_normals', None)
        # filter
        self.min_reflectivity = parameters.get('min_reflectivity', None)
        self.min_radius = parameters.get('min_radius', None)
        self.max_radius = parameters.get('max_radius', None)
        self.min_height = parameters.get('min_height', None)
        self.max_height = parameters.get('max_height', None)

        # a reduced/voxelized pointcloud
        # self.pointcloud_filtered = None
        # self.pointcloud_ground_plane = None
        # self.pointcloud_non_ground_plane = None
        # used for global FPFH registration
        # self.pointcloud_fpfh = None
        #
        # self.voxel_size_normals_ground_plane = 0.5
        # self.voxel_size_normals = 0.3
        # self.max_radius = ICP_PARAMETERS.max_radius
        # self.min_radius = ICP_PARAMETERS.min_radius
        # self.max_height = ICP_PARAMETERS.max_height
        # self.min_height = ICP_PARAMETERS.min_height
        # self.plane_model = None
        # self.pre_processed = False

        # self.voxel_size_fpfh = 5*self.voxel_size
        # self.icp_threshold = 5
        # self.fpfh_threshold = 5

        # read the lidar pcd
        # filename = directory + '/robot0/lidar/data/' + str(scan_time) + '.pcd'
        # the original complete pointcloud

    def load_pointcloud(self):
        filename = self.directory + '/robot0/lidar/data/' + str(self.scan_time) + '.pcd'
        print('Reading pointcloud: ', filename)
        # Load the original complete pointcloud
        self.pointcloud = o3d.io.read_point_cloud(filename)

    def save_pointcloud(self):
        filename = self.directory + '/robot0/lidar/dataply/' + str(self.scan_time) + '.ply'
        print('Saving pointcloud: ', filename)
        # Load the original complete pointcloud
        o3d.io.write_point_cloud(filename, self.pointcloud)

    def filter_points(self):
        self.down_sample()
        self.filter_radius()
        self.filter_height()

    def down_sample(self):
        if self.voxel_size is None:
            return
        self.pointcloud = self.pointcloud.voxel_down_sample(voxel_size=self.voxel_size)

    def filter_radius(self, radii=None):
        if radii is None:
            self.pointcloud = self.filter_by_radius(self.min_radius, self.max_radius)
        else:
            self.pointcloud = self.filter_by_radius(radii[0], radii[1])

    def filter_height(self, heights=None):
        if heights is None:
            self.pointcloud = self.filter_by_height(-120.0, 120.0)
        else:
            self.pointcloud = self.filter_by_height(heights[0], heights[1])

    def filter_by_radius(self, min_radius, max_radius):
        points = np.asarray(self.pointcloud.points)
        [x, y, z] = points[:, 0], points[:, 1], points[:, 2]
        r2 = x ** 2 + y ** 2
        # idx = np.where(r2 < max_radius ** 2) and np.where(r2 > min_radius ** 2)
        idx2 = np.where((r2 < max_radius ** 2) & (r2 > min_radius ** 2))
        return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[idx2]))

    def filter_by_height(self, min_height, max_height):
        points = np.asarray(self.pointcloud.points)
        [x, y, z] = points[:, 0], points[:, 1], points[:, 2]
        idx2 = np.where((z > min_height) & (z < max_height))
        return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[idx2]))

    def estimate_normals(self):
        self.pointcloud.estimate_normals(
             o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size_normals,
                                                  max_nn=self.max_nn_estimate_normals))

    def draw_cloud(self):
        # o3d.visualization.draw_geometries([self.pointcloud],
        #                                   zoom=0.3412,
        #                                   front=[0.4257, -0.2125, -0.8795],
        #                                   lookat=[2.6172, 2.0475, 1.532],
        #                                   up=[-0.0694, -0.9768, 0.2024])
        o3d.visualization.draw_geometries([self.pointcloud])

    def draw_cloud_visualizer(self):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(self.pointcloud)
        try:
            while True:
                if not vis.poll_events():
                    print("Window closed by user")
                    break
                vis.update_renderer()
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Interrupted by user")
        vis.destroy_window()

    def draw_registration_result(self, other, transformation):
        source_temp = copy.deepcopy(self.pointcloud)
        target_temp = copy.deepcopy(other.pointcloud)
        source_temp.paint_uniform_color([1, 0, 0])
        target_temp.paint_uniform_color([0, 0, 1])
        source_temp.transform(transformation)
        # o3d.visualization.draw_geometries([source_temp, target_temp],
        #                                   zoom=1.0,
        #                                   front=[0, 0, 10],
        #                                   lookat=[0, 0, 0],
        #                                   up=[0, 0, 1])
        o3d.visualization.draw_geometries([source_temp, target_temp])


    def unload_pointcloud(self):
        print('Removing pointclouds from memory (filtered, planes, fpfh): ')
        del self.pointcloud
        # del self.pointcloud_filtered
        # del self.pointcloud_fpfh
        # del self.pointcloud_ground_plane
        # del self.pointcloud_non_ground_plane
        self.pointcloud = None
        # self.pointcloud_filtered = None
        # self.pointcloud_ground_plane = None
        # self.pointcloud_non_ground_plane = None
        # self.pointcloud_fpfh = None

    # def registration(self, other, initial_transform, option='pointpoint'):
    #     """
    #     use icp to compute transformation using an initial estimate.
    #     caution, initial_transform is a np array.
    #     """
    #     if initial_transform is None:
    #         initial_transform = np.eye(4)
    #
    #     print("Initial transformation. Viewing initial transform:")
    #     # other.draw_registration_result(self, initial_transform)
    #
    #     print("Apply point-to-plane ICP. Local registration")
    #     threshold = ICP_PARAMETERS.distance_threshold
    #     # Initial version v1.0
    #     # if option == 'pointpoint':
    #     #     reg_p2p = o3d.pipelines.registration.registration_icp(
    #     #         other.pointcloud_filtered, self.pointcloud_filtered, threshold, initial_transform,
    #     #         o3d.pipelines.registration.TransformationEstimationPointToPoint())
    #     # elif option == 'pointplane':
    #
    #     reg_p2p = o3d.pipelines.registration.registration_icp(
    #                         other.pointcloud, self.pointcloud, threshold, initial_transform,
    #                         o3d.pipelines.registration.TransformationEstimationPointToPlane())
    #     # else:
    #     #     print('UNKNOWN OPTION. Should be pointpoint or pointplane')
    #     print('Registration result: ', reg_p2p)
    #     # print("Transformation is:")
    #     # print(reg_p2p.transformation)
    #     # other.draw_registration_result(self, reg_p2p.transformation)
    #     T = HomogeneousMatrix(reg_p2p.transformation)
    #     return T


    # def save_pointcloud_as_mesh(self):
    #     https: // www.open3d.org / docs / release / tutorial / geometry / surface_reconstruction.html
        # filename = self.directory + '/robot0/lidar/dataply/' + str(self.scan_time) + '.ply'
        # print('Saving pointcloud. Converting alpha shape: ', filename)
        # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(self.pointcloud, 0.01)
        # Load the original complete pointcloud
        # o3d.io.write_triangle_mesh(filename, mesh)
    #
    # def unload_pointcloud(self):
    #     print('Removing pointclouds from memory (filtered, planes, fpfh): ')
    #     del self.pointcloud
    #     del self.pointcloud_filtered
    #     del self.pointcloud_fpfh
    #     del self.pointcloud_ground_plane
    #     del self.pointcloud_non_ground_plane
    #     self.pointcloud = None
    #     self.pointcloud_filtered = None
    #     self.pointcloud_ground_plane = None
    #     self.pointcloud_non_ground_plane = None
    #     self.pointcloud_fpfh = None
    #
    # def filter_radius_height(self, radii=None, heights=None):
    #     if radii is None:
    #         min_radius = self.min_radius
    #         max_radius = self.max_radius
    #     else:
    #         min_radius = radii[0]
    #         max_radius = radii[1]
    #     if heights is None:
    #         min_height = self.min_height
    #         max_height = self.max_height
    #     else:
    #         min_height = heights[0]
    #         max_height = heights[1]
    #
    #     points = np.asarray(self.pointcloud.points)
    #     [x, y, z] = points[:, 0], points[:, 1], points[:, 2]
    #     r2 = x ** 2 + y ** 2
    #     idx = np.where(r2 < max_radius ** 2) and np.where(r2 > min_radius ** 2)
        # idx2 = np.where((r2 < max_radius ** 2) & (r2 > min_radius ** 2) & (z > min_height) & (z < max_height))
        # self.pointcloud_filtered = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[idx2]))
        # return self.pointcloud_filtered



    # def pre_process(self, method=False):
    #     if self.pre_processed:
    #         print('Already preprocessed, exiting')
    #         return
    #     if method == 'icppointpoint':
    #         self.preprocess_icp_point_point()
    #     elif method == 'icppointplane':
    #         self.preprocess_icp_point_plane()
    #     elif method == 'icp2planes':
    #         self.preprocess_icp2planes()
    #     elif method == 'fpfh':
    #         self.preprocess_fpfh()
    #
        # if simple:
        #     return
        # # advanced preprocessing for the two planes scanmatcher
        # if plane_model is None:
        #     self.plane_model = self.calculate_plane(pcd=self.pointcloud_filtered)
        # else:
        #     self.plane_model = plane_model
        #
        # pcd_ground_plane, pcd_non_ground_plane = self.segment_plane(self.plane_model, pcd=self.pointcloud_filtered)
        # self.pointcloud_ground_plane = pcd_ground_plane
        # self.pointcloud_non_ground_plane = pcd_non_ground_plane
        #
        # # self.draw_pointcloud(pcd_ground_plane)
        # # self.draw_pointcloud(pcd_non_ground_plane)
        #
        # self.pointcloud_ground_plane.estimate_normals(
        #     o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size_normals_ground_plane,
        #                                          max_nn=ICP_PARAMETERS.max_nn_gd))
        # self.pointcloud_non_ground_plane.estimate_normals(
        #     o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size_normals,
        #                                          max_nn=ICP_PARAMETERS.max_nn))

    # def preprocess_icp_point_point(self):
    #     self.pointcloud_filtered = self.filter_radius_height()
    #     if self.voxel_size is not None:
    #         self.pointcloud_filtered = self.pointcloud_filtered.voxel_down_sample(voxel_size=self.voxel_size)
    #     self.pointcloud_filtered.estimate_normals(
    #         o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size_normals,
    #                                              max_nn=ICP_PARAMETERS.max_nn))
    #
    # def preprocess_icp_point_plane(self):
    #     self.pointcloud_filtered = self.filter_radius_height()
    #     if self.voxel_size is not None:
    #         self.pointcloud_filtered = self.pointcloud_filtered.voxel_down_sample(voxel_size=self.voxel_size)
    #     self.pointcloud_filtered.estimate_normals(
    #         o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size_normals,
    #                                              max_nn=ICP_PARAMETERS.max_nn))

    # def preprocess_icp2planes(self):
    #     self.pointcloud_filtered = self.filter_radius_height()
    #     if self.voxel_size is not None:
    #         self.pointcloud_filtered = self.pointcloud_filtered.voxel_down_sample(voxel_size=self.voxel_size)
    #     self.pointcloud_filtered.estimate_normals(
    #         o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size_normals,
    #                                              max_nn=ICP_PARAMETERS.max_nn))
    #     # advanced preprocessing for the two planes scanmatcher
    #     # if plane_model is None:
    #     self.plane_model = self.calculate_plane(pcd=self.pointcloud_filtered)
    #     # else:
    #     #     self.plane_model = plane_model
    #
    #     pcd_ground_plane, pcd_non_ground_plane = self.segment_plane(self.plane_model, pcd=self.pointcloud_filtered)
    #     self.pointcloud_ground_plane = pcd_ground_plane
    #     self.pointcloud_non_ground_plane = pcd_non_ground_plane
    #
    #     # self.draw_pointcloud(pcd_ground_plane)
    #     # self.draw_pointcloud(pcd_non_ground_plane)
    #
    #     self.pointcloud_ground_plane.estimate_normals(
    #         o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size_normals_ground_plane,
    #                                              max_nn=ICP_PARAMETERS.max_nn_gd))
    #     self.pointcloud_non_ground_plane.estimate_normals(
    #         o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size_normals,
    #                                              max_nn=ICP_PARAMETERS.max_nn))
    #
    # def preprocess_fpfh(self):
    #     self.pointcloud_filtered = self.filter_radius_height()
    #     if self.voxel_size is not None:
    #         self.pointcloud_filtered = self.pointcloud_filtered.voxel_down_sample(voxel_size=self.voxel_size)
    #     self.pointcloud_filtered.estimate_normals(
    #         o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size_normals,
    #                                              max_nn=ICP_PARAMETERS.max_nn))
    #     # self.pointcloud_fpfh = o3d.pipelines.registration.compute_fpfh_feature(self.pointcloud_filtered,
    #     #                                                 o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size_normals,
    #     #                                                                                       max_nn=100))
    #     # self.pointcloud_filtered = self.filter_radius_height()
    #     if self.voxel_size is not None:
    #         self.pointcloud_filtered = self.pointcloud_filtered.voxel_down_sample(voxel_size=self.voxel_size)
    #     # self.pointcloud_filtered.estimate_normals(
    #     #     o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size_normals,
    #     #                                          max_nn=ICP_PARAMETERS.max_nn))
    #     # advanced preprocessing for the two planes scanmatcher
    #     # if plane_model is None:
    #     self.plane_model = self.calculate_plane(pcd=self.pointcloud_filtered)
    #     # else:
    #     #     self.plane_model = plane_model
    #
    #     pcd_ground_plane, pcd_non_ground_plane = self.segment_plane(self.plane_model, pcd=self.pointcloud_filtered)
    #     self.pointcloud_ground_plane = pcd_ground_plane
    #     self.pointcloud_non_ground_plane = pcd_non_ground_plane
    #
    #     # self.draw_pointcloud(pcd_ground_plane)
    #     # self.draw_pointcloud(pcd_non_ground_plane)
    #
    #     self.pointcloud_ground_plane.estimate_normals(
    #         o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size_normals_ground_plane,
    #                                              max_nn=ICP_PARAMETERS.max_nn_gd))
    #     self.pointcloud_non_ground_plane.estimate_normals(
    #         o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size_normals,
    #                                              max_nn=ICP_PARAMETERS.max_nn))
    #
    #     self.pointcloud_fpfh = o3d.pipelines.registration.compute_fpfh_feature(self.pointcloud_non_ground_plane,
    #                                                  o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size_normals,
    #                                                                                        max_nn=100))


    #
    # def local_registration_two_planes(self, other, initial_transform):
    #     """
    #     use icp to compute transformation using an initial estimate.
    #     caution, initial_transform is a np array.
    #     """
    #     print("Apply point-to-plane ICP. Local registration in two phases")
    #     threshold = ICP_PARAMETERS.distance_threshold
    #
    #     if initial_transform is None:
    #         initial_transform = np.eye(4)
    #
    #     # POINT TO PLANE ICP in two phases
    #     reg_p2pa = (o3d.pipelines.
    #                 registration.registration_icp(other.pointcloud_ground_plane,
    #                                               self.pointcloud_ground_plane, threshold, initial_transform,
    #                                               o3d.pipelines.registration.TransformationEstimationPointToPlane()))
    #     reg_p2pb = (o3d.pipelines.
    #                 registration.registration_icp(other.pointcloud_non_ground_plane,
    #                                               self.pointcloud_non_ground_plane, threshold, initial_transform,
    #                                               o3d.pipelines.registration.TransformationEstimationPointToPlane()))
    #
    #     t1 = HomogeneousMatrix(reg_p2pa.transformation).t2v(n=3)
    #     t2 = HomogeneousMatrix(reg_p2pb.transformation).t2v(n=3)
    #     # build solution using both solutions
    #     tx = t2[0]
    #     ty = t2[1]
    #     tz = t1[2]
    #     alpha = t1[3]
    #     beta = t1[4]
    #     gamma = t2[5]
    #     T = HomogeneousMatrix(np.array([tx, ty, tz]), Euler([alpha, beta, gamma]))
    #     # other.draw_registration_result(self, T.array)
    #     print('Registration results: ', reg_p2pb)
    #     return T
    #
    # def global_registration(self, other):
    #     """
    #     perform global registration followed by icp
    #     """
    #     # initial_transform = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
    #     #     other.pointcloud, self.pointcloud, other.pointcloud_fpfh, self.pointcloud_fpfh,
    #     #     o3d.pipelines.registration.FastGlobalRegistrationOption(maximum_correspondence_distance=self.fpfh_threshold))
    #
    #     # first try: keeping the ground plane and removing tree canopies
    #     # initial_transform = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
    #     #     other.pointcloud_filtered, self.pointcloud_filtered, other.pointcloud_fpfh, self.pointcloud_fpfh, True,
    #     #     self.fpfh_threshold,
    #     #     o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
    #     #     3, [
    #     #         o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
    #     #             0.9),
    #     #         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
    #     #             self.fpfh_threshold)
    #     #     ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    #
    #     # try as second version: removing the ground plane
    #     initial_transform = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
    #         other.pointcloud_non_ground_plane, self.pointcloud_non_ground_plane, other.pointcloud_fpfh,
    #         self.pointcloud_fpfh, True, self.fpfh_threshold,
    #         o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
    #         3, [
    #             o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
    #                 0.9),
    #             o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
    #                 self.fpfh_threshold)
    #         ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    #
    #     # other.draw_registration_result(self, initial_transform.transformation)
    #
    #     print("Apply point-to-plane ICP. Local registration")
    #     threshold = ICP_PARAMETERS.distance_threshold
    #     reg_p2p = o3d.pipelines.registration.registration_icp(
    #         other.pointcloud_filtered, self.pointcloud_filtered, threshold, initial_transform.transformation,
    #         o3d.pipelines.registration.TransformationEstimationPointToPlane())
    #
    #     # reg_p2p = o3d.pipelines.registration.registration_icp(
    #     #     other.pointcloud_filtered, self.pointcloud_filtered, threshold, initial_transform.transformation,
    #     #     o3d.pipelines.registration.TransformationEstimationPointToPlane())
    #     # other.draw_registration_result(self, reg_p2p.transformation)
    #     print(reg_p2p)
    #     print("Refined transformation is:")
    #     print(reg_p2p.transformation)
    #     T = HomogeneousMatrix(reg_p2p.transformation)
    #     return T
    #
    # def draw_registration_result(self, other, transformation):
    #     source_temp = copy.deepcopy(self.pointcloud_filtered)
    #     target_temp = copy.deepcopy(other.pointcloud_filtered)
    #     source_temp.paint_uniform_color([1, 0, 0])
    #     target_temp.paint_uniform_color([0, 0, 1])
    #     source_temp.transform(transformation)
    #     # o3d.visualization.draw_geometries([source_temp, target_temp],
    #     #                                   zoom=1.0,
    #     #                                   front=[0, 0, 10],
    #     #                                   lookat=[0, 0, 0],
    #     #                                   up=[0, 0, 1])
    #     o3d.visualization.draw_geometries([source_temp, target_temp])
    # # def save_registration_result(self, other, transformation):
    # #     source_temp = copy.deepcopy(self.pointcloud_filtered)
    # #     target_temp = copy.deepcopy(other.pointcloud_filtered)
    # #     source_temp.paint_uniform_color([1, 0, 0])
    # #     target_temp.paint_uniform_color([0, 0, 1])
    # #     source_temp.transform(transformation)
    # #     o3d.visualization.draw_geometries([source_temp, target_temp],
    # #                                       zoom=0.4459,
    # #                                       front=[0.9288, -0.2951, -0.2242],
    # #                                       lookat=[1.6784, 2.0612, 1.4451],
    # #                                       up=[-0.3402, -0.9189, -0.1996])
    #     # o3d.visualization.draw_geometries([source_temp, target_temp],
    #     #                                   zoom=0.4459,
    #     #                                   front=[0.9288, -0.2951, -0.2242],
    #     #                                   lookat=[1.6784, 2.0612, 1.4451],
    #     #                                   up=[0, 0, 1])
    #     # o3d.visualization.draw_geometries([source_temp, target_temp])
    #

    #
    # def draw_pointcloud(self, pointcloud):
    #     o3d.visualization.draw_geometries([pointcloud])
    #
    # def filter_max_dist(self, max_dist=5):
    #     points = np.asarray(self.pointcloud.points)
    #     d = np.linalg.norm(points, axis=1)
    #     index = d < max_dist
    #     self.pointcloud.points = o3d.utility.Vector3dVector(points[index, :])
    #
    # def filter_max_height(self, max_height=1.0):
    #     points = np.asarray(self.pointcloud.points)
    #     index = points[:, 2] < max_height
    #     self.pointcloud.points = o3d.utility.Vector3dVector(points[index, :])
    #
    # def transform(self, T):
    #     return self.pointcloud_filtered.transform(T)
    #
    #

    #
    # def calculate_plane(self, pcd=None, height=-0.5, thresholdA=0.01):
    #     # find a plane by removing some of the points at a given height
    #     # this best estimates a ground plane.
    #
    #     if pcd is None:
    #         points = np.asarray(self.pointcloud_filtered.points)
    #     else:
    #         points = np.asarray(pcd.points)
    #
    #     idx = points[:, 2] < height
    #     pcd_plane = o3d.geometry.PointCloud()
    #
    #     pcd_plane.points = o3d.utility.Vector3dVector(points[idx])
    #
    #     plane_model, inliers = pcd_plane.segment_plane(distance_threshold=thresholdA, ransac_n=3,
    #                                                    num_iterations=1000)
    #     [a, b, c, d] = plane_model
    #     print(f"Plane model calculated: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    #
    #     # plane_model = [0, 0, 1, 0.69]
    #     return plane_model
    #
    # def segment_plane(self, plane_model, pcd=None, thresholdB=0.4):
    #     """
    #     filter roughly the points that may belong to the plane.
    #     then estimate the plane with these points.
    #     find the distance of the points to the plane and classify
    #     """
    #     # find a plane by removing some of the points at a given height
    #     # this best estimates a ground plane.
    #     if pcd is None:
    #         points = np.asarray(self.pointcloud_filtered.points)
    #     else:
    #         points = np.asarray(pcd.points)
    #     [a, b, c, d] = plane_model
    #
    #     dist = np.abs(a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / np.sqrt(a * a + b * b + c * c)
    #     condicion = dist < thresholdB
    #     inliers_final = np.where(condicion == True)
    #     inliers_final = inliers_final[0]
    #
    #     # now select the final pointclouds
    #     plane_cloud = pcd.select_by_index(inliers_final)
    #     non_plane_cloud = pcd.select_by_index(inliers_final, invert=True)
    #     return plane_cloud, non_plane_cloud

    # def icp_corrected_transforms(self, keyframe_j, transformation_initial):
    #
    #     threshold = ICP_PARAMETERS.distance_threshold
    #
    #     # POINT TO PLANE ICP
    #
    #     reg_p2pa = (o3d.pipelines.
    #                 registration.registration_icp(keyframe_j.pointcloud_ground_plane,
    #                                               self.pointcloud_ground_plane, threshold, transformation_initial,
    #                                               o3d.pipelines.registration.TransformationEstimationPointToPlane()))
    #     reg_p2pb = (o3d.pipelines.
    #                 registration.registration_icp(keyframe_j.pointcloud_non_ground_plane,
    #                                               self.pointcloud_non_ground_plane, threshold, transformation_initial,
    #                                               o3d.pipelines.registration.TransformationEstimationPointToPlane()))
    #
    #     t1 = HomogeneousMatrix(reg_p2pa.transformation).t2v(n=3)
    #     t2 = HomogeneousMatrix(reg_p2pb.transformation).t2v(n=3)
    #     # build solution using both solutions
    #     tx = t2[0]
    #     ty = t2[1]
    #     tz = t1[2]
    #     alpha = t1[3]
    #     beta = t1[4]
    #     gamma = t2[5]
    #     T = HomogeneousMatrix(np.array([tx, ty, tz]), Euler([alpha, beta, gamma]))
    #     return T

    # def icp_corrected_transformsv2(self, keyframe_j, transformation_initial):
    #
    #     threshold = ICP_PARAMETERS.distance_threshold
    #
    #     reg_p2pc = (o3d.pipelines.
    #                 registration.registration_icp(keyframe_j.pointcloud_filtered,
    #                                               self.pointcloud_filtered, threshold, transformation_initial,
    #                                               o3d.pipelines.registration.TransformationEstimationPointToPlane()))
    #
    #     T = HomogeneousMatrix(reg_p2pc.transformation)
    #     return T



    # def pre_processv2(self):
    #     self.pointcloud_filtered = self.filter_by_radius(self.min_radius, self.max_radius)
    #     self.pointcloud_filtered, ind = self.pointcloud_filtered.remove_radius_outlier(nb_points=3, radius=0.3)
        #
        # if self.voxel_downsample_size is not None:
        #     self.pointcloud_filtered = self.pointcloud_filtered.voxel_down_sample(voxel_size=self.voxel_downsample_size)
        #
        # self.pointcloud_filtered.estimate_normals(
        #     o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size_normals,
        #                                          max_nn=ICP_PARAMETERS.max_nn))




    # def set_global_transform(self, transform):
    #     self.transform = transform
    #     return

    # def transform_to_global(self, point_cloud_sampling=10):
    #     """
    #         Use open3d to fast transform to global coordinates.
    #         Returns the pointcloud in global coordinates
    #     """
    #     T = HomogeneousMatrix(self.transform)
    #     pointcloud = self.pointcloud.uniform_down_sample(every_k_points=point_cloud_sampling)
    #     return pointcloud.transform(T.array)








