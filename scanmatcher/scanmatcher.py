import time
import numpy as np
from artelib.homogeneousmatrix import HomogeneousMatrix
import open3d as o3d
from lidarscanarray.lidarscan import LiDARScan
from eurocreader.eurocreader import EurocReader
from tools.sampling import sample_times
import yaml


class ScanMatcher():
    def __init__(self, lidarscanarray, icp_threshold=1):
        """
        given a list of scan times (ROS times), each pcd is read on demand
        """
        self.lidarscanarray = lidarscanarray
        self.icp_threshold = icp_threshold

    def registration(self, i, j, Tij, method='icppointplane', show=False):
        """
        Compute relative transformation using different methods:
        - Simple ICP.
        - Two planes ICP.
        - A global FPFH feature matching (which could be followed by a simple ICP)
        """
        # transform = self.lidarscanarray[i].registration(self.lidarscanarray[j], initial_transform=Tij.array)
        # requires precomputation of normals
        if method == 'icppointplane':
            transform = self.registration_icp_point_plane(self.lidarscanarray[i], self.lidarscanarray[j],
                                                          initial_transform=Tij.array, show=show)
        else:
            transform = self.registration_icp_point_point(self.lidarscanarray[i], self.lidarscanarray[j],
                                                          initial_transform=Tij.array, show=show)


        # # TODO: Compute inintial transformation from IMU
        # if self.method == 'icppointpoint':
        #     transform = self.keyframes[i].local_registration_simple(self.keyframes[j], initial_transform=Tij.array,
        #                                                             option='pointpoint')
        # elif self.method == 'icppointplane':
        #     transform = self.keyframes[i].local_registration_simple(self.keyframes[j], initial_transform=Tij.array,
        #                                                             option='pointplane')
        # elif self.method == 'icp2planes':
        #     transform = self.keyframes[i].local_registration_two_planes(self.keyframes[j], initial_transform=Tij.array)
        # elif self.method == 'fpfh':
        #     transform = self.keyframes[i].global_registration(self.keyframes[j])
        # else:
        #     print('Unknown registration method')
        #     transform = None
        # if self.show_registration_result:
        #     self.keyframes[j].draw_registration_result(self.keyframes[i], transformation=transform.array)
        return transform

    def registration_icp_point_plane(self, one, other, initial_transform, show=False):
        """
        use icp to compute transformation using an initial estimate.
        caution, initial_transform is a np array.
        """
        if initial_transform is None:
            initial_transform = np.eye(4)
        if show:
            print("Initial transformation. Viewing initial transform:")
            other.draw_registration_result(one, initial_transform)

        print("Apply point-to-plane ICP. Local registration")
        # selfthreshold = ICP_PARAMETERS.distance_threshold
        # Initial version v1.0
        # if option == 'pointpoint':
        #     reg_p2p = o3d.pipelines.registration.registration_icp(
        #         other.pointcloud_filtered, self.pointcloud_filtered, threshold, initial_transform,
        #         o3d.pipelines.registration.TransformationEstimationPointToPoint())
        # elif option == 'pointplane':

        reg_p2p = o3d.pipelines.registration.registration_icp(
                            other.pointcloud, one.pointcloud, self.icp_threshold, initial_transform,
                            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        # else:
        #     print('UNKNOWN OPTION. Should be pointpoint or pointplane')
        print('Registration result: ', reg_p2p)
        print("Transformation is:")
        print(reg_p2p.transformation)
        if show:
            other.draw_registration_result(one, reg_p2p.transformation)
        T = HomogeneousMatrix(reg_p2p.transformation)
        return T

    def registration_icp_point_point(self, one, other, initial_transform, show=False):
        """
        use icp to compute transformation using an initial estimate.
        caution, initial_transform is a np array.
        """
        if initial_transform is None:
            initial_transform = np.eye(4)

        if show:
            print("Initial transformation. Viewing initial transform:")
            other.draw_registration_result(self, initial_transform)

        print("Apply point-to-plane ICP. Local registration")
        # selfthreshold = ICP_PARAMETERS.distance_threshold
        # Initial version v1.0
        # if option == 'pointpoint':
        #     reg_p2p = o3d.pipelines.registration.registration_icp(
        #         other.pointcloud_filtered, self.pointcloud_filtered, threshold, initial_transform,
        #         o3d.pipelines.registration.TransformationEstimationPointToPoint())
        # elif option == 'pointplane':

        reg_p2p = o3d.pipelines.registration.registration_icp(
                            other.pointcloud, one.pointcloud, self.icp_threshold, initial_transform,
                            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        # else:
        #     print('UNKNOWN OPTION. Should be pointpoint or pointplane')
        print('Registration result: ', reg_p2p)
        # print("Transformation is:")
        # print(reg_p2p.transformation)
        if show:
            other.draw_registration_result(self, reg_p2p.transformation)
        T = HomogeneousMatrix(reg_p2p.transformation)
        return T









