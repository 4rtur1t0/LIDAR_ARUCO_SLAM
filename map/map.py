# import numpy as np
# from artelib.homogeneousmatrix import HomogeneousMatrix
# from artelib.tools import slerp
# from eurocreader.eurocreader import EurocReader
# from artelib.vector import Vector
# from artelib.quaternion import Quaternion
# import bisect
# import matplotlib.pyplot as plt
from lidarscanarray.lidarscanarray import LiDARScanArray
from observations.posesarray import PosesArray, ArucoPosesArray, ArucoLandmarksPosesArray


class Map():
    """
    A Map!
    The map is composed by:
    - A list of estimated poses X as the robot followed a path.
    - A LiDAR scan associated to each pose.
    - A number of ARUCO landmarks. This part is optional, however



    A number of methods are included in this class:
    - Plotting the global pointcloud, including the path and landmarks.
    - Plotting the pointcloud

    - Includes a method to localize on the map


    """
    def __init__(self, times=None, values=None):
        """
        given a list of scan times (ROS times), each pcd is read on demand
        """
        self.robotpath = None
        self.lidarscanarray = None
        self.landmarks = None

    def __len__(self):
        return len(self.robotpath)

    def read_data(self, directory):
        """
        Read the estimated path of the robot, landmarks and LiDAR scans.
        the map is formed by a set of poses, each pose associated to a pointcloud.
        no global map is built and stored. Though, the view_map method allows
        """
        self.robotpath = PosesArray()
        self.robotpath.read_data(directory=directory, filename='/robot0/SLAM/solution_graphslam_lidar.csv')
        # Load the LiDAR scan array. Each pointcloud with its associated time.
        # Each lidar scan is associated to a given pose in the robotpath
        self.lidarscanarray = LiDARScanArray(directory=directory)
        self.lidarscanarray.read_parameters()
        self.lidarscanarray.read_data()
        # remove scans without corresponding odometry (in consequence, without scanmatching)
        self.lidarscanarray.remove_orphan_lidars(pose_array=self.robotpath)
        # lidarscanarray.remove_orphan_lidars(pose_array=smobsarray)
        # load the scans according to the times, do not load the corresponding pointclouds
        self.lidarscanarray.add_lidar_scans()
        # also load the ARUCO landmarks
        self.landmarks = ArucoLandmarksPosesArray()
        self.landmarks.read_data(directory=directory, filename='/robot0/SLAM/solution_graphslam_landmarks.csv')

    def draw_all_clouds(self):
        # self.lidarscanarray.draw_all_clouds_visualizer()
        self.lidarscanarray.draw_all_clouds()

    def draw_map(self, terraplanist=False):
        """
        Possibilities:
        - view path
        - view pointclouds
        - view ARUCO landmarks
        """
        global_transforms = self.robotpath.get_transforms()
        self.lidarscanarray.draw_map(global_transforms=global_transforms,
                                     voxel_size=0.2,
                                     radii=[1, 20],
                                     heights=[-2, 1.0],
                                     keyframe_sampling=20,
                                     terraplanist=terraplanist)
        # drawposes_openstreet
        # draw


