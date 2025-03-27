"""

"""
from __future__ import print_function
import gtsam
import gtsam.utils.plot as gtsam_plot
import matplotlib.pyplot as plt
import numpy as np
from artelib.homogeneousmatrix import HomogeneousMatrix
from gtsam.symbol_shorthand import X, L

# Declare the 3D translational standard deviations of the prior factor's Gaussian model, in meters.
from eurocreader.eurocreader import EurocReader

prior_xyz_sigma = 10000.0000000
# Declare the 3D rotational standard deviations of the prior factor's Gaussian model, in degrees.
prior_rpy_sigma = 1000.0000000
# Declare the 3D translational standard deviations of the odometry factor's Gaussian model, in meters.
odo_xyz_sigma = 0.1
# Declare the 3D rotational standard deviations of the odometry factor's Gaussian model, in degrees.
odo_rpy_sigma = 10
# Declare the 3D translational standard deviations of the scanmatcher factor's Gaussian model, in meters.
icp_xyz_sigma = 0.1
# Declare the 3D rotational standard deviations of the odometry factor's Gaussian model, in degrees.
icp_rpy_sigma = 5
# GPS noise: in UTM, x, y, height
gps_xy_sigma = 2.5
gps_altitude_sigma = 3.0

# gps_xy_sigma = 0.5
# gps_altitude_sigma = 2.0

# Declare the noise models
PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([prior_rpy_sigma*np.pi/180,
                                                         prior_rpy_sigma*np.pi/180,
                                                         prior_rpy_sigma*np.pi/180,
                                                         prior_xyz_sigma,
                                                         prior_xyz_sigma,
                                                         prior_xyz_sigma]))
# noise from the scanmatcher
SM_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([icp_rpy_sigma*np.pi/180,
                                                            icp_rpy_sigma*np.pi/180,
                                                            icp_rpy_sigma*np.pi/180,
                                                            icp_xyz_sigma,
                                                            icp_xyz_sigma,
                                                            icp_xyz_sigma]))

ODO_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([odo_rpy_sigma*np.pi/180,
                                                            odo_rpy_sigma*np.pi/180,
                                                            odo_rpy_sigma*np.pi/180,
                                                            odo_xyz_sigma,
                                                            odo_xyz_sigma,
                                                            odo_xyz_sigma]))

GPS_NOISE = gtsam.Point3(gps_xy_sigma, gps_xy_sigma, gps_altitude_sigma)


class GraphSLAM():
    def __init__(self, T0, T0_gps, max_number_of_landmarks=1000):
        # self.current_index = 0
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimate = gtsam.Values()
        self.current_estimate = gtsam.Values()
        # transforms
        self.T0 = T0
        self.T0_gps = T0_gps
        # noises
        self.PRIOR_NOISE = PRIOR_NOISE
        self.SM_NOISE = SM_NOISE
        self.ODO_NOISE = ODO_NOISE
        self.GPS_NOISE = gtsam.noiseModel.Diagonal.Sigmas(GPS_NOISE)
        # landmarks
        self.max_number_of_landmarks = max_number_of_landmarks
        # Solver parameters
        parameters = gtsam.ISAM2Params()
        parameters.setRelinearizeThreshold(0.1)
        parameters.relinearizeSkip = 1
        self.isam = gtsam.ISAM2(parameters)

    def init_graph(self):
        T = self.T0
        # init graph starting at 0 and with initial pose T0 = eye
        self.graph.push_back(gtsam.PriorFactorPose3(X(0), gtsam.Pose3(T.array), self.PRIOR_NOISE))
        # CAUTION: the initial T0 transform is the identity.
        self.initial_estimate.insert(X(0), gtsam.Pose3())
        # self.current_estimate = self.initial_estimate
        self.current_estimate.insert(X(0), gtsam.Pose3())

    def add_initial_estimate(self, atb, k):
        next_estimate = self.current_estimate.atPose3(X(k-1)).compose(gtsam.Pose3(atb.array))
        self.initial_estimate.insert(X(k), next_estimate)
        self.current_estimate.insert(X(k), next_estimate)

    def add_initial_landmark_estimate(self, atb, k, landmark_id):
        """
        Landmark k observed from pose i
        """
        landmark_estimate = self.current_estimate.atPose3(X(k)).compose(gtsam.Pose3(atb.array))
        self.initial_estimate.insert(L(landmark_id), landmark_estimate)
        self.current_estimate.insert(L(landmark_id), landmark_estimate)

    def add_edge(self, atb, i, j, noise_type):
        """
        Adds edge between poses i and j
        """
        noise = self.select_noise(noise_type)
        # add consecutive observation
        self.graph.push_back(gtsam.BetweenFactorPose3(X(i), X(j), gtsam.Pose3(atb.array), noise))

    def add_edge_pose_landmark(self, atb, i, j, sigmas):
        """
        Adds edge between poses i and j
        """
        noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([sigmas[0]*np.pi/180,
                                                            sigmas[1]*np.pi/180,
                                                            sigmas[2]*np.pi/180,
                                                            sigmas[3],
                                                            sigmas[4],
                                                            sigmas[5]]))
        # add consecutive observation
        self.graph.push_back(gtsam.BetweenFactorPose3(X(i), L(j), gtsam.Pose3(atb.array), noise))

    def add_GPSfactor(self, utmx, utmy, utmaltitude, gpsnoise, i):
        utm = gtsam.Point3(utmx, utmy, utmaltitude)
        if gpsnoise is None:
            self.graph.add(gtsam.GPSFactor(X(i), utm, self.GPS_NOISE))
        else:
            gpsnoise = gtsam.Point3(gpsnoise[0], gpsnoise[1], gpsnoise[2])
            gpsnoise = gtsam.noiseModel.Diagonal.Sigmas(sigmas=gpsnoise)
            self.graph.add(gtsam.GPSFactor(X(i), utm, gpsnoise))

    def optimize(self):
        self.isam.update(self.graph, self.initial_estimate)
        self.current_estimate = self.isam.calculateEstimate()
        self.initial_estimate.clear()

    def select_noise(self, noise_type):
        if noise_type == 'ODO':
            return self.ODO_NOISE
        elif noise_type == 'SM':
            return self.SM_NOISE
        elif noise_type == 'GPS':
            return self.GPS_NOISE

    def plot2D(self, plot_uncertainty_ellipse=False, skip=1):
        """Print and plot incremental progress of the robot for 3D Pose SLAM using iSAM2."""
        # Compute the marginals for all states in the graph.
        if plot_uncertainty_ellipse:
            marginals = gtsam.Marginals(self.graph, self.current_estimate)

        # Plot the newly updated iSAM2 inference.
        fig = plt.figure(0)
        i = 0
        while self.current_estimate.exists(i):
            if plot_uncertainty_ellipse:
                gtsam_plot.plot_pose2(0, self.current_estimate.atPose3(i), 0.5,
                                          marginals.marginalCovariance(i))
            else:
                gtsam_plot.plot_pose2(0, self.current_estimate.atPose3(i), 0.5, None)
            i += np.max([skip, 1])
        plt.pause(.01)

    def plot3D(self, plot_uncertainty_ellipse=False, skip=1):
        """Print and plot incremental progress of the robot for 3D Pose SLAM using iSAM2."""
        # Compute the marginals for all states in the graph.
        if plot_uncertainty_ellipse:
            marginals = gtsam.Marginals(self.graph, self.current_estimate)

        # Plot the newly updated iSAM2 inference.
        fig = plt.figure(1)
        axes = fig.gca(projection='3d')
        plt.cla()
        i = 0
        while self.current_estimate.exists(i):
            if plot_uncertainty_ellipse:
                gtsam_plot.plot_pose3(0, self.current_estimate.atPose3(i), 0.5,
                                                marginals.marginalCovariance(i))
            else:
                gtsam_plot.plot_pose3(0, self.current_estimate.atPose3(i), 0.5, None)
            i += np.max([skip, 1])
        plt.pause(.01)

    def plot(self, plot3D=True, plot_uncertainty_ellipse=True, skip=1):
        """Print and plot incremental progress of the robot for 3D Pose SLAM using iSAM2."""
        # Compute the marginals for all states in the graph.
        if plot_uncertainty_ellipse:
            marginals = gtsam.Marginals(self.graph, self.current_estimate)

        # Plot the newly updated iSAM2 inference.
        if plot3D:
            fig = plt.figure(1)
            axes = fig.gca(projection='3d')
            plt.cla()
        else:
            fig = plt.figure(0)

        i = 0
        while self.current_estimate.exists(i):
            if plot_uncertainty_ellipse:
                if plot3D:
                    gtsam_plot.plot_pose3(0, self.current_estimate.atPose3(i), 0.5,
                                                marginals.marginalCovariance(i))
                else:
                    gtsam_plot.plot_pose2(0, self.current_estimate.atPose3(i), 0.5,
                                          marginals.marginalCovariance(i))
            else:
                if plot3D:
                    gtsam_plot.plot_pose3(0, self.current_estimate.atPose3(i), 0.5, None)
                else:
                    gtsam_plot.plot_pose2(0, self.current_estimate.atPose3(i), 0.5, None)

            i += np.max([skip, 1])
        plt.pause(.01)

    def plot_simple(self, plot3D=True, skip=1, gps_utm_readings=None):
        """
        Print and plot the result simply (no covariances or orientations)
        """
        # include estimates for poses X
        i = 0
        positions = []
        while self.current_estimate.exists(X(i)):
            ce = self.current_estimate.atPose3(X(i))
            T = HomogeneousMatrix(ce.matrix())
            positions.append(T.pos())
            i += np.max([skip, 1])
        positions = np.array(positions)
        landmarks = []
        for j in range(self.max_number_of_landmarks):
            if self.current_estimate.exists(L(j)):
                ce = self.current_estimate.atPose3(L(j))
                T = HomogeneousMatrix(ce.matrix())
                landmarks.append(T.pos())
        landmarks = np.array(landmarks)
        if plot3D:
            # Plot the newly updated iSAM2 inference.
            fig = plt.figure(1)
            axes = fig.gca(projection='3d')
            plt.cla()
            if len(positions):
                axes.scatter(positions[:, 0], positions[:, 1], positions[:, 2], marker='.', color='blue')
            if len(landmarks) > 0:
                axes.scatter(landmarks[:, 0], landmarks[:, 1], marker='o', color='green')
            if gps_utm_readings is not None and len(gps_utm_readings) > 0:
                gps_utm_readings = np.array(gps_utm_readings)
                axes.scatter(gps_utm_readings[:, 0], gps_utm_readings[:, 1], gps_utm_readings[:, 3], marker='o', color='red')
            axes.legend()
        else:
            # Plot the newly updated iSAM2 inference.
            fig = plt.figure(0)
            plt.cla()
            if len(positions):
                plt.scatter(positions[:, 0], positions[:, 1], marker='.', color='blue')
            if len(landmarks) > 0:
                plt.scatter(landmarks[:, 0], landmarks[:, 1], marker='o', color='green')
            if gps_utm_readings is not None and len(gps_utm_readings) > 0:
                gps_utm_readings = np.array(gps_utm_readings)
                plt.scatter(gps_utm_readings[:, 0], gps_utm_readings[:, 1], marker='o', color='red')
            plt.xlabel('X (m, UTM)')
            plt.ylabel('Y (m, UTM)')

        plt.pause(0.00001)

    # def plot_compare_GPS(self, df_gps, correspondences):
    #     """
    #     Print and plot the result simply.
    #     """
    #     plt.figure(3)
    #     i = 0
    #     data = []
    #     while self.current_estimate.exists(i):
    #         ce = self.current_estimate.atPose3(i)
    #         T = HomogeneousMatrix(ce.matrix())
    #         data.append(T.pos())
    #         i += 1
    #     data = np.array(data)
    #     # data = data[0:150]
    #     # df_gps = df_gps[0:150]
    #     plt.plot(data[:, 0], data[:, 1], marker='.', color='blue')
    #     plt.plot(df_gps['x'], df_gps['y'], marker='o', color='red')
    #     plt.legend(['GraphSLAM estimation', 'GPS UTM'])
    #     plt.title('Correspondences (estimation, GPS)')
    #     # plt.figure()
    #     for c in correspondences:
    #         x = [data[c[0], 0], df_gps['x'].iloc[c[1]]]
    #         y = [data[c[0], 1], df_gps['y'].iloc[c[1]]]
    #         plt.plot(x, y, color='black', linewidth=5)
    #         # plt.show()
    #     plt.pause(0.01)
    #     plt.show()

    def get_solution(self):
        return self.current_estimate

    def get_solution_transforms(self):
        """
        This returns the states X as a homogeneous transform matrix.
        In this particular example, the state is represented as the position of the GPS on top of the robot.
        Using shorthand for X(i) (state at i)
        """
        solution_transforms = []
        i = 0
        while self.current_estimate.exists(X(i)):
            ce = self.current_estimate.atPose3(X(i))
            T = HomogeneousMatrix(ce.matrix())
            solution_transforms.append(T)
            i += 1
        return solution_transforms

    def get_solution_transforms_lidar(self):
        """
        This returns the states X as a homogeneous transform matrix.
        In this particular example, the state is represented as the position of the GPS on top of the robot.
        We transform this state to the center of the LiDAR.
        Using shorthand for X(i) (state at i)
        """
        solution_transforms = []
        i = 0
        while self.current_estimate.exists(X(i)):
            ce = self.current_estimate.atPose3(X(i))
            T = HomogeneousMatrix(ce.matrix())
            solution_transforms.append(T*self.T0_gps.inv())
            i += 1
        return solution_transforms

    def get_solution_transforms_landmarks(self):
        """
        Using shorthand for L(j) (landmark j)
        """
        solution_transforms = []
        # landmarks ids
        landmark_ids = []
        for i in range(self.max_number_of_landmarks):
            if self.current_estimate.exists(L(i)):
                ce = self.current_estimate.atPose3(L(i))
                T = HomogeneousMatrix(ce.matrix())
                solution_transforms.append(T)
                # Using i as aruco_id identifier
                landmark_ids.append(i)
            i += 1
        return solution_transforms, landmark_ids

    def save_solution(self, scan_times, directory):
        """
        Save the map.
        Saving a number of poses. Two reference systems:
        a) GPS reference system
        b) LiDAR reference system
        """
        euroc_read = EurocReader(directory=directory)
        global_transforms_gps = self.get_solution_transforms()
        global_transforms_lidar = self.get_solution_transforms_lidar()
        global_transforms_landmarks, landmark_ids = self.get_solution_transforms_landmarks()
        euroc_read.save_transforms_as_csv(scan_times, global_transforms_gps,
                                          filename='/robot0/SLAM/solution_graphslam_gps.csv')
        euroc_read.save_transforms_as_csv(scan_times, global_transforms_lidar,
                                          filename='/robot0/SLAM/solution_graphslam_lidar.csv')
        euroc_read.save_landmarks_as_csv(landmark_ids=landmark_ids, transforms=global_transforms_landmarks,
                                         filename='/robot0/SLAM/solution_graphslam_landmarks.csv')
