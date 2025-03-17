"""

"""
from __future__ import print_function
import gtsam
import gtsam.utils.plot as gtsam_plot
import matplotlib.pyplot as plt
import numpy as np
from artelib.homogeneousmatrix import HomogeneousMatrix


# Declare the 3D translational standard deviations of the prior factor's Gaussian model, in meters.
prior_xyz_sigma = 10.0000000
# Declare the 3D rotational standard deviations of the prior factor's Gaussian model, in degrees.
prior_rpy_sigma = 10.0000000
# Declare the 3D translational standard deviations of the odometry factor's Gaussian model, in meters.
odo_xyz_sigma = 0.05
# Declare the 3D rotational standard deviations of the odometry factor's Gaussian model, in degrees.
odo_rpy_sigma = 3
# Declare the 3D translational standard deviations of the scanmatcher factor's Gaussian model, in meters.
icp_xyz_sigma = 0.05
# Declare the 3D rotational standard deviations of the odometry factor's Gaussian model, in degrees.
icp_rpy_sigma = 0.05
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
    def __init__(self, T0, T0_gps):
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

        # Solver parameters
        parameters = gtsam.ISAM2Params()
        parameters.setRelinearizeThreshold(0.1)
        parameters.relinearizeSkip = 1
        self.isam = gtsam.ISAM2(parameters)

    def init_graph(self):
        T = self.T0
        # init graph starting at 0 and with initial pose T0 = eye
        self.graph.push_back(gtsam.PriorFactorPose3(0, gtsam.Pose3(T.array), self.PRIOR_NOISE))
        # CAUTION: the initial T0 transform is the identity.
        self.initial_estimate.insert(0, gtsam.Pose3())
        # self.current_estimate = self.initial_estimate
        self.current_estimate.insert(0, gtsam.Pose3())

    def add_edge(self, atb, i, j, noise_type):
        noise = self.select_noise(noise_type)
        # add consecutive observation
        self.graph.push_back(gtsam.BetweenFactorPose3(i, j, gtsam.Pose3(atb.array), noise))

    def add_GPSfactor(self, utmx, utmy, utmaltitude, i):
        utm = gtsam.Point3(utmx, utmy, utmaltitude)
        self.graph.add(gtsam.GPSFactor(i, utm, self.GPS_NOISE))

    def add_initial_estimate(self, atb, k):
        next_estimate = self.current_estimate.atPose3(k-1).compose(gtsam.Pose3(atb.array))
        self.initial_estimate.insert(k, next_estimate)
        self.current_estimate.insert(k, next_estimate)

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

    def plot_simple(self, plot3D = True, skip=1):
        """
        Print and plot the result simply.
        """
        if plot3D:
            # Plot the newly updated iSAM2 inference.
            fig = plt.figure(1)
            axes = fig.gca(projection='3d')
            plt.cla()

            i = 0
            data = []
            while self.current_estimate.exists(i):
                ce = self.current_estimate.atPose3(i)
                T = HomogeneousMatrix(ce.matrix())
                data.append(T.pos())
                i += np.max([skip, 1])
            data = np.array(data)
            axes.scatter(data[:, 0], data[:, 1], data[:, 2])
        else:
            # Plot the newly updated iSAM2 inference.
            fig = plt.figure(0)
            plt.cla()
            i = 0
            data = []
            while self.current_estimate.exists(i):
                ce = self.current_estimate.atPose3(i)
                T = HomogeneousMatrix(ce.matrix())
                data.append(T.pos())
                i += np.max([skip, 1])
            data = np.array(data)
            plt.plot(data[:, 0], data[:, 1], '.', color='blue')
            plt.xlabel('X (m, UTM)')
            plt.ylabel('Y (m, UTM)')
        plt.pause(0.00001)

    def plot_compare_GPS(self, df_gps, correspondences):
        """
        Print and plot the result simply.
        """
        plt.figure(3)
        i = 0
        data = []
        while self.current_estimate.exists(i):
            ce = self.current_estimate.atPose3(i)
            T = HomogeneousMatrix(ce.matrix())
            data.append(T.pos())
            i += 1
        data = np.array(data)
        # data = data[0:150]
        # df_gps = df_gps[0:150]
        plt.plot(data[:, 0], data[:, 1], marker='.', color='blue')
        plt.plot(df_gps['x'], df_gps['y'], marker='o', color='red')
        plt.legend(['GraphSLAM estimation', 'GPS UTM'])
        plt.title('Correspondences (estimation, GPS)')
        # plt.figure()
        for c in correspondences:
            x = [data[c[0], 0], df_gps['x'].iloc[c[1]]]
            y = [data[c[0], 1], df_gps['y'].iloc[c[1]]]
            plt.plot(x, y, color='black', linewidth=5)
            # plt.show()
        plt.pause(0.01)
        plt.show()

    def get_solution(self):
        return self.current_estimate

    def get_solution_transforms(self):
        solution_transforms = []
        i = 0
        while self.current_estimate.exists(i):
            ce = self.current_estimate.atPose3(i)
            T = HomogeneousMatrix(ce.matrix())
            solution_transforms.append(T)
            i += 1
        return solution_transforms

    def get_solution_transforms_lidar(self):
        solution_transforms = []
        i = 0
        while self.current_estimate.exists(i):
            ce = self.current_estimate.atPose3(i)
            T = HomogeneousMatrix(ce.matrix())
            solution_transforms.append(T*self.T0_gps.inv())
            i += 1
        return solution_transforms

