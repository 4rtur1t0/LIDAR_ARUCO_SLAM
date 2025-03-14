"""
Run a scanmatcher using O3D for consecutive scans.
"""
from eurocreader.eurocreader import EurocReader
from keyframemanager.keyframemanager import KeyFrameManager
from artelib.homogeneousmatrix import HomogeneousMatrix, compute_global_transformations, \
    compute_relative_transformations
import time
import numpy as np
import matplotlib.pyplot as plt
from tools.gpsconversions import gps2utm
from tools.sampling import sample_odometry, sample_times
from artelib.homogeneousmatrix import compute_homogeneous_transforms
import getopt
import sys
from artelib.euler import Euler
import yaml

def find_options():
    argv = sys.argv[1:]
    euroc_path = None
    try:
        opts, args = getopt.getopt(argv, "hi:", ["ifile="])
    except getopt.GetoptError:
        print('python run_scanmatcher.py -i <euroc_directory>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('python run_scanmatcher.py -i <euroc_directory>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            euroc_path = arg
    print('Input find_options directory is: ', euroc_path)
    return euroc_path


def read_scanmatcher_parameters(directory):
    yaml_file_global = directory + '/' + 'robot0/scanmatcher_parameters.yaml'
    with open(yaml_file_global) as file:
        parameters = yaml.load(file, Loader=yaml.FullLoader)
    return parameters


def plot_odometry(df_odo):
    plt.figure()
    plt.scatter(df_odo['x'], df_odo['y'])
    plt.show()


def plot_transformations(transforms):
    positions = []
    orientations = []
    for i in range(len(transforms)):
        t = transforms[i].t2v(n=3)
        positions.append([t[0], t[1], t[2]])
        orientations.append([t[3], t[4], t[5]])

    positions = np.array(positions)
    orientations = np.array(orientations)
    # plt.figure()
    plt.scatter(positions[:, 0], positions[:, 1])
    # plt.show()
    #plot, x, y, z
    # plot alpha, beta, gamma
    # plt.figure()
    # plt.scatter(positions[:, 0], positions[:, 1])
    # plt.show()


def view_results(relative_transforms_scanmatcher, relative_transforms_odo, df_gps):
    """
    Comparing odo, the scanmatcher result and GPS if available.
    """
    utm_coords_x, utm_coords_y = gps2utm(df_gps)
    delta_x_utm = utm_coords_x[2]-utm_coords_x[0]
    delta_y_utm = utm_coords_y[2]-utm_coords_y[0]
    gamma = np.arctan2(delta_y_utm, delta_x_utm)
    T0 = HomogeneousMatrix([0, 0, 0], Euler([0, 0, gamma]))

    # compute global odo transforms from local relative movements and the previous T0
    global_transforms_odo = compute_global_transformations(relative_transforms_odo, T0=T0)
    # compute global transforms from local scanmatcher transforms
    global_transforms_scanmatcher = compute_global_transformations(relative_transforms_scanmatcher, T0=T0)

    plt.figure()
    plot_transformations(global_transforms_odo)
    plot_transformations(global_transforms_scanmatcher)
    plt.plot(utm_coords_x, utm_coords_y)
    plt.show()


def compute_relative_odometry_transformations(df_odo):
    # Now compute absolute and relative transforms for odometry
    # compute homogeneous transforms from odometry
    global_transforms_odo = compute_homogeneous_transforms(df_odo)
    # compute relative homogeneous transforms form global transforms
    relative_transforms_odo = compute_relative_transformations(global_transforms=global_transforms_odo)
    return relative_transforms_odo


def prepare_experiment_data(euroc_read, start_index=20, delta_time=1.0):
    """
    Read times from LiDAR. Sample times uniformily using delta_time, then obtain the closest data readings at each time
    """
    # Read LiDAR data and sample times
    df_lidar = euroc_read.read_csv(filename='/robot0/lidar/data.csv')
    scan_times = df_lidar['#timestamp [ns]'].to_numpy()
    scan_times = sample_times(sensor_times=scan_times, start_index=start_index, delta_time=delta_time * 1e9)
    # read odometry and get the closest data to each of the times above
    df_odo = euroc_read.read_csv(filename='/robot0/odom/data.csv')
    odo_times = df_odo['#timestamp [ns]'].to_numpy()
    # this finds the data that appear in closest times to the scans (master_sensor_times, sensor_times)
    odo_times = euroc_read.get_closest_times(master_sensor_times=scan_times, sensor_times=odo_times)
    # now for each time, get the corresponding odometry value
    df_odo = euroc_read.get_df_at_times(df_data=df_odo, time_list=odo_times)
    # read and sample gps data, if possible
    try:
        df_gps = euroc_read.read_csv(filename='/robot0/gps0/data.csv')
        gps_times = df_gps['#timestamp [ns]'].to_numpy()
        gps_times = euroc_read.get_closest_times(master_sensor_times=scan_times, sensor_times=gps_times)
        df_gps = euroc_read.get_df_at_times(df_data=df_gps, time_list=gps_times)
    except FileNotFoundError:
        gps_times = None
        df_gps = None
    return scan_times, odo_times, gps_times, df_odo, df_gps

def get_scan_times(euroc_read):
    """
    Read times from LiDAR. Sample times uniformily using delta_time, then obtain the closest data readings at each time
    """
    # Read LiDAR data and sample times
    df_lidar = euroc_read.read_csv(filename='/robot0/lidar/data.csv')
    scan_times = df_lidar['#timestamp [ns]'].to_numpy()

    return scan_times


def converter(directory=None):
    """
    The script samples LiDAR data from a starting index.
    Initially, LiDAR scans are sampled based on the movement of the robot (odometry).
    A scanmatching procedure using ICP is carried out.
    The basic parameters to obtain an estimation of the robot movement are:
    - delta_time: the time between LiDAR scans. Beware that, in the ARVC dataset, the initial sample time for LiDARS may be
    as high as 1 second. In this case, a sensible delta_time would be 1s, so as to use all LiDAR data.
    - voxel_size: whether to reduce the pointcloud

    CAUTION:  the scanmatcher produces the movement of the LIDAR as installed on the robot. Transformation from odometry to LIDAR
    and LIDAR TO GPS may be needed to produce quality results for mapping or SLAM.
    """
    ################################################################################################
    # CONFIGURATION
    ################################################################################################
    if directory is None:
        # INDOOR
        # directory = '/media/arvc/INTENSO/DATASETS/INDOOR/I1-2024-03-06-13-44-09'
        # directory = '/media/arvc/INTENSO/DATASETS/INDOOR/I2-2024-03-06-13-50-58'
        # directory = '/media/arvc/INTENSO/DATASETS/INDOOR/I3-2024-04-22-15-21-28'
        # OUTDOOR
        directory = '/media/arvc/INTENSO/DATASETS/OUTDOOR/O1-2024-03-06-17-30-39'
        # directory = '/media/arvc/INTENSO/DATASETS/OUTDOOR/O2-2024-03-07-13-33-34'
        # directory = '/media/arvc/INTENSO/DATASETS/OUTDOOR/O3-2024-03-18-17-11-17'
        # directory = '/media/arvc/INTENSO/DATASETS/OUTDOOR/O4-2024-03-20-13-26-40'
        # directory = '/media/arvc/INTENSO/DATASETS/OUTDOOR/O5-2024-04-24-12-47-35'

        # mixed INDOOR/OUTDOOR
        # directory = '/media/arvc/INTENSO/DATASETS/INDOOR_OUTDOOR/IO1-2024-05-03-09-51-52'

    ###############################################################################################
    # READ EXPERIMENT DATA
    euroc_read = EurocReader(directory=directory)
    # caution, remove 20 samples (approx.) from the LiDAR data until data capture is stabilized
    # GPS is read only to check results
    scan_times = get_scan_times(euroc_read=euroc_read)
    keyframe_manager = KeyFrameManager(directory=directory, scan_times=scan_times, voxel_size=None)
    # OPTIONAL: visualize resulting map
    keyframe_manager.add_keyframes(keyframe_sampling=1)


    # now run the scanmatcher routine, for each pair of scans
    for i in range(0, len(scan_times) - 1):
        keyframe_manager.load_pointcloud(i)
        keyframe_manager.draw_keyframe(i)
        # keyframe_manager.save_pointcloud(i)
        keyframe_manager.save_pointcloud_as_mesh(i)
        keyframe_manager.unload_pointcloud(i)


if __name__ == "__main__":
    directory = find_options()
    converter(directory=directory)
