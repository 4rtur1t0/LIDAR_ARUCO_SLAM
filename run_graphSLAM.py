"""
Using GTSAM in a GraphSLAM context.
We are integrating odometry, scanmatching odometry and (if present) GPS.
    The state X is the position and orientation frame of the robot, placed on the GPS sensor.

"""
import numpy as np
from artelib.homogeneousmatrix import HomogeneousMatrix
from artelib.vector import Vector
from artelib.euler import Euler
from graphslam.loopclosing import LoopClosing
from graphslam.loopclosing2 import LoopClosing2
from helpers.helper_functions import process_odometry, process_gps, process_aruco_landmarks, \
    process_triplets_scanmatching
from lidarscanarray.lidarscanarray import LiDARScanArray
from observations.gpsarray import GPSArray
from observations.posesarray import PosesArray, ArucoPosesArray
import getopt
import sys
from graphslam.graphSLAM import GraphSLAM
import matplotlib.pyplot as plt
from gtsam.symbol_shorthand import X, L
from itertools import combinations
from scanmatcher.scanmatcher import ScanMatcher


def find_options():
    argv = sys.argv[1:]
    euroc_path = None
    try:
        opts, args = getopt.getopt(argv, "hi:", ["ifile="])
    except getopt.GetoptError:
        print('python run_graphSLAM.py -i <euroc_directory>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('python run_graphSLAM.py -i <euroc_directory>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            euroc_path = arg
    print('Input find_options directory is: ', euroc_path)
    return euroc_path


def plot_sensors(odoarray, smarray, gpsarray):
    plt.figure()
    odo = odoarray.get_transforms()
    sm = smarray.get_transforms()
    gps = gpsarray.get_utm_positions()
    podo = []
    psm = []
    pgps = []
    for i in range(len(odo)):
        podo.append(odo[i].pos())
    for i in range(len(sm)):
        psm.append(sm[i].pos())
    for i in range(len(gps)):
        if gps[i].status >= 0:
            pgps.append([gps[i].x, gps[i].y, gps[i].altitude])
    podo = np.array(podo)
    psm = np.array(psm)
    pgps = np.array(pgps)
    plt.scatter(podo[:, 0], podo[:, 1], label='odo')
    plt.scatter(psm[:, 0], psm[:, 1], label='sm')
    plt.scatter(pgps[:, 0], pgps[:, 1], label='gps')
    plt.legend()
    plt.xlabel('X (m, UTM)')
    plt.ylabel('Y (m, UTM)')
    # plt.show()
    plt.pause(0.01)


def process_loop_closing_lidar(graphslam, lidarscanarray, **kwargs):
    """
        Find possible loop closings. The loop closing are computed as triplets (i, j, k) where i and j are close in the
        trajectory (with a low uncertainty) and k may be far. This allows to compute three transformations using ICP:
            - Tij (using as seed Tij0, from poses)
            - Tik (using a seed Tik0, from the estimated poses)
            - Tjk (using a seed Tjk0, from the estimated poses)
        It must be: Tij*Tik*Tjk.inv()=I. The three tran
        possesses a low uncertainty from the , whereas Tik and Tjk may have larger uncertainty. The observations
        The ARUCO observations lead to a trajectory that is mostly correct.
        We plan to find candidates for loop closing in terms of triplets that can be filtered
    """
    # Find candidates for loop_closing. Computing triplets. Find unique triplets
    loop_closing = LoopClosing2(graphslam=graphslam, distance_backwards=7.0, radius_threshold=5.0)
    # Compute unique triplets with scanmatching
    triplets = loop_closing.find_feasible_triplets()
    print('FOUND triplet candidates. A total of: ', len(triplets))
    print('COMPUTING CANDIDATES with scanmatching: ', len(triplets))
    graphslam.plot_loop_closings(triplets)
    # process_triplets_scanmathing: given the triplets, a scanmatchin procedure is performed to try to close the loops
    # this process highly depends on the ability of the ICP procedure to find a consistent registration (transformation)
    # the transformations are then filtered assuring that the triplets are consistent with Tij*Tjk*Tik.inv()==I
    triplets_transforms = process_triplets_scanmatching(graphslam=graphslam, lidarscanarray=lidarscanarray, triplets=triplets)
    print('Checking triplets. ', len(triplets_transforms))
    # Filter out wrong scanmatchings (the transformation may be wrong). Check Tij*Tjk*Tik.inv()==I
    triplets_transforms = loop_closing.check_triplet_transforms(graphslam=graphslam, triplet_transforms=triplets_transforms)
    print('After filtering!! Adding a total of triplets:', len(triplets_transforms))
    add_loopclosing_edges(graphslam, triplets_transforms)



def add_loopclosing_edges(graphslam, triplets_transforms, **kwargs):
    """
    Add edge relations to the map
    """
    Tlidar_gps = graphslam.T0_gps
    Tlidar_gps_inv = Tlidar_gps.inv()
    # skip_optimization = kwargs.get('skip_optimization')
    skip_optimization = 50
    #################################################################################################
    # loop through all edges and add them to the graph
    #################################################################################################
    n = 0
    for triplet in triplets_transforms:
        print('ADDING TRIPLET AS EDGES TO THE GRAPH. Adding triplet, ', n, 'out of ', len(triplets_transforms))
        # i, i+1 edges.
        i = triplet[0]
        j = triplet[1]
        k = triplet[2]
        Tij = triplet[3]
        Tjk = triplet[4]
        Tik = triplet[5]
        print('ADDING EDGE (i, j): (', i, ',', j, ')')
        print('ADDING EDGE (i, j): (', j, ',', k, ')')
        print('ADDING EDGE (i, j): (', i, ',', k, ')')
        # transfrom from the relative lidar reference system to the gps reference system
        # yes... this formula should be applied
        Tij = Tlidar_gps_inv * Tij * Tlidar_gps
        Tjk = Tlidar_gps_inv * Tjk * Tlidar_gps
        Tik = Tlidar_gps_inv * Tik * Tlidar_gps
        # !! SMLC--> sm for loop closing may be noisier
        graphslam.add_edge(Tij, i, j, 'SM')
        graphslam.add_edge(Tjk, j, k, 'SM')
        graphslam.add_edge(Tik, i, k, 'SM')
        if i % skip_optimization == 0:
            print('GRAPHSLAM OPTIMIZE')
            print(50 * '*')
            graphslam.optimize()
        #     graphslam.plot_simple(plot3D=False)
        n += 1
    graphslam.optimize()
    graphslam.plot_simple(plot3D=False)
    graphslam.plot_simple(plot3D=True)


def run_graphSLAM(directory):
    """
    The SLAM map is created on the basis of the LiDAR times, everything is interpolated to that time
    """
    # Add the dataset directory
    if directory is None:
        # directory = '/media/arvc/INTENSO/DATASETS/INDOOR/I1-2024-03-06-13-44-09'
        # directory = '/media/arvc/INTENSO/DATASETS/INDOOR/I2-2024-03-06-13-50-58'
        # directory = '/media/arvc/INTENSO/DATASETS/INDOOR/I3-2024-04-22-15-21-28'
        # OUTDOOR
        # directory = '/media/arvc/INTENSO/DATASETS/OUTDOOR/O1-2024-03-06-17-30-39'
        # directory = '/media/arvc/INTENSO/DATASETS/OUTDOOR/O2-2024-03-07-13-33-34'
        # directory = '/media/arvc/INTENSO/DATASETS/OUTDOOR/O3-2024-03-18-17-11-17'
        # directory = '/media/arvc/INTENSO/DATASETS/OUTDOOR/O4-2024-04-22-13-27-47'
        #  directory = '/media/arvc/INTENSO/DATASETS/OUTDOOR/O5-2024-04-24-12-47-35'
        # directory = '/media/arvc/INTENSO/DATASETS/OUTDOOR/O6-2024-04-10-11-09-24'
        # directory = '/media/arvc/INTENSO/DATASETS/OUTDOOR/O7-2024-04-22-13-45-50'
        # directory = '/media/arvc/INTENSO/DATASETS/OUTDOOR/O8-2024-04-24-13-05-16'
        # # MIXTO
        # directory = '/media/arvc/INTENSO/DATASETS/INDOOR_OUTDOOR/IO1-2024-05-03-09-51-52'
        # directory = '/media/arvc/INTENSO/DATASETS/INDOOR_OUTDOOR/IO1-2024-05-03-09-51-52'
        # directory = '/media/arvc/INTENSO/DATASETS/test_arucos/test_arucos4'
        directory = '/media/arvc/INTENSO/DATASETS/INDOOR_OUTDOOR/IO2-2025-03-25-16-54-17'

    skip_optimization = 200
    # T0: Define the initial transformation (Prior for GraphSLAM)
    T0 = HomogeneousMatrix()
    # Caution: Actually, we are estimating the position and orientation of the GPS at this position at the robot.
    # T LiDAR-GPS
    Tlidar_gps = HomogeneousMatrix(Vector([0.36, 0, -0.4]), Euler([0, 0, 0]))
    # T GPS - LiDAR
    Tgps_lidar = Tlidar_gps.inv()
    # T LiDAR-camera
    Tlidar_cam = HomogeneousMatrix(Vector([0, 0.17, 0]), Euler([0, np.pi/2, -np.pi/2]))
    # odometry
    odoobsarray = PosesArray()
    odoobsarray.read_data(directory=directory, filename='/robot0/odom/data.csv')
    # scanmatcher
    smobsarray = PosesArray()
    smobsarray.read_data(directory=directory, filename='/robot0/scanmatcher/data.csv')
    # ARUCO observations. In the camera reference frame
    arucoobsarray = ArucoPosesArray()
    arucoobsarray.read_data(directory=directory, filename='/robot0/aruco/data.csv')
    # remove spurious ARUCO IDs
    arucoobsarray.filter_aruco_ids()
    # gpsobservations
    gpsobsarray = GPSArray()
    gpsobsarray.read_data(directory=directory, filename='/robot0/gps0/data.csv')
    gpsobsarray.read_config_ref(directory=directory)
    gpsobsarray.filter_measurements()
    # gpsobsarray.plot_xyz_utm()

    # Plot initial sensors as raw data
    plot_sensors(odoarray=odoobsarray, smarray=smobsarray, gpsarray=gpsobsarray)
    # create scan Array, We are actually estimating the poses at which
    lidarscanarray = LiDARScanArray(directory=directory)
    lidarscanarray.read_parameters()
    lidarscanarray.read_data()
    # remove scans without corresponding odometry (in consequence, without scanmatching)
    lidarscanarray.remove_orphan_lidars(pose_array=odoobsarray)
    lidarscanarray.remove_orphan_lidars(pose_array=smobsarray)
    # load the scans according to the times, do not load the corresponding pointclouds
    lidarscanarray.add_lidar_scans()
    # create the graphslam graph
    graphslam = GraphSLAM(T0=T0, T0_gps=Tlidar_gps)
    graphslam.init_graph()
    #####################################################################################################
    # Process odometry information first. The function includes odometry, scanmatching. Everything
    # (odometry and scanmatching odometry is projected to the reference system on top of the GPS.
    # edges_odo=[pose i, pose j]
    #####################################################################################################
    edges_odo = process_odometry(graphslam=graphslam, odoobsarray=odoobsarray,
                     smobsarray=smobsarray, lidarscanarray=lidarscanarray,
                     T0gps=Tlidar_gps, skip_optimization=skip_optimization)
    #####################################################################################################
    # Now include gps readings as GPSFactor. Lidar times and indexes are in the same order and equivalent
    #####################################################################################################
    gps_utm_factors = process_gps(graphslam=graphslam, gpsobsarray=gpsobsarray, lidarscanarray=lidarscanarray,
                                  skip_optimization=skip_optimization)
    #####################################################################################################
    # Now add ARUCO landmark observations. The ARUCO are fed using the ARUCO id and ARUCO observation
    # int the camera reference system.
    # edges_landmarks = [aruco_id, pose i], aruco_id (L(j)) seen from pose i X(i)
    #####################################################################################################
    edges_landmarks = process_aruco_landmarks(graphslam=graphslam, arucoobsarray=arucoobsarray, lidarscanarray=lidarscanarray,
                                              Tgps_lidar=Tgps_lidar, Tlidar_cam=Tlidar_cam, skip_optimization=skip_optimization)

    #####################################################################################################
    # process loop closing lidar. The following observations are based only on the LiDAR computation.
    # We loop through the trajectory and look for poses in a triangle. For each pose i, we look for a pose
    # j near it (with a low travelled distance) and a pose k (with a high travelled distance). By selecting different
    # travelled distances, we can find loopclosings that are close in distance or very far away
    # (a short loop or a large loope). Both kind of loops are beneficial to build the map.
    # Being a triplet, we look for an identity transform of the kind Tij*Tjk*Tki=I.
    # KEY IDEA: the estimation of the paths has been aligned with the inclusion of the ARUCO landmarks.
    # in particular indoor. As a result, the ICP-based scanmatching algorithm usually is able to obtain
    # a correct registration.
    # Adding these observations to the map refines
    #####################################################################################################
    process_loop_closing_lidar(graphslam=graphslam, lidarscanarray=lidarscanarray,
                               Tgps_lidar=Tgps_lidar, Tlidar_cam=Tlidar_cam, skip_optimization=skip_optimization)

    #####################################################################################################
    # SAVE THE MAP!
    # find candidates for loop closing for the rest of edges process them
    #####################################################################################################
    _, landmark_ids = graphslam.get_solution_transforms_landmarks()
    print('Landmarks in map. ', landmark_ids)
    # print('Landmarks with edges. ', landmarks_with_edges_ids)
    print('FINAL OPTIMIZATION OF THE MAP')
    graphslam.optimize()
    graphslam.plot_simple(skip=1, plot3D=False, gps_utm_readings=gps_utm_factors)
    print('ENDED SLAM!! SAVING RESULTS!!')
    graphslam.save_solution(directory=directory, scan_times=lidarscanarray.get_times())


if __name__ == "__main__":
    directory = find_options()
    run_graphSLAM(directory=directory)
