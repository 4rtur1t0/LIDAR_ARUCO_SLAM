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


# def read_aruco_ids(directory, filename):
#     euroc_read = EurocReader(directory=directory)
#     df_aruco = euroc_read.read_csv(filename=filename)
#     aruco_ids = df_aruco['aruco_id'].to_numpy()
#     return aruco_ids


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


# def plot_gps_graphslam(utmfactors, graphslam):
#     utmfactors = np.array(utmfactors)
#     fig = plt.figure(0)
#     plt.cla()
#     i = 0
#     data = []
#     while graphslam.current_estimate.exists(X(i)):
#         ce = graphslam.current_estimate.atPose3(X(i))
#         T = HomogeneousMatrix(ce.matrix())
#         data.append(T.pos())
#         i += 1
#     data = np.array(data)
#     plt.plot(utmfactors[:, 0], utmfactors[:, 1], 'o', color='red')
#     plt.plot(data[:, 0], data[:, 1], '.', color='blue')
#     plt.xlabel('X (m, UTM)')
#     plt.ylabel('Y (m, UTM)')
#     plt.pause(0.001)


def compute_relative_transformation(lidarscanarray, posesarray, i, j, T0gps):
    """
    Gets the times at the LiDAR observations at i and i+1.
    Gets the interpolated odometry values at those times.
    Computes an Homogeneous transform and computes the relative transformation Tij
    The T0gps transformation is considered. We are transforming from the LiDAR odometry to the GPS odometry (p. e.
    mounted to the front of the robot).
    This severs as a initial estimation for the ScanMatcher
    """
    timei = lidarscanarray.get_time(i)
    timej = lidarscanarray.get_time(j)
    odoi = posesarray.interpolated_pose_at_time(timestamp=timei)
    odoj = posesarray.interpolated_pose_at_time(timestamp=timej)
    Ti = odoi.T()*T0gps
    Tj = odoj.T()*T0gps
    Tij = Ti.inv()*Tj
    return Tij


def find_within_radius(graphslam, j_idx):
    candidates = []
    # all p
    combs = np.array(list(combinations(j_idx, 2)))
    # now find combinations with distance in the poses (2D) below threshold
    for pair in combs:
        i = pair[0]
        j = pair[1]
        posi = graphslam.current_estimate.atPose3(X(i))
        posj = graphslam.current_estimate.atPose3(X(j))
        d = distance2d(posi, posj)
        if d < 5:
            candidates.append(pair)
    return candidates

def distance2d(posi, posj):
    T_i = HomogeneousMatrix(posi.matrix())
    T_j = HomogeneousMatrix(posj.matrix())
    diff = T_i.pos()-T_j.pos()
    d = np.linalg.norm(diff[0:2])
    return d

def find_candidate_aruco_loop_closings(graphslam, edges_landmarks):
    """
    Looks for all the seen arruco ids in the map
    For each ARUCO, find the poses from which the ARUCO was observed. Find a number of poses that observed
    the same ARUCO and are close to each other.
    """
    aruco_ids_in_map = np.unique(edges_landmarks[:, 0])
    global_pairs = []
    # iterate through each landmark
    for aruco_id in aruco_ids_in_map:
        # find the indices of the poses from which this aruco_id was observed
        j_idx = np.where(edges_landmarks[:, 0] == aruco_id)[0]
        # for each j in j_idx, find pairs that match d<dmax=5m
        # these pairs constitute the poses in graphslam from which the aruco_id was observed
        pairs = find_within_radius(graphslam, j_idx)
        global_pairs.append(pairs)
    # flatten list
    global_pairs = [item for sublist in global_pairs for item in sublist]
    global_pairs = np.array(global_pairs)
    return global_pairs


def process_pairs_scanmatching(graphslam, lidarscanarray, pairs, n_pairs):
    result = []
    scanmatcher = ScanMatcher(lidarscanarray=lidarscanarray)
    Tlidar_gps = graphslam.T0_gps
    k = 0
    # process randomly a number of pairs n_random
    n_random = n_pairs
    source_array = np.arange(len(pairs))
    random_elements = np.random.choice(source_array, n_random, replace=False)
    pairs = pairs[random_elements]
    for pair in pairs:
        print('Process pairs scanmatching: ', k, ' out of ', len(pairs))
        i = pair[0]
        j = pair[1]
        lidarscanarray.load_pointcloud(i)
        lidarscanarray.filter_points(i)
        lidarscanarray.estimate_normals(i)
        lidarscanarray.load_pointcloud(j)
        lidarscanarray.filter_points(j)
        lidarscanarray.estimate_normals(j)
        # current transforms
        Ti = HomogeneousMatrix(graphslam.current_estimate.atPose3(X(i)).matrix())
        Tj = HomogeneousMatrix(graphslam.current_estimate.atPose3(X(j)).matrix())
        # transform from GPS to Lidar
        Ti = Ti * Tlidar_gps.inv()
        Tj = Tj * Tlidar_gps.inv()
        Tij_0 = Ti.inv() * Tj
        Tij = scanmatcher.registration(i=i, j=j, Tij_0=Tij_0, show=False)
        lidarscanarray.unload_pointcloud(i)
        lidarscanarray.unload_pointcloud(j)
        result.append([i, j, Tij])
        k += 1
    return result

def process_odometry(graphslam, odoobsarray, smobsarray, lidarscanarray, **kwargs):
    """
    Add edge relations to the map
    """
    Tlidar_gps = kwargs.get('T0gps')
    skip_optimization = kwargs.get('skip_optimization')
    base_time = lidarscanarray.get_time(0)
    edges_odo = []
    #################################################################################################
    # loop through all edges first, include relative measurements such as odometry and scanmatching
    #################################################################################################
    for i in range(len(lidarscanarray) - 1):
        # i, i+1 edges.
        print('ADDING EDGE (i, j): (', i, ',', i + 1, ')')
        print('At experiment times i: ', (lidarscanarray.get_time(i) - base_time) / 1e9)
        print('At experiment times i+1: ', (lidarscanarray.get_time(i + 1) - base_time) / 1e9)
        atb_odo = compute_relative_transformation(lidarscanarray=lidarscanarray, posesarray=odoobsarray, i=i, j=i + 1,
                                                  T0gps=Tlidar_gps)
        atb_sm = compute_relative_transformation(lidarscanarray=lidarscanarray, posesarray=smobsarray, i=i, j=i + 1,
                                                 T0gps=Tlidar_gps)
        # create the initial estimate of node i+1 using SM
        graphslam.add_initial_estimate(atb_sm, i + 1)
        # graphslam.add_initial_estimate(atb_odo, i + 1)
        # add edge observations between vertices. We are adding a binary factor between a newly observed state and
        # the previous state. Using scanmatching odometry and raw odometry
        graphslam.add_edge(atb_sm, i, i + 1, 'SM')
        graphslam.add_edge(atb_odo, i, i + 1, 'ODO')
        edges_odo.append([i, i+1])
        if i % skip_optimization == 0:
            print('GRAPHSLAM OPTIMIZE')
            print(50 * '*')
            graphslam.optimize()
            graphslam.plot_simple(plot3D=False)
    graphslam.optimize()
    graphslam.plot_simple(plot3D=False)
    graphslam.plot_simple(plot3D=True)
    return np.array(edges_odo)


def process_gps(graphslam, gpsobsarray, lidarscanarray, **kwargs):
    skip_optimization = kwargs.get('skip_optimization')
    utmfactors = []
    for i in range(len(lidarscanarray)):
        lidar_time = lidarscanarray.get_time(index=i)
        # given the lidar time, find the two closest GPS observations and get an interpolated GPS value
        gps_interp_reading = gpsobsarray.interpolated_utm_at_time(timestamp=lidar_time)
        if gps_interp_reading is not None:
            print('*** Adding GPS estimation at pose i: ', i)
            utmfactors.append([gps_interp_reading.x, gps_interp_reading.y, gps_interp_reading.altitude, i])
            graphslam.add_GPSfactor(utmx=gps_interp_reading.x, utmy=gps_interp_reading.y,
                                    utmaltitude=gps_interp_reading.altitude,
                                    gpsnoise=np.sqrt(gps_interp_reading.covariance),
                                    i=i)
        if i % skip_optimization == 0:
            print('GRAPHSLAM OPTIMIZE')
            print(50 * '*')
            graphslam.optimize()
            graphslam.plot_simple(plot3D=False, gps_utm_readings=utmfactors)
    graphslam.optimize()
    graphslam.plot_simple(plot3D=False, gps_utm_readings=utmfactors)
    graphslam.plot_simple(plot3D=True, gps_utm_readings=utmfactors)
    return utmfactors

def add_loopclosing_edges(graphslam, edges_transforms, **kwargs):
    """
    Add edge relations to the map
    """
    Tlidar_gps = kwargs.get('Tlidar_gps')
    skip_optimization = kwargs.get('skip_optimization')
    #################################################################################################
    # loop through all edges and add them
    # t
    #################################################################################################
    for edge in edges_transforms:
        # i, i+1 edges.
        i = edge[0]
        j = edge[1]
        Tij = edge[2]
        print('ADDING EDGE (i, j): (', i, ',', j, ')')
        Tij = Tij*Tlidar_gps
        graphslam.add_edge(Tij, i, j, 'SM')
        if i % skip_optimization == 0:
            print('GRAPHSLAM OPTIMIZE')
            print(50 * '*')
            graphslam.optimize()
            graphslam.plot_simple(plot3D=False)
    graphslam.optimize()
    graphslam.plot_simple(plot3D=False)
    graphslam.plot_simple(plot3D=True)



def process_aruco_landmarks(graphslam, arucoobsarray, lidarscanarray, **kwargs):
    # Filter ARUCO readings.
    Tgps_lidar = kwargs.get('Tgps_lidar')
    Tlidar_cam = kwargs.get('Tlidar_cam')
    skip_optimization = kwargs.get('skip_optimization')
    landmark_edges = []
    for j in range(len(arucoobsarray)):
        time_aruco = arucoobsarray.get_time(index=j)
        arucoobs = arucoobsarray.get(j)
        # the ARUCO observation from the camera reference system to the reference system placed on the GPS (the GPSfACTOR is directly applied)
        # transform the observation to the reference system on the GPS
        Tc_aruco = arucoobs.T()
        Tgps_aruco = Tgps_lidar*Tlidar_cam*Tc_aruco
        aruco_id = arucoobsarray.get_aruco_id(j)
        # The observation is attached to a pose X if the time is close to that correponding to the pose.
        # this is a simple solution, if the ARUCO observations are abundant it is highly possible to occur
        idx_lidar_graphslam, time_lidar_graphslam = lidarscanarray.get_index_closest_to_time(timestamp=time_aruco,
                                                                                           delta_threshold_s=0.05)
        # if no time was found, simply continue the process
        if idx_lidar_graphslam is None:
            continue
        # if the landmark does not exist, create it from pose idx_lidar_graphslam
        # CAUTION: the landmarks is exactly numbered as the ARUCO identifier
        # if the landmarks does not exist --> then create it. We create the landmark estimate using the index of the
        # closest pose (idx_lidar_graphslam), the observation Tgps_aruco (expressed in the reference system of the GPS)
        # and the aruco_id itself
        if not graphslam.current_estimate.exists(L(aruco_id)):
            graphslam.add_initial_landmark_estimate(Tgps_aruco, idx_lidar_graphslam, aruco_id)
        else:
            # if the landmark exists, create edge between pose i and landmark aruco_id
            # sigmas stand for alpha, beta, gamma, sigmax, sigmay, sigmaz (sigmax is larger, since corresponds to Zc)
            graphslam.add_edge_pose_landmark(atb=Tgps_aruco, i=idx_lidar_graphslam, j=aruco_id,
                                             sigmas=np.array([5, 5, 5, 0.2, 0.1, 0.1]))
            landmark_edges.append([aruco_id,  idx_lidar_graphslam])
        if j % skip_optimization == 0:
            print('GRAPHSLAM OPTIMIZE')
            print(50 * '*')
            graphslam.optimize()
            graphslam.plot_simple(plot3D=False)
    graphslam.optimize()
    graphslam.plot_simple(plot3D=False)
    graphslam.plot_simple(plot3D=True)
    return np.array(landmark_edges)


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

    ###################################################################################
    # FINALLY, compute scanmatching between edges including ARUCO observations and poses
    # Use ARUCO for loop closing
    # Process:
    # Th
    ###################################################################################
    # graphslam.plot_advanced()
    # refine_loop_closing_with_lidar
    candidate_pairs_scanmatching_aruco = find_candidate_aruco_loop_closings(graphslam=graphslam, edges_landmarks=edges_landmarks)
    # [i, j, Tij], actually, compute the transformations. The function computes the transforms in the lidar reference system
    edges_transforms = process_pairs_scanmatching(graphslam=graphslam, lidarscanarray=lidarscanarray,
                                                  pairs=candidate_pairs_scanmatching_aruco,
                                                  n_pairs=300)
    # add edges to map and optimize
    add_loopclosing_edges(graphslam=graphslam, edges_transforms=edges_transforms, Tlidar_gps=Tlidar_gps,
                          skip_optimization=20)

    # find candidates for loop closing for the rest of edges process them



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
