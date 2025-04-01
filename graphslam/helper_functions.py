import numpy as np
from artelib.homogeneousmatrix import HomogeneousMatrix
from scanmatcher.scanmatcher import ScanMatcher
from gtsam.symbol_shorthand import X, L


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


def process_odometry(graphslam, odoobsarray, smobsarray, lidarscanarray):
    """
    Add edge relations to the map
    """
    Tlidar_gps = graphslam.Tlidar_gps
    skip_optimization = graphslam.skip_optimization
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
    # graphslam.plot_simple(plot3D=True)
    return np.array(edges_odo)


def process_gps(graphslam, gpsobsarray, lidarscanarray, **kwargs):
    skip_optimization = graphslam.skip_optimization
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
    # graphslam.plot_simple(plot3D=True, gps_utm_readings=utmfactors)
    return utmfactors


def process_aruco_landmarks(graphslam, arucoobsarray, lidarscanarray):
    # Filter ARUCO readings.
    Tgps_lidar = graphslam.Tgps_lidar
    Tlidar_cam = graphslam.Tlidar_cam
    skip_optimization = graphslam.skip_optimization
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
    # graphslam.plot_simple(plot3D=True)
    return np.array(landmark_edges)


def process_pairs_scanmatching(graphslam, lidarscanarray, pairs, n_pairs):
    result = []
    scanmatcher = ScanMatcher(lidarscanarray=lidarscanarray)
    Tlidar_gps = graphslam.Tlidar_gps
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


def process_triplets_scanmatching(graphslam, lidarscanarray, triplets):
    """
    Actually computing the transformation for each of the triplets with indexes (i, j, k)
    """
    result = []
    n = 0
    for triplet in triplets:
        print('Process pairs scanmatching: ', n, ' out of ', len(triplets))
        i = triplet[0]
        j = triplet[1]
        k = triplet[2]
        lidarscanarray.load_pointcloud(i)
        lidarscanarray.filter_points(i)
        lidarscanarray.estimate_normals(i)
        lidarscanarray.load_pointcloud(j)
        lidarscanarray.filter_points(j)
        lidarscanarray.estimate_normals(j)
        lidarscanarray.load_pointcloud(k)
        lidarscanarray.filter_points(k)
        lidarscanarray.estimate_normals(k)
        # the function compute_scanmatchin uses the initial estimation from the current estimation and then
        # performs scanmatching
        # CAUTION: the result is expressed in the LiDAR reference system, since it considers
        Tij = compute_scanmathing(graphslam=graphslam, lidarscanarray=lidarscanarray, i=i, j=j)
        Tik = compute_scanmathing(graphslam=graphslam, lidarscanarray=lidarscanarray, i=i, j=k)
        Tjk = compute_scanmathing(graphslam=graphslam, lidarscanarray=lidarscanarray, i=j, j=k)
        # remove lidar from memory
        lidarscanarray.unload_pointcloud(i)
        lidarscanarray.unload_pointcloud(j)
        lidarscanarray.unload_pointcloud(k)
        # result: the triplet and tranformations Tik, Tjk
        result.append([i, j, k, Tij, Tjk, Tik])
        n += 1
    return result


def compute_scanmathing(graphslam, lidarscanarray, i, j):
    scanmatcher = ScanMatcher(lidarscanarray=lidarscanarray)
    Tlidar_gps = graphslam.Tlidar_gps
    # current transforms. Compute initial transformation
    Ti = HomogeneousMatrix(graphslam.current_estimate.atPose3(X(i)).matrix())
    Tj = HomogeneousMatrix(graphslam.current_estimate.atPose3(X(j)).matrix())
    # transform from GPS to Lidar
    Ti = Ti * Tlidar_gps.inv()
    Tj = Tj * Tlidar_gps.inv()
    # initial approximation from current state
    Tij_0 = Ti.inv() * Tj
    Tij = scanmatcher.registration(i=i, j=j, Tij_0=Tij_0, show=False)
    return Tij