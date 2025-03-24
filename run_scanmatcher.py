"""
Run a scanmatcher using O3D for consecutive scans.
"""
from artelib.homogeneousmatrix import HomogeneousMatrix
from lidarscanarray.lidarscanarray import LiDARScanArray
from observations.posesarray import PosesArray, Pose
from scanmatcher.scanmatcher import ScanMatcher
import getopt
import sys
import matplotlib.pyplot as plt


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


def compute_global_transforms(relative_transforms, T0):
    """
    Compute the global transforms based
    """
    global_transforms = []
    global_transforms.append(T0)
    T = T0
    for i in range(len(relative_transforms)):
        Tij = relative_transforms[i]
        T = T*Tij
        global_transforms.append(T)
    return global_transforms


def compute_initial_relative_transformation(lidarscanarray, odoobsarray, i, j):
    """
    Gets the times at the LiDAR observations at i and i+1.
    Gets the interpolated odometry values at those times.
    Computes an Homogeneous transform and computes the relative transformation Tij
    This severs as a initial estimation for the ScanMatcher
    """
    timei = lidarscanarray.get_time(i)
    timej = lidarscanarray.get_time(j)
    odoi = odoobsarray.interpolated_pose_at_time(timestamp=timei)
    odoj = odoobsarray.interpolated_pose_at_time(timestamp=timej)
    Ti = odoi.T()
    Tj = odoj.T()
    Tij = Ti.inv()*Tj
    return Tij


def plot_transforms( transforms):
    x = []
    y = []
    for i in range(len(transforms)):
        T = transforms[i]
        t = T.pos()
        x.append(t[0])
        y.append(t[1])
    plt.figure()
    plt.scatter(x, y)
    plt.show()





def run_scanmatcher(directory=None):
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
        directory = '/media/arvc/INTENSO/DATASETS/test_arucos/test_arucos4'

    # odometry
    odoobsarray = PosesArray()
    odoobsarray.read_data(directory=directory, filename='/robot0/odom/data.csv')
    odoobsarray.plot_xy()
    # create scan Array,
    lidarscanarray = LiDARScanArray(directory=directory)
    lidarscanarray.read_parameters()
    lidarscanarray.read_data()
    # lidarscanarray.scan_times = lidarscanarray.scan_times[0:10]
    # remove lidars times without close odometry in time
    lidarscanarray.remove_orphan_lidars(pose_array=odoobsarray)
    lidarscanarray.add_lidar_scans()
    # create the scanmatching object
    scanmatcher = ScanMatcher(lidarscanarray=lidarscanarray)
    # Run the scanmatcher
    results_sm_relative = []
    lidarscanarray.load_pointcloud(0)
    lidarscanarray.filter_points(0)
    lidarscanarray.estimate_normals(0)
    for i in range(len(lidarscanarray)-1):
        print("PROCESSING SCAN i: ", i, "OUT OF ", len(lidarscanarray))
        # compute relative initial transformation from odometry from Lidar at time i and i+1
        Tini = compute_initial_relative_transformation(lidarscanarray, odoobsarray, i=i, j=i+1)
        lidarscanarray.load_pointcloud(i+1)
        lidarscanarray.filter_points(i+1)
        lidarscanarray.estimate_normals(i+1)
        sm_result_i = scanmatcher.registration(i=i, j=i+1, Tij_0=Tini)
        results_sm_relative.append(sm_result_i)
        lidarscanarray.unload_pointcloud(i)

    # Save the scanmatching result
    # The initial T0 transformations is chosen as the identity
    T0 = HomogeneousMatrix()
    results_sm_global = compute_global_transforms(results_sm_relative, T0)
    plot_transforms(results_sm_global)
    # results_sm_global = to_poses(results_sm_global)
    # Results array.  saving the results from the scanmatching process
    # caution, saving global transformation as results!
    sm_resultsarray = PosesArray()
    sm_resultsarray.from_transforms(times=lidarscanarray.get_times(), transforms=results_sm_global)
    sm_resultsarray.plot_xy()
    sm_resultsarray.save_data(directory=directory+'/robot0/scanmatcher', filename='/data.csv')
    # plot results
    sm_resultsarray.plot_xy()
    print('FINISHED SCANMATCHING!')


if __name__ == "__main__":
    directory = find_options()
    run_scanmatcher(directory=directory)
