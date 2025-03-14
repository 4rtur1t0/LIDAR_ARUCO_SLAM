"""
    Visualize map from known/ground truth trajectory and LIDAR.

    Author: Arturo Gil.
    Date: 03/2024

    TTD: Save map for future use in MCL localization.
         The map can be now saved in PCL format in order to use o3D directly.
         Also, the map can be saved in a series of pointclouds along with their positions, however path planning using,
         for example, PRM, may not be direct
"""
from eurocreader.eurocreader import EurocReader
from keyframemanager.keyframemanager import KeyFrameManager
from artelib.homogeneousmatrix import compute_homogeneous_transforms
import open3d as o3d
import matplotlib.pyplot as plt

#
# def visualize_map_online(global_transforms, keyframe_manager, keyframe_sampling=10, radii=None, heights=None):
#     """
#     Builds map rendering updates at each frame.
#
#     Caution: the map is not built, but the o3d window is in charge of storing the points
#     and viewing them.
#     """
#     if radii is None:
#         radii = [0.5, 35.0]
#     if heights is None:
#         heights = [-120.0, 120.0]
#     # caution: sample transforms
#     sampled_global_transforms = []
#     for i in range(0, len(global_transforms), keyframe_sampling):
#         sampled_global_transforms.append(global_transforms[i])
#
#     print("COMPUTING MAP FROM KEYFRAMES")
#     # First: add all keyframes with the known sampling
#     keyframe_manager.add_keyframes(keyframe_sampling=keyframe_sampling)
#     # keyframe_manager.load_pointclouds()
#     keyframe_manager.visualize_map_online(global_transforms=sampled_global_transforms, radii=radii, heights=heights)
#
#
# def build_map(global_transforms, keyframemanager, keyframe_sampling=10):
#     """
#     Caution: in this case, the map is built using a pointcloud and adding the points to it. This may require a great
#     amount of memory, however the result may be saved easily
#     """
#     print("COMPUTING MAP FROM KEYFRAMES")
#     keyframemanager.add_keyframes(keyframe_sampling=keyframe_sampling)
#     keyframemanager.build_map(global_transforms=global_transforms, keyframe_sampling=keyframe_sampling)
# from tools.plottools import plot_3D_data

def plot_3D_with_loop_closures(df_data, loop_closures):
    """
        Print and plot the result simply. in 3D
    """
    plt.figure(0)
    # axes = fig.gca(projection='3d')
    # plt.cla()
    # plt.scatter(df_data['x'], df_data['y'], df_data['z'])
    plt.scatter(df_data['x'], df_data['y'])
    for k in range(len(loop_closures)):
        i = loop_closures['i'].iloc[k]
        j = loop_closures['j'].iloc[k]
        x = [df_data['x'].iloc[i], df_data['x'].iloc[j]]
        y = [df_data['y'].iloc[i], df_data['y'].iloc[j]]
        # z = [df_data['y'].iloc[lc[0]], df_data['y'].iloc[lc[1]]]
        plt.plot(x, y, color='black', linewidth=3)

    plt.show()


def view_result_map(global_transforms, directory, scan_times, keyframe_sampling, radii, heights, voxel_size):
    """
    View the map (visualize_map_online) or build it.
    When building it, an open3D kd-tree is obtained, which can be saved to a file (i.e.) a csv file.
    Also, the map can be viewed as a set of poses (i.e. x,y,z, alpha, beta, gamma) at certain timestamps associated to
    a scan reading at that time.
    """
    # use, for example, 1 out of 5 LiDARS to build the map
    # keyframe_sampling = 5
    # sample tran
    sampled_global_transforms = []
    for i in range(0, len(global_transforms), keyframe_sampling):
        sampled_global_transforms.append(global_transforms[i])
    # use, for example, voxel_size=0.2. Use voxel_size=None to use full resolution
    # voxel_size = None
    keyframe_manager = KeyFrameManager(directory=directory, scan_times=scan_times, voxel_size=voxel_size)
    # OPTIONAL: visualize resulting map
    keyframe_manager.add_keyframes(keyframe_sampling=keyframe_sampling)
    # keyframe_manager.load_pointclouds()
    # caution: only visualization. All points are kept by the visualization window
    # caution: the global transforms correspond to the scan_times
    keyframe_manager.visualize_map_online(global_transforms=sampled_global_transforms, radii=radii, heights=heights, clear=True)
    # the build map method actually returns a global O3D pointcloud
    pointcloud_global = keyframe_manager.build_map(global_transforms=global_transforms,
                                                   keyframe_sampling=keyframe_sampling, radii=radii, heights=heights)
    # pointcloud_global se puede guardar




def main():
    # Read the final transform (i.e. via GraphSLAM)
    # You may be using different estimations to build the map: i.e. scanmatching or the results from graphSLAM
    # select as desired
    # directory = '/media/arvc/INTENSO/DATASETS/INDOOR/I1-2024-03-06-13-44-09'
    # directory = '/media/arvc/INTENSO/DATASETS/INDOOR/I2-2024-03-06-13-50-58'
    # directory = '/media/arvc/INTENSO/DATASETS/INDOOR/I3-2024-04-22-15-21-28'
    # directory = '/media/arvc/INTENSO/DATASETS/OUTDOOR/O1-2024-03-06-17-30-39'
    # directory = '/media/arvc/INTENSO/DATASETS/OUTDOOR/O2-2024-03-07-13-33-34'
    directory = '/media/arvc/INTENSO/DATASETS/OUTDOOR/O3-2024-03-18-17-11-17'
    # directory = '/media/arvc/INTENSO/DATASETS/OUTDOOR/O4-2024-04-22-13-27-47'
    # directory = '/media/arvc/INTENSO/DATASETS/OUTDOOR/O5-2024-04-24-12-47-35'
    # directory = '/media/arvc/INTENSO/DATASETS/OUTDOOR/O6-2024-04-10-11-09-24'
    # directory = '/media/arvc/INTENSO/DATASETS/OUTDOOR/O7-2024-04-22-13-45-50'
    # directory = '/media/arvc/INTENSO/DATASETS/OUTDOOR/O8-2024-04-24-13-05-16'
    # directory = '/media/arvc/INTENSO/DATASETS/INDOOR_OUTDOOR/IO1-2024-05-03-09-51-52'

    # Select a solution from SLAM or from a scanmatcher, for example
    # filename = '/robot0/scanmatcher/scanmatcher_global.csv'
    filename = '/robot0/SLAM/solution_graphslam.csv'
    # use, for example, 1 out of 5 LiDARS to build the map
    keyframe_sampling = 20
    # use, for example, voxel_size=0.2. Use voxel_size=None to use full resolution
    voxel_size = None

    # Remove by filtering max and min radius and heights
    # basic scan filtering to build the map (Radius_min, Radius_max, Height_min, Height_max)
    # CAUTION: add mor
    radii = [0, 35.0]
    heights = [-100, 5.0]

    # read data
    euroc_read = EurocReader(directory=directory)
    df_map_poses = euroc_read.read_csv(filename=filename)
    global_transforms = compute_homogeneous_transforms(df_data=df_map_poses)
    scan_times = df_map_poses['#timestamp [ns]'].to_numpy()

    loop_closures = euroc_read.read_csv(filename='/robot0/SLAM/loop_closures.csv')
    plot_3D_with_loop_closures(df_data=df_map_poses, loop_closures=loop_closures)

    # keyframe_manager = KeyFrameMan    # ager(directory=directory, scan_times=scan_times, voxel_size=voxel_size)

    # Either view the map and visualize or visualize it as it goes
    # Option 1: build the map in a open3D cloud, then render it in a single shot
    # build_map(global_transforms, keyframe_manager, keyframe_sampling=keyframe_sampling)
    # Option 2: use the open3D renderer to add points and view them.
    # visualize_map_online(global_transforms, keyframe_manager, keyframe_sampling=keyframe_sampling,
    #                      radii=radii, heights=heights)
    view_result_map(global_transforms=global_transforms, directory=directory,
                    scan_times=scan_times, keyframe_sampling=keyframe_sampling,
                    radii=radii, heights=heights, voxel_size=voxel_size)



if __name__ == '__main__':
    main()

