"""
    Visualize map from known/ground truth trajectory and LIDAR.

    Author: Arturo Gil.
    Date: 03/2024

    TTD: Save map for future use in MCL localization.
         The map can be now saved in PCL format in order to use o3D directly.
         Also, the map can be saved in a series of pointclouds along with their positions, however path planning using,
         for example, PRM, may not be direct
"""
from map.map import Map

# def plot_3D_with_loop_closures(df_data, loop_closures):
#     """
#         Print and plot the result simply. in 3D
#     """
#     plt.figure(0)
#     # axes = fig.gca(projection='3d')
#     # plt.cla()
#     # plt.scatter(df_data['x'], df_data['y'], df_data['z'])
#     plt.scatter(df_data['x'], df_data['y'])
#     for k in range(len(loop_closures)):
#         i = loop_closures['i'].iloc[k]
#         j = loop_closures['j'].iloc[k]
#         x = [df_data['x'].iloc[i], df_data['x'].iloc[j]]
#         y = [df_data['y'].iloc[i], df_data['y'].iloc[j]]
#         # z = [df_data['y'].iloc[lc[0]], df_data['y'].iloc[lc[1]]]
#         plt.plot(x, y, color='black', linewidth=3)
#
#     plt.show()

def map_viewer():
    # Read the final transform (i.e. via GraphSLAM)
    # You may be using different estimations to build the map: i.e. scanmatching or the results from graphSLAM
    # select as desired
    directory = '/media/arvc/INTENSO/DATASETS/INDOOR_OUTDOOR/IO2-2025-03-25-16-54-17'

    # use, for example, 1 out of 5 LiDARS to build the map
    keyframe_sampling = 20
    # use, for example, voxel_size=0.2. Use voxel_size=None to use full resolution
    voxel_size = 0.2
    maplidar = Map()
    maplidar.read_data(directory=directory)
    # visualize the clouds relative to the LiDAR reference frame
    # maplidar.draw_all_clouds()
    # visualize the map on the UTM reference frame
    maplidar.draw_map(terraplanist=True, keyframe_sampling=keyframe_sampling, voxel_size=voxel_size)
    # maplidar.build_map()


if __name__ == '__main__':
    map_viewer()

