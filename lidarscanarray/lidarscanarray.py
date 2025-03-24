import time
import bisect
import numpy as np
from artelib.homogeneousmatrix import HomogeneousMatrix
import open3d as o3d
from lidarscanarray.lidarscan import LiDARScan
from eurocreader.eurocreader import EurocReader
from tools.sampling import sample_times
import yaml


class LiDARScanArray:
    def __init__(self, directory):
        """
        given a list of scan times (ROS times), each pcd is read on demand
        """
        self.directory = directory
        self.scan_times = None
        self.lidar_scans = []
        self.show_registration_result = False
        self.parameters = None

    def __len__(self):
        return len(self.scan_times)

    def __getitem__(self, item):
        return self.lidar_scans[item]

    def read_parameters(self):
        yaml_file_global = self.directory + '/' + 'robot0/scanmatcher_parameters.yaml'
        with open(yaml_file_global) as file:
            scanmatcher_parameters = yaml.load(file, Loader=yaml.FullLoader)
        self.parameters = scanmatcher_parameters
        return scanmatcher_parameters

    def read_data(self):
        euroc_read = EurocReader(directory=self.directory)
        df_lidar = euroc_read.read_csv(filename='/robot0/lidar/data.csv')
        scan_times = df_lidar['#timestamp [ns]'].to_numpy()
        self.scan_times = scan_times

    def sample_data(self):
        start_index = self.parameters.get('start_index', 0)
        delta_time = self.parameters.get('delta_time', None)
        scan_times = sample_times(sensor_times=self.scan_times, start_index=start_index, delta_time=delta_time)
        self.scan_times = scan_times

    def get_time(self, index):
        """
        Get the time for a corresponding index
        """
        return self.scan_times[index]

    def get_index_closest_to_time(self, timestamp, delta_threshold_s):
        """
        Given a timestamp. Find the closest time within delta_threshold_s seconds and return its corrsponding index.
        """
        idx1, t1, idx2, t2 = self.find_closest_times_around_t_bisect(timestamp)
        d1 = abs((timestamp - t1) / 1e9)
        d2 = abs((t2 - timestamp) / 1e9)
        print('Time ARUCO times: ', d1, d2)
        if (d1 > delta_threshold_s) and (d2 > delta_threshold_s):
            print('get_index_closest_to_time could not find any close time')
            return None, None
        if d1 <= d2:
            return idx1, t1
        else:
            return idx2, t2
        # return None

    def get_times(self):
        return self.scan_times

    def get(self, index):
        return self.lidar_scans[index]

    def find_closest_times_around_t_bisect(self, t):
        # Find the index where t would be inserted in sorted_times
        idx = bisect.bisect_left(self.scan_times, t)

        # Determine the two closest times
        if idx == 0:
            # t is before the first element
            return 0, self.scan_times[0], 1, self.scan_times[1]
        elif idx == len(self.scan_times):
            # t is after the last element
            return -2, self.scan_times[-2], -1, self.scan_times[-1]
        else:
            # Take the closest two times around t
            return idx - 1, self.scan_times[idx - 1], idx, self.scan_times[idx]

    def remove_orphan_lidars(self, pose_array):
        result_scan_times = []
        for timestamp in self.scan_times:
            pose_interpolated = pose_array.interpolated_pose_at_time(timestamp=timestamp)
            if pose_interpolated is None:
                continue
            # if a correct pose can be interpolated at time
            else:
                result_scan_times.append(timestamp)
        self.scan_times = result_scan_times

    def add_lidar_scans(self, keyframe_sampling=1):
        # First: add all keyframes with the known sampling
        for i in range(0, len(self.scan_times), keyframe_sampling):
            print("Keyframemanager: Adding Keyframe: ", i, "out of: ", len(self.scan_times), end='\r')
            self.add_lidar_scan(i)

    def add_lidar_scan(self, index):
        print('Adding keyframe with scan_time: ', self.scan_times[index])
        kf = LiDARScan(directory=self.directory, scan_time=self.scan_times[index], parameters=self.parameters)
        self.lidar_scans.append(kf)

    def load_pointclouds(self):
        for i in range(0, len(self.lidar_scans)):
            print("Keyframemanager: Loading Pointcloud: ", i, "out of: ", len(self.lidar_scans), end='\r')
            self.lidar_scans[i].load_pointcloud()

    def load_pointcloud(self, i):
        self.lidar_scans[i].load_pointcloud()

    def unload_pointcloud(self, i):
        self.lidar_scans[i].unload_pointcloud()

    def save_pointcloud(self, i):
        self.lidar_scans[i].save_pointcloud()

    def save_pointcloud_as_mesh(self, i):
        self.lidar_scans[i].save_pointcloud_as_mesh()

    def pre_process(self, index):
        self.lidar_scans[index].pre_process(method=self.method)

    def filter_points(self, index):
        self.lidar_scans[index].filter_points()

    def estimate_normals(self, index):
        self.lidar_scans[index].estimate_normals()


    # def compute_transformation(self, i, j, Tij):
    #     """
    #     Compute relative transformation using different methods:
    #     - Simple ICP.
    #     - Two planes ICP.
    #     - A global FPFH feature matching (which could be followed by a simple ICP)
    #     """
    #     # TODO: Compute inintial transformation from IMU
    #     if self.method == 'icppointpoint':
    #         transform = self.keyframes[i].local_registration_simple(self.keyframes[j], initial_transform=Tij.array,
    #                                                                 option='pointpoint')
    #     elif self.method == 'icppointplane':
    #         transform = self.keyframes[i].local_registration_simple(self.keyframes[j], initial_transform=Tij.array,
    #                                                                 option='pointplane')
    #     elif self.method == 'icp2planes':
    #         transform = self.keyframes[i].local_registration_two_planes(self.keyframes[j], initial_transform=Tij.array)
    #     elif self.method == 'fpfh':
    #         transform = self.keyframes[i].global_registration(self.keyframes[j])
    #     else:
    #         print('Unknown registration method')
    #         transform = None
    #     if self.show_registration_result:
    #         self.keyframes[j].draw_registration_result(self.keyframes[i], transformation=transform.array)
    #     return transform

    def draw_cloud(self, index):
        self.lidar_scans[index].draw_cloud()

    def draw_all_clouds(self):
        for i in range(len(self.lidar_scans)):
            self.lidar_scans[i].draw_cloud()

    # def visualize_keyframe(self, index):
    #     # self.keyframes[index].visualize_cloud()
    #     self.keyframes[index].draw_cloud()
    def draw_cloud_visualizer(self, index):
        self.lidar_scans[index].draw_cloud_visualizer()

    def draw_all_clouds_visualizer(self, sample=1):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        for i in range(0, len(self.scan_times), sample):
            vis.clear_geometries()
            self.load_pointcloud(i)
            self.lidar_scans[i].filter_height()
            self.lidar_scans[i].filter_radius()
            self.lidar_scans[i].down_sample()
            # view = vis.get_view_control()
            # view.set_up(np.array([1, 0, 0]))
            vis.add_geometry(self.lidar_scans[i].pointcloud, reset_bounding_box=True)
            # vis.update_geometry(self.lidar_scans[i].pointcloud)
            if not vis.poll_events():
                print("Window closed by user")
                break
            vis.update_renderer()
            time.sleep(0.01)
        vis.destroy_window()

    # kf.filter_radius_height(radii=radii, heights=heights)
    # kf.filter_radius(radii=radii)
    # kf.filter_height(heights=heights)
    # kf.down_sample()

    def visualize_map_online(self, global_transforms, radii=None, heights=None, clear=False):
        """
        Builds map rendering updates at each frame.

        Caution: the map is not built, but the o3d window is in charge of storing the points
        and viewing them.
        """
        print("VISUALIZING MAP FROM KEYFRAMES")
        print('NOW, BUILD THE MAP')
        if radii is None:
            radii = [0.5, 35.0]
        if heights is None:
            heights = [-120.0, 120.0]

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        # transform all keyframes to global coordinates.
        # pointcloud_global = o3d.geometry.PointCloud()
        # caution, the visualizer only adds the transformed pointcloud to
        # the window, without removing the other geometries
        # the global map (pointcloud_global) is not built.
        for i in range(len(self.lidar_scans)):
            if clear:
                vis.clear_geometries()
            print("Keyframe: ", i, "out of: ", len(self.lidar_scans), end='\r')
            kf = self.lidar_scans[i]
            kf.load_pointcloud()
            kf.filter_radius_height(radii=radii, heights=heights)
            # kf.filter_radius(radii=radii)
            # kf.filter_height(heights=heights)
            kf.down_sample()
            Ti = global_transforms[i]
            # transform to global and
            pointcloud_temp = kf.transform(T=Ti.array)
            # yuxtaponer los pointclouds
            # pointcloud_global = pointcloud_global + pointcloud_temp
            # vis.add_geometry(pointcloud_global, reset_bounding_box=True)
            vis.add_geometry(pointcloud_temp, reset_bounding_box=True)
            vis.get_render_option().point_size = 1
            # vis.update_geometry(pointcloud_global)
            vis.poll_events()
            vis.update_renderer()
        print('FINISHED! Use the window renderer to observe the map!')
        vis.run()
        vis.destroy_window()

    def build_map(self, global_transforms, keyframe_sampling=10, radii=None, heights=None):
        """
        Caution: in this case, the map is built using a pointcloud and adding the points to it. This may require a great
        amount of memory, however the result may be saved easily
        """
        if radii is None:
            radii = [0.5, 35.0]
        if heights is None:
            heights = [-120.0, 120.0]
        print("COMPUTING MAP FROM KEYFRAMES")
        sampled_transforms = []
        for i in range(0, len(global_transforms), keyframe_sampling):
            sampled_transforms.append(global_transforms[i])

        print('NOW, BUILD THE MAP')
        # transform all keyframes to global coordinates.
        pointcloud_global = o3d.geometry.PointCloud()
        for i in range(len(self.lidar_scans)):
            print("Keyframe: ", i, "out of: ", len(self.lidar_scans), end='\r')
            kf = self.lidar_scans[i]
            kf.filter_radius_height(radii=radii, heights=heights)
            kf.down_sample()
            Ti = sampled_transforms[i]
            # transform to global and
            pointcloud_temp = kf.transform(T=Ti.array)
            # yuxtaponer los pointclouds
            pointcloud_global = pointcloud_global + pointcloud_temp
        print('FINISHED! Use the renderer to view the map')
        # draw the whole map
        o3d.visualization.draw_geometries([pointcloud_global])
        return pointcloud_global












