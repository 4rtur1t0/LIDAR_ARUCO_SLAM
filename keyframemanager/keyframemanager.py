import numpy as np
# import subprocess
from artelib.homogeneousmatrix import HomogeneousMatrix
import open3d as o3d
from keyframemanager.keyframe import KeyFrame


class KeyFrameManager():
    def __init__(self, directory, scan_times, voxel_size, method='icppointplane'):
        """
        given a list of scan times (ROS times), each pcd is read on demand
        """
        self.directory = directory
        self.scan_times = scan_times
        self.keyframes = []
        self.voxel_size = voxel_size
        self.method = method
        self.show_registration_result = False

    def add_keyframes(self, keyframe_sampling):
        # First: add all keyframes with the known sampling
        for i in range(0, len(self.scan_times), keyframe_sampling):
            print("Keyframemanager: Adding Keyframe: ", i, "out of: ", len(self.scan_times), end='\r')
            self.add_keyframe(i)

    def add_keyframe(self, index):
        print('Adding keyframe with scan_time: ', self.scan_times[index])
        kf = KeyFrame(directory=self.directory, scan_time=self.scan_times[index],
                      voxel_size=self.voxel_size)
        self.keyframes.append(kf)

    def load_pointclouds(self):
        for i in range(0, len(self.keyframes)):
            print("Keyframemanager: Loading Pointcloud: ", i, "out of: ", len(self.keyframes), end='\r')
            self.keyframes[i].load_pointcloud()

    def load_pointcloud(self, i):
        self.keyframes[i].load_pointcloud()

    def unload_pointcloud(self, i):
        self.keyframes[i].unload_pointcloud()

    def save_pointcloud(self, i):
        self.keyframes[i].save_pointcloud()

    def save_pointcloud_as_mesh(self, i):
        self.keyframes[i].save_pointcloud_as_mesh()

    def pre_process(self, index):
        self.keyframes[index].pre_process(method=self.method)

    def compute_transformation(self, i, j, Tij):
        """
        Compute relative transformation using different methods:
        - Simple ICP.
        - Two planes ICP.
        - A global FPFH feature matching (which could be followed by a simple ICP)
        """
        # TODO: Compute inintial transformation from IMU
        if self.method == 'icppointpoint':
            transform = self.keyframes[i].local_registration_simple(self.keyframes[j], initial_transform=Tij.array,
                                                                    option='pointpoint')
        elif self.method == 'icppointplane':
            transform = self.keyframes[i].local_registration_simple(self.keyframes[j], initial_transform=Tij.array,
                                                                    option='pointplane')
        elif self.method == 'icp2planes':
            transform = self.keyframes[i].local_registration_two_planes(self.keyframes[j], initial_transform=Tij.array)
        elif self.method == 'fpfh':
            transform = self.keyframes[i].global_registration(self.keyframes[j])
        else:
            print('Unknown registration method')
            transform = None
        if self.show_registration_result:
            self.keyframes[j].draw_registration_result(self.keyframes[i], transformation=transform.array)
        return transform

    def draw_keyframe(self, index):
        self.keyframes[index].draw_cloud()

    def visualize_keyframe(self, index):
        # self.keyframes[index].visualize_cloud()
        self.keyframes[index].draw_cloud()

    def draw_all_clouds(self, sample=3, max_dist=15, max_height=1.5):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        for i in range(0, len(self.scan_times), sample):
            vis.clear_geometries()
            self.add_keyframe(i)
            self.load_pointcloud(i)
            # self.keyframes[-1].filter_max_dist(max_dist=max_dist)
            # self.keyframes[-1].filter_max_height(max_height=max_height)
            self.keyframes[-1].filter_radius_height()
            self.keyframes[-1].down_sample()
            # view = vis.get_view_control()
            # view.set_up(np.array([1, 0, 0]))
            vis.add_geometry(self.keyframes[-1].pointcloud, reset_bounding_box=True)
            # vis.update_geometry(self.keyframes[i].pointcloud)
            vis.poll_events()
            vis.update_renderer()
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
        for i in range(len(self.keyframes)):
            if clear:
                vis.clear_geometries()
            print("Keyframe: ", i, "out of: ", len(self.keyframes), end='\r')
            kf = self.keyframes[i]
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
        for i in range(len(self.keyframes)):
            print("Keyframe: ", i, "out of: ", len(self.keyframes), end='\r')
            kf = self.keyframes[i]
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












