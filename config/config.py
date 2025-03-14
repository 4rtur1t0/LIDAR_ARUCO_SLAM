import yaml
import pandas as pd
import numpy as np
import os

class Icp_parameters():
    def __init__(self, yaml_file='icp_parameters.yaml'):
        # print(__file__)
        global_dir = os.path.dirname(os.path.abspath(__file__))
        yaml_file_global = global_dir + '/' + yaml_file
        with open(yaml_file_global) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
            self.max_radius = config.get('filter_by_radius').get('max_radius')
            self.min_radius = config.get('filter_by_radius').get('min_radius')

            self.max_height = config.get('filter_by_height').get('max_height')
            self.min_height = config.get('filter_by_height').get('min_height')

            # self.max_distance = config.get('filter_by_radius').get('max_radius')
            # self.min_distance = config.get('filter_by_radius').get('min_radius')

            self.voxel_size = config.get('down_sample').get('voxel_size')

            self.radius_gd = config.get('filter_ground_plane').get('radius_normals')
            self.max_nn_gd = config.get('filter_ground_plane').get('maximum_neighbors')

            self.radius_normals = config.get('normals').get('radius_normals')
            self.max_nn = config.get('normals').get('maximum_neighbors')

            self.distance_threshold = config.get('icp').get('distance_threshold')


# EXP_PARAMETERS = Exp_parameters()
ICP_PARAMETERS = Icp_parameters()