import numpy as np
from eurocreader.eurocreader import EurocReader
from artelib.vector import Vector
from artelib.quaternion import Quaternion

class PosesArray():
    """
    A list of observed poses (i. e. odometry), along with the time
    Can return:
    a) the interpolated Pose at a given time (from the two closest poses).
    b) the relative transformation T between two times.
    """
    def __init__(self, directory):
        """
        given a list of scan times (ROS times), each pcd is read on demand
        """
        self.directory = directory
        self.times = None
        self.data = []
        self.warning_max_time_diff_s = 1

    def init(self):
        # self.read_parameters()
        self.read_data()
        # self.add_lidar_scans()

    def read_data(self):
        euroc_read = EurocReader(directory=self.directory)
        df_odo = euroc_read.read_csv(filename='/robot0/odom/data.csv')
        self.times = df_odo['#timestamp [ns]'].to_numpy()
        for index, row in df_odo.iterrows():
            self.data.append(Pose(row))

    def get_time(self, index):
        return self.times[index]

    def get(self, index):
        return self.data[index]

    def get_closest_at_time(self, timestamp):
        d = np.abs(self.times - timestamp)
        index = np.argmin(d)
        time_diff_s = d[index]
        output_time = self.times[index]
        output_pose = self.data[index]
        if time_diff_s > self.warning_max_time_diff_s:
            print('CAUTION!!! Found time difference (s): ', time_diff_s / 1e9)
            print('CAUTION!!! Should we associate data??')
        return output_pose, output_time

    # def get_interpolated_at_time(self, timestamp):

    # def relative_transformation(timestamp1, timestamp2, Tr_gps):
    """
    Computes the relative transformation given two times
    """

class Pose():
    def __init__(self, df):
        """
        Create a pose from pandas df
        """
        self.position = Vector([df['x'], df['y'], df['z']])
        self.quaternion = Quaternion(qx=df['qx'], qy=df['qy'], qz=df['qz'], qw=df['qw'])

    # def T(self):
        # return T matrix

    # def R(self):