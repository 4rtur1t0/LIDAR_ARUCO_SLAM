import numpy as np

from artelib.tools import slerp
from eurocreader.eurocreader import EurocReader
from artelib.vector import Vector
from artelib.quaternion import Quaternion
import bisect

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
        time_diff_s = d[index] / 1e9
        output_time = self.times[index]
        output_pose = self.data[index]
        if time_diff_s > self.warning_max_time_diff_s:
            print('CAUTION!!! Found time difference (s): ', time_diff_s)
            print('CAUTION!!! Should we associate data??')
        return output_pose, output_time

    def interpolated_pose_at_time(self, timestamp):
        """
        Find a Pose for timestamp, looking for the two closest times
        """
        idx1, t1, idx2, t2 = self.find_closest_times_around_t_bisect(timestamp)
        print('Time distances: ', (timestamp-t1)/1e9, (t2-timestamp)/1e9)
        odo1 = self.data[idx1]
        odo2 = self.data[idx2]
        odointerp = self.interpolate_pose(odo1, t1, odo2, t2, timestamp)
        return odointerp

    def find_closest_times_around_t_bisect(self, t):
        # Find the index where t would be inserted in sorted_times
        idx = bisect.bisect_left(self.times, t)

        # Determine the two closest times
        if idx == 0:
            # t is before the first element
            return 0, self.times[0], 1, self.times[1]
        elif idx == len(self.times):
            # t is after the last element
            return -2, self.times[-2], -1, self.times[-1]
        else:
            # Take the closest two times around t
            return idx-1, self.times[idx - 1], idx,  self.times[idx]

    def interpolate_pose(self, odo1, t1, odo2, t2, t):
        # Compute interpolation factor
        alpha = (t - t1) / (t2 - t1)

        # Linear interpolation of position
        p_t = (1 - alpha) * odo1.position.pos() + alpha * odo2.position.pos()
        q1 = odo1.quaternion
        q2 = odo2.quaternion
        q_t = slerp(q1, q2, alpha)

        poset = {'x': p_t[0],
                'y': p_t[1],
                'z': p_t[2],
                'qx': q_t.qx,
                'qy': q_t.qy,
                'qz': q_t.qz,
                'qw': q_t.qw}
        interppose = Pose(df=poset)
        return interppose

    # Example Usage
    # times = [10.2, 3.5, 7.8, 15.0, 12.4]
    # t = 9.0  # The time for which we need the closest bounds
    # t1, t2 = find_closest_times_around_t(times, t)
    # print(f"Two closest times around {t}: {t1}, {t2}")


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
