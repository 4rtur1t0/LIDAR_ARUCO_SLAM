import numpy as np
import yaml

from artelib.homogeneousmatrix import HomogeneousMatrix
from artelib.tools import slerp
from eurocreader.eurocreader import EurocReader
from artelib.vector import Vector
from artelib.quaternion import Quaternion
import bisect
import matplotlib.pyplot as plt
from pyproj import Proj


class GPSArray():
    """
    A list of observed GPS observations, along with the time
    Can return:
    a) the interpolated GPS at a given time (from the two closest poses).
    b) the transformed UTM coordinates
    b) the distance two GPS.
    """
    def __init__(self, times=None, values=[]):
        """
        given a list of scan times (ROS times), each pcd is read on demand
        """
        # self.directory = directory
        self.times = times
        self.values = values
        self.warning_max_time_diff_s = 1
        self.config_ref = None

    # def init(self):
    #     self.read_parameters()
        # self.read_data()
        # self.add_lidar_scans()

    def read_data(self, directory, filename):
        euroc_read = EurocReader(directory=directory)
        df_odo = euroc_read.read_csv(filename=filename)
        self.times = df_odo['#timestamp [ns]'].to_numpy()
        for index, row in df_odo.iterrows():
            gpspos = GPSPosition()
            gpspos.fromdf(row)
            self.values.append(gpspos)

    def read_config_ref(self, directory):
        yaml_file_global = directory + '/' + 'robot0/gps0/reference.yaml'
        with open(yaml_file_global) as file:
            parameters = yaml.load(file, Loader=yaml.FullLoader)
        self.config_ref = parameters
        return parameters

    def save_data(self, directory, filename):
        euroc_read = EurocReader(directory=directory)
        euroc_read.save_transforms_as_csv(transforms=self.values,
                                          sensor_times=self.times, filename=filename)

    def get_time(self, index):
        return self.times[index]

    def get_times(self):
        return self.times

    def get(self, index):
        return self.values[index]

    def get_closest_at_time(self, timestamp):
        d = np.abs(self.times - timestamp)
        index = np.argmin(d)
        time_diff_s = d[index] / 1e9
        output_time = self.times[index]
        output_pose = self.values[index]
        if time_diff_s > self.warning_max_time_diff_s:
            print('CAUTION!!! Found time difference (s): ', time_diff_s)
            print('CAUTION!!! Should we associate data??')
        return output_pose, output_time

    def interpolated_utm_at_time(self, timestamp, delta_threshold_s=1):
        """
        Find a Pose for timestamp, looking for the two closest times
        """
        idx1, t1, idx2, t2 = self.find_closest_times_around_t_bisect(timestamp)
        print('Time distances: ', (timestamp-t1)/1e9, (t2-timestamp)/1e9)
        # ensure t1 < t < t2 and the time distances are below a threshold s
        if ((timestamp - t1)/1e9 > delta_threshold_s) or ((t2-timestamp)/1e9 > delta_threshold_s):
            return None
        if t1 < timestamp < t2:
            gps1 = self.values[idx1]
            gps2 = self.values[idx2]
            utm1 = gps1.to_utm(config_ref=self.config_ref)
            utm2 = gps2.to_utm(config_ref=self.config_ref)
            odointerp = self.interpolate_utm(utm1, t1, utm2, t2, timestamp)
            return odointerp
        return None

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

    def interpolate_utm(self, odo1, t1, odo2, t2, t):
        # Compute interpolation factor
        alpha = (t - t1) / (t2 - t1)
        # Linear interpolation of position and altitude
        p_t = (1 - alpha) * odo1.position.pos() + alpha * odo2.position.pos()
        interutm = UTMPosition(x=p_t[0], y=p_t[1], altitude=p_t[2])
        return interutm

    def plot_xy(self):
        x = []
        y = []
        for i in range(len(self.times)):
            T = self.values[i]
            t = T.pos()
            x.append(t[0])
            y.append(t[1])
        plt.figure()
        plt.scatter(x, y)
        plt.show()

    # Example Usage
    # times = [10.2, 3.5, 7.8, 15.0, 12.4]
    # t = 9.0  # The time for which we need the closest bounds
    # t1, t2 = find_closest_times_around_t(times, t)
    # print(f"Two closest times around {t}: {t1}, {t2}")


class GPSPosition():
    def __init__(self, latitude=None, longitude=None, altitude=None, covariance=[], status=0):
        """
        Create a pose from pandas df
        """
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.covariance = covariance
        self.status = status

    def fromdf(self, df):
        """
        Create a pose from pandas df
        """
        self.latitude = df['latitude']
        self.longitude = df['longitude']
        self.altitude = df['altitude']
        self.covariance = np.array([df['covariance_d1'], df['covariance_d2'], df['covariance_d3']])
        self.status = df['status']

    def to_utm(self, config_ref):
        x, y, altitude = gps2utm(latitude=self.latitude, longitude=self.longitude, altitude=self.altitude, config_ref=config_ref)
        return UTMPosition(x=x, y=y, altitude=altitude)


class UTMPosition():
    def __init__(self, x, y, altitude):
        self.x = x
        self.y = y
        self.altitude = altitude

    def pos(self):
        return np.array([self.x, self.y, self.altitude])


def gps2utm(latitude, longitude, altitude, config_ref):
    """
    Projects lat, lon to UTM coordinates
    using the origin (first lat, lon)
    """
    # latitude = lat['latitude']
    # longitude = lng['longitude']
    # altitude = altitude['altitude']
    # status = df_gps['status']

    # base reference system
    lat_ref = config_ref['latitude']
    lon_ref = config_ref['longitude']
    altitude_ref = config_ref['altitude']

    # status_array = np.array(status)
    myProj = Proj(proj='utm', zone='30', ellps='WGS84', datum='WGS84', preserve_units=False,
                  units='m')

    lat = np.array(latitude)
    lon = np.array(longitude)
    altitude = np.array(altitude)

    UTMx_ref, UTMy_ref = myProj(lon_ref, lat_ref)
    UTMx, UTMy = myProj(lon, lat)
    # UTMx = UTMx[idx]
    # UTMy = UTMy[idx]
    UTMx = UTMx - UTMx_ref
    UTMy = UTMy - UTMy_ref
    altitude = altitude - altitude_ref
    # df_gps.insert(2, 'x', UTMx, True)
    # df_gps.insert(2, 'y', UTMy, True)
    # df_gps['x'] = UTMx
    # df_gps['y'] = UTMy
    # df_gps['altitude'] = altitude
    return UTMx, UTMy, altitude
