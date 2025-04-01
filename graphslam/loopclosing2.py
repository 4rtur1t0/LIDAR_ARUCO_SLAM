# from artelib.homogeneousmatrix import HomogeneousMatrix
import numpy as np
from scipy.spatial import KDTree


class LoopClosing2():
    def __init__(self, graphslam, distance_backwards=7, radius_threshold=5.0):
        """
        This class provides functions for loop-closing in a ICP context using LiDAR points.
        Though called DataAssociation, it really provides ways to find out whether the computation of the
        relative transformations between LiDAR pointclouds is correct.
        Actually, data associations are performed in a lazy manner: if we find a pose near the pose i in the current
        estimation, we will try to compute a transformation between the two pointclouds.
        The method loop_closing simple, for a given observation at time i:
        a) finds other robot poses j within a radius_threshold with its corresponding pointcloud.
        b) Computes the observed transformation from i to j using both pointclouds and an ICP-based algorithm.
        c) All observations Tij are added as edges to graphslam.
        The method loop_closing triangle is far more accurate, for a given observation at time i:
        a) finds other robot poses j within a radius_threshold with its corresponding pointcloud.
        b) finds triplets of poses (i, j1, j2) considering that the indexes j1 and j2 must be close (i.e. close in time).
           Also, the indexes j1, j2 be separated a distance d1 in space and be below a distance d2.
        c) For each triplet, it must be: Tij1*Tj1j2*Tj2i=Tii=I, the identity. Due to errors, I must be different from the identity
        I is then converted to I.pos() and I.euler() and checked to find p = 0, and abg=0 approximately. If the transformation
        differs from I, the observations are discarded. On the contrary, both Tij1 and Tij2 are added to the graph.
        """
        self.graphslam = graphslam
        # look for data associations that are delta_index back in time
        self.distance_backwards = distance_backwards
        self.radius_threshold = radius_threshold
        self.positions = None

    def find_feasible_triplets(self):
        """
        Find loop closure candidates across the entire trajectory.

        For each pose i:
            - Find a pose j with distance R:  min_travel_distance <  R(i, j) < max_travel_distance.
            - Find a pose k so that: min_travel_distance < R(i,k) < max_travel_distance
                                     and min_travel_distance < R(j,k) < max_travel_distance

        Args:
            poses (np.ndarray): Nx3 or Nx2 array of poses (x, y, theta optional).
            radius (float): Distance threshold for potential loop closures.
            min_distance (float): Minimum travel distance before considering closure.

        Returns:
            dict: Dictionary where keys are indices and values are lists of loop closure candidates.
        """
        positions = self.graphslam.get_solution_positions()
        if len(positions) == 0:
            return None
        travel_distances = self.compute_travel_distances(positions)
        # Build KDTree for fast spatial queries
        tree = KDTree(positions[:, :2])  # Use (x, y) positions
        triplets_global = []
        step_index = 20
        # min_diff_index = 500
        # for each i find j. The index j is found randomly
        for i in range(0, len(positions), step_index):
            # find an index j close to i (within r_min and r_max) and close in the sequence index
            triplet = self.find_j_k_within_radii(tree=tree, positions=positions, travel_distances=travel_distances,
                                                 i=i)
            if triplet is not None:
                triplets_global.append(triplet)
        # flatten
        triplets_global = [item for sublist in triplets_global for item in sublist]
        return triplets_global

    def compute_travel_distances(self, positions):
        """
        Compute cumulative travel distance for each pose
        """
        distances = np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1)
        return np.insert(np.cumsum(distances), 0, 0)

    def find_j_k_within_radii(self, tree, positions, travel_distances, i):
        """
        Finds a candidate at a distance  r_min < d < r_max
        It should be at a close index, so that the error in poses from i to j is low
        """
        # find candidates for j within r_close. The index j must be
        r1 = 0.7
        r2 = 1.5 # this is the actual loop closing distance
        # find candidates for long loopclosing. Find candidates within r_lc that have travelled more than r_travelled
        r_lc = 3.0
        r_traveled = 3.0
        num_triplets = 5
        # for clarity, we ask the tree for candidates
        neighbors_in_r2 = tree.query_ball_point(positions[i, :2], r2)
        j_n = None
        # select only a first close candidate j within a distance of r1 and travel distance within r1 and r2
        # this candidate should be close in the sequence of observations (odometry, sm)
        for j in neighbors_in_r2:
            d = travel_distances[j] - travel_distances[i]
            if j > i and (d > r1) and (d < r2):
                j_n = j
                break
        # select another candidate with a travel distance larger than r3
        neighbors_in_r_lc = tree.query_ball_point(positions[i, :2], r_lc)
        k_n = []
        # obtain k for a long travel distance (try to perform long loop closings)
        for k in neighbors_in_r_lc:
            d = travel_distances[k] - travel_distances[i]
            if k > i and (d > r_traveled):
                k_n.append(k)
        num_triplets = min(num_triplets, len(k_n))
        # this is a uniform choice. IDEA: could try to include longer loop closings with more probability.
        # however, closer loop closings are also beneficial in this particular case
        k_n = np.random.choice(k_n, num_triplets, replace=False)
        result_triplets = []
        for k in k_n:
            if j_n is not None and k is not None:
                result_triplets.append([i, j_n, k])
                # return [i, j_n, k_n]
        return result_triplets

    def check_triplet_transforms(self, graphslam, triplet_transforms):
        """
        A better loop closing procedure. Given the current pose and index i (current_index):
                a) Find a number of past robot poses inside a radius_threshold.
                b) Chose a candidate j randomly. Find another candidate k. The distance in the indexes in j and k < d_index
                c) Compute observations using ICP for Tij and Tik.
                d) Compute Tij*Tjk*(Tik)^(-1)=I, find the error in position and orientation in I to filter the validity
                   of Tij and Tik. Tjk should be low in uncertainty, since it depends on consecutive observations.
                e) Add the observations with add_loop_closing_restrictions.
        Still, of course, sometimes, the measurement found using ICP may be wrong, in this case, it is less probable that
         both Tij and Tik have errors that can cancel each other. As a result, this is a nice manner to filter out observations.
        """
        result_triplet_transforms = []
        for triplet_transform in triplet_transforms:
            print('Checking loop closing triplet: ', triplet_transform)
            # the transformation computed from the LiDAR using ICP. Caution: expressed in the LiDAR ref frame.
            Tij = triplet_transform[3]
            Tjk = triplet_transform[4]
            Tik = triplet_transform[5]
            # compute circle transformation
            I = Tij * Tjk * Tik.inv()
            print('Found loop closing triplet I: ', I)
            if self.check_distances(I):
                print(10*'#')
                print('FOUND CONSISTENT OBSERVATIONS!')
                print('Adding loop closing observations to list!.')
                print(10 * '#')
                # result_triplet_transforms.append([i, j, k, Tij, Tjk, Tik])
                result_triplet_transforms.append(triplet_transform)
        return result_triplet_transforms

    def check_distances(self, I):
        dp = np.linalg.norm(I.pos())
        da1 = np.linalg.norm(I.euler()[0].abg)
        da2 = np.linalg.norm(I.euler()[1].abg)
        da = min([da1, da2])
        print('Found triangle loop closing distances: ', dp, da)
        if dp < 0.1 and da < 0.05:
            print('I is OK')
            return True
        print('FOUND INCONSISTENT LOOP CLOSING TRIPLET: DISCARDING!!!!!!!')
        return False

