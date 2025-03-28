import numpy as np
from artelib.homogeneousmatrix import HomogeneousMatrix


class LoopClosing():
    def __init__(self, graphslam, distance_backwards=7, radius_threshold=5.0):
        """
        This class provides functions for loop-closing in a ICP context using LiDAR points.
        Though called DataAssociation, it really provides ways to find out whether the computation of the
        relative transformations between LiDAR pointclouds is correct.
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

    def loop_closing_simple(self, current_index, number_of_candidates_DA, keyframe_manager):
        """
        A simple loop closing procedure. Given the current pose and index:
            a) Find a number of past robot poses inside a radius_threshold.
            b) Chose number_of_candidates randomly.
            c) Compute observations using ICP for each pair of candidates.
            d) Add the observations with add_loop_closing_restrictions.
        Of course, sometimes, the measurement found using ICP may be wrong, thus inducing errors in the map.
        """
        # Data association. Now, add, each cada 15, 20, observaciones (i.e.) 5 metros, ejecutar una asociación de datos
        # Determine if there is loop closure based on the odometry measurement and the previous estimate of the state.
        # find a number of candidates within a radius
        candidates = self.find_candidates()
        print(candidates)
        i = current_index
        n = np.min([len(candidates), number_of_candidates_DA])
        # generating random samples without replacement. sample the candidates up to the max length n randomly
        candidates = np.random.choice(candidates, size=n, replace=False)
        # from the randomly sampled candidates (number_of_candidates_DA), obtain relative transformations
        # and add loop closing transformations
        for j in candidates:
            Tij = self.compute_transformations_between_candidates(i=i, j=j, keyframe_manager=keyframe_manager)
            self.add_loop_closing_observation(i=i, j=j, Tij=Tij)
        return

    def loop_closing_triangle(self, current_index, number_of_triplets_loop_closing, keyframe_manager):
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
        # forming triplets (i, j1, j2) so that j1 and j2 are within d >< distance.
        triplets = self.find_feasible_triplets(current_index=current_index)
        if len(triplets) == 0:
            return
        triplet_indexes = range(len(triplets))
        # sample candidate indexes
        n = np.min([len(triplet_indexes), number_of_triplets_loop_closing])
        # find pairs of close candidates randomly
        triplet_indexes_sampled = np.random.choice(triplet_indexes, size=n, replace=False)
        added_loop_closures = []
        for k in triplet_indexes_sampled:
            print('Checking loop closing triplet: ', triplets[k])
            i = triplets[k][0]
            j1 = triplets[k][1]
            j2 = triplets[k][2]
            Tij1 = self.compute_transformations_between_candidates(i=i, j=j1, keyframe_manager=keyframe_manager)
            Tij2 = self.compute_transformations_between_candidates(i=i, j=j2, keyframe_manager=keyframe_manager)
            Tj1j2 = self.compute_consecutive_transformations(i=j1, j=j2)
            # computing a loop closing t
            I = Tij1*Tj1j2*Tij2.inv()
            print('Found loop closing triplet I: ', I)
            if self.check_distances(I):
                print(10*'#')
                print('FOUND CONSISTENT OBSERVATIONS!')
                print('Adding loop closing observations.')
                print(10 * '#')
                self.add_loop_closing_observation(i=i, j=j1, Tij=Tij1)
                self.add_loop_closing_observation(i=i, j=j2, Tij=Tij2)
                added_loop_closures.append([i, j1])
                added_loop_closures.append([i, j2])
        return added_loop_closures

    def find_feasible_triplets(self, current_index):
        triplets = []
        candidates = self.find_candidates()
        if len(candidates) == 0:
            return []
        print('Found candidates within radius distance threshold:')
        print(candidates)
        i = current_index
        candidates = np.sort(candidates)
        for k in range(len(candidates)):
            # for each j, find
            j1 = candidates[k]
            j2 = self.look_for_valid_indexes(j1, candidates[k:])
            if j2 is not None:
                triplets.append([i, j1, j2])
        return triplets

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

    def look_for_valid_indexes(self, i, rest_of_candidates):
        """
        Find that the indexes are relative (between 1 and 10) and also within a Euclidean distance
        TODO: dindex and deuclidena must be parameters in the graphSLAM class
        """
        for j in range(len(rest_of_candidates)):
            u = rest_of_candidates[j]
            # distance in indexes
            dindex = abs(u - i)
            deuclidean = self.distance(i, u)
            # print('Dindex, dEuclidean: ', dindex, deuclidean)
            if (1 < dindex < 80) and (1.0 < deuclidean < 2.0):
                return u
        return None

    def add_loop_closing_observation(self, i, j, Tij):
        """
        Adds a loop closing restrictions from LiDAR scanmatching to the graph. I.e. and observation of j from i.
        """
        print('Adding loop_closing edge (i, j): ', i, j)
        # Add a binary factor in between two existing states if loop closure is detected.
        self.graphslam.add_edge(Tij, i, j, 'SM')

    def compute_transformations_between_candidates(self, i, j, keyframe_manager):
        """
        Try to compute an observation between the scans at steps i and j in the map.
        The computation is performed considering an initial estimation Tij.
        """
        T0_gps = self.graphslam.T0_gps
        # i = current_index
        # CAUTION: do not add the keyframe to the list: it should have been added before
        # keyframe_manager.add_keyframe(i)
        keyframe_manager.load_pointcloud(i)
        keyframe_manager.pre_process(i)
        # for j in candidates:
        # compute initial error-prone estimation
        Ti = HomogeneousMatrix(self.graphslam.current_estimate.atPose3(i).matrix())
        Tj = HomogeneousMatrix(self.graphslam.current_estimate.atPose3(j).matrix())
        # Correct each estimation by the gps transformation
        # caution, we are estimating the GPS position the robot
        Ti = Ti*T0_gps.inv()
        Tj = Tj * T0_gps.inv()
        # compute the relative transformation between lidars
        # this is the initial estimation between the pointclouds
        Tij = Ti.inv() * Tj
        # compute observation using ICP (change the method)
        keyframe_manager.load_pointcloud(j)
        keyframe_manager.pre_process(j)
        # Caution: the transformation Tijsm is computed from Lidar to Lidar reference frames
        Tijsm = keyframe_manager.compute_transformation(i, j, Tij=Tij)
        # compute the transformation considering the T0_gps transform
        Tijsm = T0_gps.inv()*Tijsm*T0_gps
        # relative_transforms_scanmatcher.append(Tijsm)
        return Tijsm

    def compute_consecutive_transformations(self, i, j):
        """
        Try to compute an observation between the scans at steps i and j in the map.
        The computation is performed considering an initial estimation Tij.
        """
        T0_gps = self.graphslam.T0_gps
        # computing relative transformation from the graphslam current solution
        Ti = HomogeneousMatrix(self.graphslam.current_estimate.atPose3(i).matrix())
        Tj = HomogeneousMatrix(self.graphslam.current_estimate.atPose3(j).matrix())
        # Correct each estimation by the gps transformation
        # caution, we are estimating the GPS position the robot
        Ti = Ti*T0_gps.inv()
        Tj = Tj*T0_gps.inv()
        Tij = Ti.inv()*Tj
        return Tij

    def store_positions(self):
        poses = []
        i = 0
        # fill a vector with all positions
        while self.graphslam.current_estimate.exists(i):
            ce = self.graphslam.current_estimate.atPose3(i)
            T = HomogeneousMatrix(ce.matrix())
            poses.append(T.pos())
            i += 1
        self.positions = np.array(poses)

    def find_candidates(self):
        """
        The function to perform a simple data association.
        The computation of uncertainties is not performed.
        """
        self.store_positions()
        index = self.find_index_backwards()
        candidates = self.find_candidates_within_radius(index)
        return candidates

    def find_index_backwards(self):
        """
        Return the maximum index to which we can try to perform data associations.
        This is represented by a distance
        """
        # current_pose = self.positions[-1, :]
        d = 0
        for i in reversed(range(len(self.positions)-1)):
            posei = self.positions[i]
            posei1 = self.positions[i+1]
            di = np.linalg.norm(posei1-posei)
            d += di
            if d > self.distance_backwards:
                return i
        return None

    def find_candidates_within_radius(self, index):
        """
        Find the distance of the current_pose to all the past poses.
        Alternatively: sum the relative distances of consecutive poses.
        """
        if index is None:
            return []
        current_pose = self.positions[-1, :]
        # compute distance to all positions up to index
        positions = self.positions[0:index, :]
        d = np.linalg.norm(positions - current_pose, axis=1)
        # print('distances:')
        # print(d)
        candidates = np.where(d < self.radius_threshold)[0]
        return candidates

    def distance(self, i, j):
        # if self.graphslam.current_estimate.exists(i):
        ce = self.graphslam.current_estimate.atPose3(i)
        Ti = HomogeneousMatrix(ce.matrix())
        pi = Ti.pos()
        # else:
        #     return 1000.0
        # if self.graphslam.current_estimate.exists(j):
        ce = self.graphslam.current_estimate.atPose3(j)
        Tj = HomogeneousMatrix(ce.matrix())
        pj = Tj.pos()
        # else:
        #     return 1000.0
        return np.linalg.norm(pi-pj)


