"""
Module containing the Spectral Equal Size Clustering method adapted for affinity matrix.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering
import logging
import math

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


class SpectralEqualSizeClustering:
    """
    Uses spectral clustering to obtain an initial configuration of clusters.
    This configuration is compact but NOT equal-sized. To make clusters equal-sized (in number of points),
    we use the method cluster_equalization().
    Input parameters:
        nclusters (int): number of clusters
        nneighbors (int): number of neighbors. Used by the spectral clustering to
                          construct the affinity matrix. Good values are between 7% and 15%
                          of the dataset points.
        equity_fraction (float): Equity fraction. Value in range (0,1] which decides how equal the clusters
                           could be. The higher the fraction, the more equal the clusters BUT the less
                           compact.
        seed (int): Random seed generator.

    Attributes:
        first_clustering (data frame): Table containing the cluster labels of each point in the initialisation.
        first_cluster_dispersion (data frame): A table with indexes corresponding to each cluster and a column
                                containing the dispersion in distance of each cluster.
        first_total_cluster_dispersion (float): sum of first_cluster_dispersion
        final_clustering  (data frame): Table containing the cluster labels of each point after the balancing
                                        of the clusters in size.
        final_cluster_dispersion (data frame): A table with indexes corresponding to each cluster and a column
                                containing the dispersion in distance of each cluster (after the balancing in size).
        total_cluster_dispersion (float): sum of final_cluster_dispersion.
                                 This attribute can be used as a metric to optimise the cluster hyperparameters.


    How to use this class:
    cl = SpectralEqualSizeClustering(nclusters=2, nneighbors=100, equity_fraction=0.5, seed=11362)
    cluster_labels = cl.fit(affinity_matrix)
    """

    def __init__(self, nclusters: int = None, nneighbors: int = None, equity_fraction=0.3, seed=None):
        self.nclusters = nclusters
        self.equity_fr = equity_fraction
        self.nneighbors = nneighbors
        self.seed = seed

        self.first_clustering = None
        self.first_cluster_dispersion = None  # A table with each cluster dispersion (in distance)
        self.first_total_cluster_dispersion = None  # total cluster dispersion.

        self.range_points = None
        self.nn_df = None  # table of number of neighbors per point
        self.cneighbors = None  # Dictionary of cluster neighbors

        # Final results after equalization of clusters
        self.final_clustering = None
        self.final_cluster_dispersion = None
        self.total_cluster_dispersion = None

    @staticmethod
    def _cluster_dispersion(affinity_matrix, clusters):
        """
        Function that computes the cluster dispersion. The cluster dispersion is defined
        as the standard deviation in distance of all the elements within a cluster. The sum of the cluster dispersion
        of all the clusters in a dataset is called the total cluster dispersion. The lower the cluster dispersion,
        the more compact the clusters are.
        Inputs:
        affinity_matrix: numpy array of the affinity matrix
        clusters: table with cluster labels of each event. columns: 'label', index: points
        """

        if "label" not in clusters.columns:
            raise ValueError("Table of clusters does not have 'label' column.")

        def std_affinity(points, am):
            # Modified to handle affinity matrix
            affinities = am[np.ix_(points, points)]
            cdispersion = np.nanstd(affinities)
            return cdispersion

        nclusters = clusters["label"].nunique()
        points_per_cluster = [list(clusters[clusters.label == cluster].index) for cluster in range(nclusters)]
        wcsaffinity = [std_affinity(points_per_cluster[cluster], affinity_matrix) for cluster in range(nclusters)]
        cluster_dispersion_df = pd.DataFrame(wcsaffinity, index=np.arange(nclusters), columns=["cdispersion"])
        return cluster_dispersion_df

    @staticmethod
    def _optimal_cluster_sizes(nclusters, npoints):
        """
        Gives the optimal number of points in each cluster.
        For instance,  if we have 11 points, and we want 3 clusters,
        2 clusters will have 4 points and one cluster, 3.
        """
        min_points, max_points = math.floor(npoints / float(nclusters)), math.floor(npoints / float(nclusters)) + 1
        number_clusters_with_max_points = npoints % nclusters
        number_clusters_with_min_points = nclusters - number_clusters_with_max_points

        list1 = list(max_points * np.ones(number_clusters_with_max_points).astype(int))
        list2 = list(min_points * np.ones(number_clusters_with_min_points).astype(int))
        return list1 + list2

    @staticmethod
    def get_nneighbours_per_point(am, nneighbors):
        """
        Computes the number of neighbours of each point.
        IMPORTANT:  I do not consider the point itself as neighbour.
                    This assumption is important so don't change it!
        """
        npoints = am.shape[0]
        nn_data = [[p, list(pd.Series(am[:, p]).sort_values(ascending=False).index[1: nneighbors])] for p in range(0, npoints)]
        nn_data = pd.DataFrame(nn_data, columns=["index", "nn"]).set_index("index")
        return nn_data

    def _get_cluster_neighbors(self, df):
        """
        Function to find the cluster neighbors of each cluster.
        The cluster neighbors are selected based on a smaller number of neighbours
        because I don't want to get no neighboring clusters.
        The minimum number of nn to get cluster neighbors is 30. This choice is arbitrary.
        Inputs:
            df: a table with points as index and a "label" column
        Returns:
            A dictionary of shape: {i: [neighbor clusters]}, i= 0,..,nclusters
        """
        if self.nn_df is None:
            raise Exception(
                "Nearest neighbour table not found. Use self.get_nneighbours_per_point(affinity_matrix, nneighbors)")

        def cluster_neighbor_for_point(nn_list, nneighbours):
            nn_labels = df1.loc[nn_list[0:nneighbours], 'label']
            return np.unique(nn_labels)

        df1 = df.copy()
        df1 = pd.merge(df1, self.nn_df, left_index=True, right_index=True)
        nn = min(30, self.nneighbors)  # nearest neighbours to compute border points: this def is arbitrary
        df1["unique_clusters"] = df1.apply(lambda row: cluster_neighbor_for_point(row["nn"], nn), axis=1)

        temp = df1[["unique_clusters"]]
        # get neighbor clusters (remove own cluster)
        neighbors = {}
        for c in range(self.nclusters):
            points_in_cluster = df1.label == c
            neighbors_in_cluster = temp.loc[points_in_cluster, "unique_clusters"].to_list()
            neighbors[c] = {i for l in neighbors_in_cluster for i in l if i != c}
        return neighbors

    @staticmethod
    def _get_clusters_outside_range(clustering, minr, maxr):
        """
        Function to get clusters outside the min_range, max_range
        Input: clustering: table with idx as points, and a "label" column
        """
        csizes = clustering.label.value_counts().reset_index()
        csizes.columns = ["cluster", "npoints"]

        large_c = list(csizes[csizes.npoints > maxr]["cluster"].values)
        small_c = list(csizes[csizes.npoints < minr]["cluster"].values)

        return large_c, small_c

    @staticmethod
    def _get_no_large_clusters(clustering, maxr):
        """
        Function to get clusters smaller than max_range
        Input: clustering: table with idx as points, and a "label" column
        """
        csizes = clustering.label.value_counts().reset_index()
        csizes.columns = ["cluster", "npoints"]

        return list(csizes[(csizes.npoints < maxr)]["cluster"].values)

    @staticmethod
    def _get_points_to_switch(am, cl_elements, clusters_to_modify, idxc):
        """
        Function to obtain the closest distance of points in cl_elements with respect to the clusters in
        clusters_to_modify
        Inputs:
            am: affinity matrix
            cl_elements: list of points of the cluster(s) that give points
            cluster_to_modify: a list of labels of clusters that receive points.
            idxc: dictionary with keys
