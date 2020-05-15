"""
Script to implement the K-means algorithm

How to run
"""

import numpy as np

np.random.seed(42)

def euclidean_distance(value1, value2):
    """
            Parameters
            -----------
                value1 - first point
                value2 - second point

            Return
            --------
                Euclidean_distance between the two points

        """
    return np.sqrt(np.sum((x1 - x2)**2))




class KMeans():
    """
        
        Parameters
        ------------
            k = number
            max_iters = maximum number of iteration

        Methods
        ---------
            fit(self, X, y) = use to fit / train the model
            predict(self, X, y) = It is used to predict
            _predict(self, x) = Its a helper function for the method predict(self, X)

    """

    def __init__(self, K=5, max_iters=100):
        self.K = K
        self.max_iters = max_iters

        # list of sample indices for each cluster
        self.clusters = [[] for _ in range(self.K)]
        # the centers (mean feature vector) for each cluster
        self.centroids = []

    def predict(self, X):

        """
            Parameters
            -----------
                X - Input values

            Return
            --------
                a list labels of clusters

        """
        self.X = X
        self.n_samples, self.n_features = X.shape
        
        # initialize 
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]

        # Optimize clusters
        for _ in range(self.max_iters):
            # Assign samples to closest centroids (create clusters)
            self.clusters = self._create_clusters(self.centroids)

            # Calculate new centroids from the clusters
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)
            
            # check if clusters have changed
            if self._is_converged(centroids_old, self.centroids):
                break

        # Classify samples as the index of their clusters
        return self._get_cluster_labels(self.clusters)


    def _get_cluster_labels(self, clusters):
        """
            Helper function
            Parameters
            -----------
                clusters - 

            Return
            --------
                labels

        """
        # each sample will get the label of the cluster it was assigned to
        labels = np.empty(self.n_samples)

        for cluster_idx, cluster in enumerate(clusters):
            for sample_index in cluster:
                labels[sample_index] = cluster_idx
        return labels

    def _create_clusters(self, centroids):
        """
            Parameters
            -----------
                centroids - the centers of clusters

            Return
            --------
                clusters

        """

        # Assign the samples to the closest centroids to create clusters
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        """
            Parameters
            -----------
                sample - a single input value
                centroids - the centers of every cluster

            Return
            --------
                index of closest centroid

        """
        # distance of the current sample to each centroid
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_index = np.argmin(distances)
        return closest_index

    def _get_centroids(self, clusters):
        # assign mean value of clusters to centroids
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _is_converged(self, old, centroids):
        """
            Parameters
            -----------
                 old - old centroids
                 centroids - new centroids

            Return
            --------
                boolean value = either is has converged or not

        """

        # distances between each old and new centroids, fol all centroids
        distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)]
        return sum(distances) == 0


