from sklearn.cluster import SpectralClustering

class RecursiveSpectralClustering:
    def __init__(self, n_clusters=8, affinity="rbf", assign_labels="kmeans", n_neighbors=10):
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.assign_labels = assign_labels
        self.n_neighbors = n_neighbors
        self.labels_ = None

    def __get_clusters(self, adjacency_matrix):
        clustering = SpectralClustering(n_clusters=2, affinity="precomputed", assign_labels=self.assign_labels)
        clustering.fit(adjacency_matrix)
        clusters = {}
        for i in range(len(clustering.labels_)):
            if clustering.labels_[i] not in clusters:
                clusters[clustering.labels_[i]] = []
            clusters[clustering.labels_[i]].append(i)
        return list(clusters.values())

    def __recursive_clustering(self, n, k, X, A, indices):
        if k <= 1:
            return [indices]
        m = round(n/k)
        temp_clusters = self.__get_clusters(X)
        L = len(temp_clusters[0])
        R = len(temp_clusters[1])
        if L%m == 0 or R%m == 0:
            indices_L = temp_clusters[0]
            indices_R = temp_clusters[1]
        else:
            if L > R:
                total_scores = {}
                for i in temp_clusters[0]:
                    total_scores[i] = 0
                    for j in temp_clusters[1]:
                        total_scores[i] += X[i][j] + X[j][i]
                sorted_scores = sorted(total_scores.items(), key=lambda x: x[1], reverse=True)
                indices_L = [i for i, _ in sorted_scores[L%m:]]
                indices_R = temp_clusters[1] + [i for i, _ in sorted_scores[:L%m]]
            else:
                total_scores = {}
                for i in temp_clusters[1]:
                    total_scores[i] = 0
                    for j in temp_clusters[0]:
                        total_scores[i] += X[i][j] + X[j][i]
                sorted_scores = sorted(total_scores.items(), key=lambda x: x[1], reverse=True)
                indices_L = temp_clusters[0] + [i for i, _ in sorted_scores[:R%m]]
                indices_R = [i for i, _ in sorted_scores[R%m:]]
        k1 = min(k-1, round(len(indices_L)/(len(indices_L) + len(indices_R))*k))
        k2 = k - k1
        P = X[indices_L][:, indices_L]
        Q = X[indices_R][:, indices_R]
        temp_clusters_L = self.__recursive_clustering(P.shape[0], k1, P, A, [indices[i] for i in indices_L])
        temp_clusters_R = self.__recursive_clustering(Q.shape[0], k2, Q, A, [indices[i] for i in indices_R])
        clusters = temp_clusters_L + temp_clusters_R
        return clusters

    def fit(self, X):
        if X.shape[0] < self.n_clusters:
            return Exception("Number of clusters greater than number of data points")
        if self.affinity in ["rbf"]:
            X = SpectralClustering(n_clusters=1, affinity=self.affinity).fit(X).affinity_matrix_
        elif self.affinity in ["nearest_neighbors", "precomputed_nearest_neighbors"]:
            X = SpectralClustering(n_clusters=1, affinity=self.affinity, n_neighbors=self.n_neighbors).fit(X).affinity_matrix_.toarray()
        n = X.shape[0]
        k = self.n_clusters
        clusters = self.__recursive_clustering(n, k, X, X, list(range(n)))
        labels = [0]*n
        label = 0
        for c in clusters:
            for i in c:
                labels[i] = label
            label += 1
        self.labels_ = labels
        return labels
