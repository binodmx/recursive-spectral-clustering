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
        l_ids = temp_clusters[0]
        r_ids = temp_clusters[1]
        l = len(l_ids)
        r = len(r_ids)
        if l%m != 0 and r%m != 0:
            if l > r:
                total_scores = {}
                for i in l_ids:
                    total_scores[i] = 0
                    for j in r_ids:
                        total_scores[i] += X[i][j] + X[j][i]
                sorted_scores = sorted(total_scores.items(), key=lambda x: x[1], reverse=True)
                l_ids = [i for i, _ in sorted_scores[l%m:]]
                r_ids = r_ids + [i for i, _ in sorted_scores[:l%m]]
            else:
                total_scores = {}
                for i in r_ids:
                    total_scores[i] = 0
                    for j in l_ids:
                        total_scores[i] += X[i][j] + X[j][i]
                sorted_scores = sorted(total_scores.items(), key=lambda x: x[1], reverse=True)
                l_ids = l_ids + [i for i, _ in sorted_scores[:r%m]]
                r_ids = [i for i, _ in sorted_scores[r%m:]]
        l_k = round(len(l_ids)/(len(l_ids) + len(r_ids))*k)
        r_k = k - l_k
        l_X = X[l_ids][:, l_ids]
        r_X = X[r_ids][:, r_ids]
        l_clusters = self.__recursive_clustering(l_X.shape[0], l_k, l_X, A, [indices[i] for i in l_ids])
        r_clusters = self.__recursive_clustering(r_X.shape[0], r_k, r_X, A, [indices[i] for i in r_ids])
        clusters = l_clusters + r_clusters
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
