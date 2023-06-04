import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import chisquare

class FairKMeans:
    def __init__(self, n_clusters=3, max_iter=300, random_state=0, tol=1e-4, sensitive_attr=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.tol = tol
        self.kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter, random_state=random_state, tol=tol)
        self.sensitive_attr = sensitive_attr

    def fit(self, X):
        self.kmeans.fit(X)
        self.centroids = self.kmeans.cluster_centers_
        self.labels = np.zeros(X.shape[0], dtype=np.int32)
        self.fairness_violations = np.zeros(self.n_clusters)
        self.cluster_counts = np.zeros(self.n_clusters)
        self.cluster_positive_counts = np.zeros(self.n_clusters)

        for _ in range(self.max_iter):
            old_labels = self.labels.copy()
            distances = np.sum((X[:, np.newaxis] - self.centroids) ** 2, axis=2)
            sorted_cluster_indices = np.argsort(distances, axis=1)

            for i in range(X.shape[0]):
                for j in range(self.n_clusters):
                    candidate_cluster = sorted_cluster_indices[i, j]
                    if self.can_assign_to_cluster(i, candidate_cluster):
                        self.assign_to_cluster(i, candidate_cluster)
                        break

            if np.all(old_labels == self.labels):
                break

    def predict(self, X):
        return self.labels

    def can_assign_to_cluster(self, i, cluster):
        if self.sensitive_attr[i] == 1:
            if (self.cluster_positive_counts[cluster] + 1) / (self.cluster_counts[cluster] + 1) > (self.sensitive_attr.sum() + 1) / (self.labels.shape[0] + 1):
                return False
        else:
            if (self.cluster_counts[cluster] + 1 - self.cluster_positive_counts[cluster]) / (self.cluster_counts[cluster] + 1) > (self.labels.shape[0] - self.sensitive_attr.sum() + 1) / (self.labels.shape[0] + 1):
                return False
        return True

    def assign_to_cluster(self, i, cluster):
        if self.labels[i] != -1:
            self.cluster_counts[self.labels[i]] -= 1
            self.cluster_positive_counts[self.labels[i]] -= self.sensitive_attr[i]
        self.labels[i] = cluster
        self.cluster_counts[cluster] += 1
        self.cluster_positive_counts[cluster] += self.sensitive_attr[i]
    
    def calculate_fairness_score(cluster_sensitive_attr, total_sensitive_attr):
        """
        Calculate a fairness score for a cluster using Chi-square test.
        
        Args:
        cluster_sensitive_attr: np.array of sensitive attribute values in the cluster
        total_sensitive_attr: np.array of sensitive attribute values in the entire dataset
        
        Returns:
        p-value of Chi-square test. Higher means the cluster's distribution is closer to the total distribution.
        """
        cluster_counts = np.bincount(cluster_sensitive_attr)
        total_counts = np.bincount(total_sensitive_attr)
        
        # We normalize counts to get probabilities
        cluster_probs = cluster_counts / cluster_counts.sum()
        total_probs = total_counts / total_counts.sum()
        
        # Make sure that the two arrays have the same size
        size_diff = total_probs.size - cluster_probs.size
        if size_diff > 0:
            cluster_probs = np.pad(cluster_probs, (0, size_diff))
        elif size_diff < 0:
            total_probs = np.pad(total_probs, (0, -size_diff))
            
        _, p_value = chisquare(cluster_probs, total_probs)
        return p_value