import numpy as np

np.random.rand(42)

K = 3
N = 200
D = 2

min_convergence = 0.001
dataclasses = np.random.rand(N, D)
clusters = np.random.rand(K, D)
data_clusters = np.zeros(N)

def euclid_distance(a,b):
    total = 0
    for i in range(len(a)):
        total = total + (a[i]-b[i])**2
    return np.sqrt(total)


def get_average(points):
    n = len(points)
    if n == 0:
        return np.zeros(D)
    avg = np.zeros(D)
    for p in points:
        for i in range(D):
            avg[i] = avg[i] + p[i]
    for i in range(D):
        avg[i] = avg[i] / n
    return avg

def assign_clusters(data, clusters, data_clusters):
    for i in range(len(data)):
        min_dist = None
        for k in range(len(clusters)):
            dist = euclid_distance(data[i], clusters[k])
            if min_dist is None or dist < min_dist:
                min_dist = dist
                data_clusters[i] = k
    return data_clusters


def update_clusters(data, clusters, data_clusters):
    for k in range(len(clusters)):
        points = [data[i] for i in range(len(data)) if data_clusters[i] == k]
        if points:
            new_center = get_average(points)
            diff = euclid_distance(new_center, clusters[k])
            if diff > min_convergence:
                clusters[k] = new_center
    return clusters

if __name__ == "__main__":
    iteration = 0
    while True:
        iteration += 1
        old_clusters = clusters.copy()
        data_clusters = assign_clusters(dataclasses, clusters, data_clusters)
        clusters = update_clusters(dataclasses, clusters, data_clusters)
        print(f"Iteration {iteration}")
        print("Clusters:")
        print(clusters)
        diffs = [np.sqrt(np.sum((old_clusters[k] - clusters[k])**2)) for k in range(K)]
        if all(diff<=min_convergence for diff in diffs):
            break


