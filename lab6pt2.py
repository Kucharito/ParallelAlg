import numpy as np
import math
np.random.seed(42)

K = 3 #cluster count
N = 500 #points count
D = 1000 #dimension
min_convergence = 0.001
data_clusters = np.zeros(N)

data = np.empty((0,D))
for i in range(5):
    np.random.seed(i*10)
    data = np.vstack([data, np.random.randn(N//5,D) * 0.3 + np.random.rand(1,D) * 5])

np.random.seed(100)
clusters = np.random.rand(K,D)

def euclid_distance(a, b):
    total = 0
    for i in range(len(a)):
        total += (a[i]-b[i])**2
    return math.sqrt(total)

def get_average(points):
    n = len(points)
    if n == 0:
        return np.zeros(D)
    avg = np.zeros(D)
    for p in points:
        for i in range(D):
            avg[i] += p[i]
    for i in range(D):
        avg[i] /= n
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
        data_clusters = assign_clusters(data, clusters, data_clusters)
        clusters = update_clusters(data, clusters, data_clusters)
        print(f"Iteration {iteration}, clusters: {clusters}")
        diffs = [np.sqrt(np.sum((old_clusters[k]-clusters[k])**2)) for k in range(len(clusters))]
        if all(diff <= min_convergence for diff in diffs):
            break


    #count shilouete indexes for each cluster
    silhouette_indexes = np.zeros(K)
    for k in range(K):
        #a = average distance to points in same cluster
        #b = distance between centroid I and nearest centroid
        a = 0
        b = float('inf')
        points = np.array([data[i] for i in range(len(data)) if data_clusters[i] == k])
        if points.size > 1:
            for p in points:
                a += euclid_distance(p, clusters[k])
            a /= len(points)
        else:
            a = 0

        for j in range(K):
            if j != k:
                b_temp = euclid_distance(clusters[k], clusters[j])
                if b_temp < b:
                    b = b_temp
        if max(a,b) > 0:
            s = (b-a)/max(a,b)
        else:
            s = 0
        print(f"Cluster {k}: a={a}, b={b}, s={s}")




    try:
        import matplotlib.pyplot as plt
        colors = ['r', 'g', 'b', 'y', 'c', 'm']
        for k in range(len(clusters)):
            points = np.array([data[i] for i in range(len(data)) if data_clusters[i] == k])
            if points.size > 0:
                plt.scatter(points[:,0], points[:,1], c=colors[k%len(colors)], label=f'Cluster {k}')
            plt.scatter(clusters[k][0], clusters[k][1], c='k', marker='x', s=100, label=f'Center {k}' if k==0 else "")
        plt.title('K-Means Clustering')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.legend()
        plt.show()
    except ImportError:
        print("matplotlib not installed, skipping visualization")


