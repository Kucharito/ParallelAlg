import threading
import time
from concurrent.futures import ThreadPoolExecutor
import concurrent
import numpy as np
import math
import matplotlib.pyplot as plt
import multiprocessing as mp

def square(num):
    print(f"Square: {num*num}")
    #print thread number
    print(f"Thread ID (square): {threading.get_ident()}")

def cube(num):
    print(f"Cube: {num*num*num}")
    print(f"Thread ID (cube): {threading.get_ident()}")

np.random.seed(42)

N = 15000
D = 100
K = 5
min_convergence = 0.001
data = np.empty((0,D))
data_clusters = np.zeros(N)

for i in range(5):
    np.random.seed(i*10)
    data = np.vstack([data, np.random.randn(N//5,D) * 0.3 + np.random.rand(1,D) * 5])

np.random.seed(10)
clusters = np.random.rand(K,D)

def euclid_distance(a, b):
    total = 0
    for i in range(len(a)):
        total += (a[i]-b[i])**2
    return math.sqrt(total)

def get_center(points):
    n = len(points)
    center = np.zeros(D)
    if n == 0:
        return center
    for p in points:
        for i in range(D):
            center[i] += p[i]
    for i in range(D):
        center[i] /= n
    return center

def assign_clusters(data, clusters, data_clusters):
    #print current process
    print(f"Process ID (assign_clusters): {threading.get_ident()}, processing {len(data)} points")
    start_time = time.time()
    for i in range(len(data)):
        min_dist = None
        for k in range(len(clusters)):
            dist = euclid_distance(data[i],clusters[k]) #np.sqrt(np.sum((data[i]-clusters[k])**2))
            if min_dist is None or dist < min_dist:
                min_dist = dist
                data_clusters[i] = k
    end_time = time.time()
    print(f"Process ID (assign_clusters): {threading.get_ident()}, time taken: {end_time - start_time:.4f} seconds")
    return data_clusters

def update_clusters(data, clusters, data_clusters):
    for k in range(len(clusters)):
        points = [data[i] for i in range(len(data)) if data_clusters[i] == k]
        if points:
            new_center = get_center(points) #np.mean(points, axis=0)
            diff = euclid_distance(new_center, clusters[k]) #np.sqrt(np.sum((new_center - clusters[k])**2))
            if diff > min_convergence:
                clusters[k] = new_center
    return clusters

if __name__ == "__main__":
    data_clusters = assign_clusters(data, clusters, data_clusters)
    iteration = 0
    start_time = time.time()
    while True:
        iteration += 1
        old_clusters = clusters.copy()
        thread_count = 20
        if True:
            with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor:
                # Split data into chunks for each process
                chunk_size = len(data) // thread_count
                data_chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
                all_data = []
                arguments = [(chunk, clusters, np.zeros(len(chunk))) for chunk in data_chunks]
                for result in executor.map(lambda p: assign_clusters(*p), arguments):
                    all_data.append(result)
                data_clusters = np.concatenate(all_data)
        else:
            data_clusters = assign_clusters(data, clusters, data_clusters)

        clusters = update_clusters(data, clusters, data_clusters)
        print(f"Iteration {iteration} done")
        diffs = [np.sqrt(np.sum((old_clusters[k]-clusters[k])**2)) for k in range(len(clusters))]
        if all(diff <= min_convergence for diff in diffs):
            break

    end_time = time.time()
    print(f"K-means converged in {iteration} iterations, time taken: {end_time - start_time:.4f} seconds")