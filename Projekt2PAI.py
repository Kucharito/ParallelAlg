import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

def mean_shift_step(args):
    data, point, R = args
    distances = np.linalg.norm(data - point, axis=1)
    neighbors = data[distances < R]
    if len(neighbors) == 0:
        return point
    return neighbors.mean(axis=0)

def mean_shift(data, start_point, R, threshold):
    current_point = start_point.copy()
    while True:
        distances = np.linalg.norm(data - current_point, axis=1)
        neighbors = data[distances < R]
        if len(neighbors) == 0:
            break
        new_point = neighbors.mean(axis=0)
        if np.linalg.norm(new_point - current_point) < threshold:
            break
        current_point = new_point
    return current_point

def find_clusters(final_positions, tolerance=2.0):
    clusters, labels = [], np.full(len(final_positions), -1)
    for i, p in enumerate(final_positions):
        for c_idx, center in enumerate(clusters):
            if np.linalg.norm(p - center) < tolerance:
                labels[i] = c_idx
                break
        else:
            clusters.append(p)
            labels[i] = len(clusters) - 1
    return np.array(clusters), labels

def main():
    N, R, threshold, workers = 500, 30, 1e-4, 5
    np.random.seed(42)
    data = np.random.rand(N, 2) * 100
    original_data = data.copy()

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 8))
    scat = ax.scatter(data[:, 0], data[:, 1], s=15, c='blue')
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.set_title("Mean Shift - štart")
    plt.show()
    plt.pause(1)

    # Každý bod konverguje samostatne (paralelne)
    args = [(original_data, p, R, threshold) for p in original_data]
    with Pool(workers) as pool:
        final_positions = np.array(pool.starmap(mean_shift, args, chunksize=N//workers))

    # Animácia - postupné zobrazenie konvergencie
    for step in range(20):
        t = (step + 1) / 20
        interpolated = original_data + t * (final_positions - original_data)
        scat.set_offsets(interpolated)
        ax.set_title(f"Mean Shift - konvergencia {int(t*100)}%")
        plt.draw()
        plt.pause(0.1)

    plt.ioff()
    plt.close()

    centers, labels = find_clusters(final_positions, tolerance=2.0)
    print(f"Počet clusterov: {len(centers)}")

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(original_data[:, 0], original_data[:, 1], s=15, c='blue', alpha=0.5)
    plt.title("Pôvodné dáta")
    plt.xlim(-5, 105)
    plt.ylim(-5, 105)

    plt.subplot(1, 2, 2)
    colors = plt.cm.tab20(np.linspace(0, 1, len(centers)))
    for cluster_id in range(len(centers)):
        mask = labels == cluster_id
        plt.scatter(original_data[mask, 0], original_data[mask, 1], s=50, c=[colors[cluster_id]], alpha=0.7)
    plt.scatter(centers[:, 0], centers[:, 1], c='gray', s=150, marker='o')
    plt.title(f"Mean Shift (R={R}) - {len(centers)} clusterov")
    plt.xlim(-5, 105)
    plt.ylim(-5, 105)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("TEST: Závislosť počtu clusterov od R")
    for R_test in [5, 10, 15, 20]:
        np.random.seed(42)
        data = np.random.rand(500, 2) * 100
        with Pool(5) as pool:
            results = np.array(pool.starmap(mean_shift, [(data, p, R_test, 1e-4) for p in data]))
        centers, _ = find_clusters(results)
        print(f"  R = {R_test:2d} -> {len(centers):3d} clusterov")
    print("Väčšie R = menej clusterov\n")
    main()
