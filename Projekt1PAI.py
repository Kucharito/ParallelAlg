
import multiprocessing
import time
from typing import List, Tuple

def load_data(path: str):
    with open(path) as f:
        lines = f.readlines()
        number_of_items = int(lines[0])
        widths = [int(x) for x in lines[1].split()]
        data = [[float(part) for part in line.split()] 
                for line in lines[2:2 + number_of_items]]

    n = len(data)
    for i in range(n):
        for j in range(i + 1, n):
            data[j][i] = data[i][j]

    return widths, data

def cost(permutation, widths, data):
    n = len(permutation)
    centers = []
    position = 0
    for i in permutation:
        position += widths[i]
        centers.append(position - widths[i] / 2)
    
    total_cost = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            total_cost += data[permutation[i]][permutation[j]] * abs(centers[j] - centers[i])
    return total_cost


def partial_cost(partial_perm, widths, data):
    n = len(partial_perm)
    if n < 2:
        return 0.0
    
    # Vypočítaj centrá už umiestnených zariadení
    centers = []
    position = 0
    for i in partial_perm:
        position += widths[i]
        centers.append(position - widths[i] / 2)

    # Spočítaj cenu len medzi zariadeniami v partial_perm
    pcost = 0.0
    for i in range(n):
        pi = partial_perm[i]
        ci = centers[i]
        for j in range(i + 1, n):
            pcost += data[pi][partial_perm[j]] * abs(centers[j] - ci)
    return pcost


def lower_bound(partial_perm, widths, data):
    n = len(data)
    
    if len(partial_perm) >= n - 1:
        return partial_cost(partial_perm, widths, data)
    
    # Cena umiestnených zariadení
    lb = partial_cost(partial_perm, widths, data)
    
    # Hrubý dolný odhad pre zostávajúce zariadenia
    placed_set = set(partial_perm)
    remaining = [i for i in range(n) if i not in placed_set]
    
    if len(remaining) >= 2:
        min_width = min(widths[i] for i in remaining)
        max_data = max(data[i][j] for i in remaining for j in remaining if i != j)
        lb += max_data * min_width * len(remaining)

    return lb


def branch_and_bound(partial_perm, widths, data, best_cost, best_perm):
    n = len(data)
    
    if len(partial_perm) == n:
        c = cost(partial_perm, widths, data)
        if c < best_cost[0]:
            best_cost[0] = c
            best_perm[0] = partial_perm.copy()
            print(f"New best cost: {best_cost[0]:.2f} -> perm {best_perm[0]}")
        return

    # Spočítaj dolnú hranicu
    lb = lower_bound(partial_perm, widths, data)
    if lb >= best_cost[0]:
        return

    # Rozšír vetvu
    placed_set = set(partial_perm)
    for nxt in range(n):
        if nxt not in placed_set:
            partial_perm.append(nxt)
            branch_and_bound(partial_perm, widths, data, best_cost, best_perm)
            partial_perm.pop()


def parallel_worker(start_device, widths, data, shared_best_cost, shared_best_perm, lock):
    local_best_cost = [shared_best_cost.value] 
    local_best_perm = [None]
    branch_and_bound([start_device], widths, data, local_best_cost, local_best_perm)

    if local_best_cost[0] < shared_best_cost.value:
        with lock:
            if local_best_cost[0] < shared_best_cost.value:
                shared_best_cost.value = local_best_cost[0]
                perm = local_best_perm[0]
                if perm[0] != max(perm):
                    perm = perm[::-1]
                shared_best_perm[:] = perm
                print(f"Process {start_device}: New global best cost: {shared_best_cost.value:.2f}")

if __name__ == "__main__":
    start = time.time()
    widths, data = load_data("Y-10_t.txt")

    manager = multiprocessing.Manager()
    shared_best_cost = manager.Value('d', float('10000000'))
    shared_best_perm = manager.list()
    lock = manager.Lock()

    n = len(widths)
    print("Start\n")

    # Vytvor a spusti procesy
    processes = [
        multiprocessing.Process(
            target=parallel_worker,
            args=(start_device, widths, data, shared_best_cost, shared_best_perm, lock)
        )
        for start_device in range(n)
    ]
    
    for p in processes:
        p.start()

    for p in processes:
        p.join()

    end = time.time()
    print(f"Best permutation: {list(shared_best_perm)}")
    print(f"Best cost: {shared_best_cost.value:.2f}")
    print(f"Time: {end - start:.2f} seconds")