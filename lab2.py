import itertools as it
import time

import multiprocessing as mp
'''
Check the example below and notice that:
  - there is a module called `itertools` that provides some functions with permutations (e.g., generate next permutation)
  - this can be used as a part of the for cycle
  - it is sometimes useful to generate permutations with a fixed prefix (e.g., [0, 1, ...], [0, 2, ...], etc.)
  - the 2D array D holds the distances between cities in a toy TSP example  
  - the `calc_perm` and `calc_perm_pref` are stubs of functions to compute the length of the path that need to be implemented.



Challenge:
  - implement the `calc_perm, calc_perm_pref` functions and demonstrate their functionality
  - bonus: implement a function that loads a distance matrix from a real TSP instance, 
  https://github.com/pdrozdowski/TSPLib.Net/blob/master/TSPLIB95/tsp/ulysses16.tsp
  - compute the length of a few tours for the ulysses16 instance (why do we not traverse all permutations in this case?)

'''

D = [[0, 5, 40, 11],
     [5, 0, 9, 6],
     [40, 9, 0, 8],
     [11, 6, 8, 0]]


def calc_perm(p, D):
  suma = 0
  n = len(p)
  for i in range(n):
      suma = suma + D[p[i]][p[(i+1)%n]]
      print("D[p[{}]][p[{}]]".format(i, (i+1)%n), D[p[i]][p[(i+1)%n]])
      print(suma)


#D = matica vzdialenosti
#pref = prve miesto
#p = permutacia bez prveho miesta
def calc_perm_pref(pref, p, D):
  suma = 0
  n = len(p)
  suma = suma + D[pref][p[0]] # prefox X ove mesto v permutacii
  for i in range (n-1): # cyklus cez permutaciu bez prveho miesta
      suma = suma + D[p[i]][p[i+1]]
  suma = suma + D[p[n - 1]][pref] # posledne miesto v permutacii X prefix



def test_permutations(n):
    for p in it.permutations(range(n)):
        print(p, calc_perm(p, D))



def test_permutations_pref(pref, n):
    for p in it.permutations(range(n)[1:]):
        print(pref, p, calc_perm_pref(pref, p, D))

def load_ulysses(max_coords):
    coords = []
    path = 'ulysses16.tsp'
    with open(path, 'r') as f:
        lines = f.readlines()
        coord_lines = lines[7:-1]
        for lines in coord_lines:
            parts = lines.split()
            coords.append((float(parts[1]), float(parts[2])))
    return coords


def compute_distance_matrix(coords):
    import math
    n = len(coords)
    D = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i][j] = math.sqrt((coords[i][0]-coords[j][0])**2 + (coords[i][1]-coords[j][1])**2)
    return D

test_permutations(4)
test_permutations_pref(0, 4)


N = 3
coords = load_ulysses(N)
D = compute_distance_matrix(coords)


def compute_for_fixed_two(k,D,distances,lock):
    n = len(D)
    remaining = [i for i in range(n) if i != 0 and i != k]
    print("Process {} handling prefix (0, {})".format(mp.current_process().pid, k))
    for tail in it.permutations(remaining):
        tour = (0, k) + tail
        L = calc_perm(tour, D)
        with lock:
            if k not in distances:
                distances[k] = L
            distances[k] = min(distances[k], L)

if __name__ == "__main__":
    with mp.Manager() as manager:
        distances = manager.dict()
        lock = manager.Lock()
        with mp.Pool(N-1) as p:
            result = p.starmap(compute_for_fixed_two, [(i, distances, lock) for i in range(1, N)])


'''
start_time = time.time()
test_permutations(N)
end_time = time.time()
print("Time for all permutations TEST PERMUT with N={}: ".format(N), end_time - start_time)

start_time = time.time()
test_permutations_pref(0, N)
end_time = time.time()
print("Time for permutations with prefix: TEST PERMUT PREF with N={}: ".format(N), end_time - start_time)

'''