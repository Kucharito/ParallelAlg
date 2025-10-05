import itertools as it
import time


import multiprocessing as mp

D = [
    [0, 5, 40, 11],
    [5, 0, 9, 6],
    [40, 9, 0, 8],
    [11, 6, 8, 0]
    ]

def calc_perm(p, D):
    suma = 0
    n = len(p)
    for i in range(n):
        suma += D[p[i]][p[(i+1)%n]]
        #print("D[p[{}]][p[{}]]".format(i, (i+1)%n))
    return suma

#suma +=
#D = matice vzdalenostĂ­
#pref = prvnĂ­ mesto
#p = permutace ostatnĂ­ch mÄ›st
#pref, p[0]
def calc_perm_pref(pref, p, D):
    suma = 0
    n = len(p)
    suma += D[pref][p[0]] # (prefix X prvnĂ­ mÄ›sto v permutaci)
    for i in range(n-1):  #mezi mÄ›sty v permutaci, kromÄ› (poslednĂ­ X prvnĂ­)
        suma += D[p[i]][p[i+1]]
    suma += D[p[-1]][pref] #(poslednĂ­ mÄ›sto v permutaci X prefix)

    return suma

def test_permutations(n):
  for p in it.permutations(range(n)):
      #print(p)
      calc_perm(p, D)
    #print(p, calc_perm(p, D))

#[a:b:c]
range(4)#(0,1,2,3)
range(4)[1:]#(1,2,3)

def test_permutations_pref(pref, n):
    for p in it.permutations(range(n)[1:]):
        calc_perm_pref(pref, p, D)
        #print(pref, p, calc_perm_pref(pref, p, D))



def load_ulysses(max_coords):
    coords = []
    path = "ulysses16.tsp"
    index = -1 - (16 - max_coords)
    with open(path) as f:
        lines = f.readlines()
        #skip 7 lines and read lines until EOF
        coord_lines = lines[7:index]
        for line in coord_lines:
            parts = line.split()
            coords.append((float(parts[1]), float(parts[2])))
    return coords

#test_permutations(4)
#test_permutations_pref(0,4)

def compute_distance_matrix(coords):
    import math
    n = len(coords)
    D = [[0]*n for _ in range(n)]

    '''
    D = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(0)
        D.append(row)
    '''

    for i in range(n):
        for j in range(n):
            if i != j:
                D[i][j] = round(math.sqrt((coords[i][0]-coords[j][0])**2 + (coords[i][1]-coords[j][1])**2),2)
    return D


def compute_for_fixed_two(k,D,distances,lock):
    n = len(D)
    best_len = None
    best_tour = None
    remaining = [i for i in range(n) if i not in (0, k)]
 #   print("Process {} computing for fixed two places: 0 and {}, remaining: {}".format(mp.current_process().pid, k, remaining))
    for tail in it.permutations(remaining):
        tour = (0, k) + tail
        L = calc_perm(tour, D)
        if best_len is None or L < best_len:
            best_len = L
            best_tour = tour
    with lock:
        distances[k] = {
            "pid": mp.current_process().pid,
            "best_len": best_len,
            "best_tour": best_tour,
        }



N = 9
coords = load_ulysses(N)
D = compute_distance_matrix(coords)

if __name__ == "__main__":
    with mp.Manager() as manager:

        distances = manager.dict()
        lock = manager.Lock()
        with mp.Pool(N-1) as p:
            result = p.starmap(
                compute_for_fixed_two,
                [(k, D, distances, lock) for k in range(1, N)]
            )
        global_best_len = None
        global_best_tour = None
        for k, v in distances.items():
            if global_best_len is None or v["best_len"] < global_best_len:
                global_best_len = v["best_len"]
                global_best_tour = v["best_tour"]
            print("For k={} best_len={}, best_tour={}, pid={}".format(k, v["best_len"], v["best_tour"], v["pid"]))
        print("Global best_len={}, best_tour={}".format(global_best_len, global_best_tour))

