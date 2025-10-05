import itertools
import numpy as np
import multiprocessing as mp
import time
'''
Check the example below and notice that:
  - we can make a random symmetric cost matrix to play with TSP problems of a suitable size
  - a BnB implementation will reduce the number of permutations for which we need to compute the cost


Challenge:
  - implement the `evaluate_bnb, test_permutations_bnb` functions and demonstrate their functionality
  - compare the no. of cost computations of the brute force and BnB solutions

'''

n = 10
np.random.seed(42)
C = np.random.rand(n, n)
## make it symmetric
C = C + C.T

cntr = 0


###########################################
def evaluate(C, p):
  global cntr
  cntr = cntr + 1

  cost = 0
  for i in range(len(p)):
    cost = cost + C[p[i], p[(i + 1) % n]]
  return cost


def test_permutations(n):
  min_cost = 1000000
  for p in itertools.permutations(range(n)):
    if p[0] == 0:
      cost = evaluate(C, p)
      if cost < min_cost:
        min_cost = cost
        min_perm = p
    else:
      break

  print('>', min_perm, min_cost)


##########################################
def evaluate_bnb(C, p, min_cost):
  global cntr
  cntr = cntr + 1

  cost = 0
  for i in range(len(p)):
    cost = cost + C[p[i], p[(i + 1) % n]]
    if cost > min_cost:
        return -1*(i+1)
  return cost


def test_permutations_bnb(n):
    min_cost = 1000000000
    skip_index = -1
    skip_value = -1
    for p in itertools.permutations(range(n)):
        if p[0] == 0:
            if skip_index >= 0:
                if p[skip_index] == skip_value:
                    continue
                else:
                    skip_index = -1
                    skip_value = -1
            cost = evaluate_bnb(C, p, min_cost)
            if cost < 0:
                if -cost < n:
                    skip_index = -cost
                    skip_value = p[skip_index]
            else:
                if cost < min_cost:
                    min_cost = cost
                    min_perm = p


    print('>', min_perm, min_cost)

def compute_bnb_parallel(k,C,min_cost,lock):
    global cntr

    n = len(C)
    cntr = 0
    min_perm = None
    skip_index = -1
    skip_value = -1
    remaining = [i for i in range(n) if i not in (0,k)]

    for tail in itertools.permutations(remaining):
        p = (0,k) + tail
        if skip_index >= 0:
            if p[skip_index] == skip_value:
                continue
            else:
                skip_index = -1
                skip_value = -1
        cost = evaluate_bnb(C, p, min_cost.value)
        if cost < 0:
            if -cost < n:
                skip_index = -cost
                skip_value = p[skip_index]
        else:
            if cost < min_cost.value:
                min_cost.value = cost
                min_perm = p

    with lock:
        print(f"Process {k}: permutations evaluated: {cntr}, best_len: {min_cost}, best_perm: {min_perm}")
    return


if __name__ == "__main__":

    time_start = time.time()

    n = len(C)
    with mp.Manager() as manager:
        min_cost_parallel = manager.Value('d', 1e10)
        lock = manager.Lock()
        with mp.Pool(n-1) as p:
            result = p.starmap(
                compute_bnb_parallel,
                [(k, C, min_cost_parallel, lock) for k in range(1, n)],
            )

        global_best_len = min_cost_parallel.value
        print(f"Global best len={global_best_len}")
    time_end = time.time()
    print("Time taken (parallel BnB):", time_end - time_start, "seconds")


    time_start = time.time()
    min_cost = 1e10
    cntr = 0
    test_permutations_bnb(n)
    time_end = time.time()
    print('Evaluated permutations:', cntr)
    print("Time taken (BnB):", time_end - time_start, "seconds")