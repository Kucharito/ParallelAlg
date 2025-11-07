#game of life
import numpy as np
import time
D = 40
MATRIX = [[0 for _ in range(D)] for _ in range(D)]

for i in range(D):
    for j in range(D):
        if np.random.rand() < 0.3:
            MATRIX[i][j] = 1

def print_matrix():
    # Clear the console
    print("\033[H\033[J", end="")
    for row in MATRIX:
        for char in row:
            if char == 0:
                print('.', end=' ')
            else:
                print('#', end=' ')
        print()


for iteration in range(1000):
    print_matrix()
    new_matrix = [[0 for _ in range(D)] for _ in range(D)]
    for i in range(D):
        for j in range(D):
            live_neighbors = 0
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < D and 0 <= nj < D:
                        live_neighbors += MATRIX[ni][nj]
            if MATRIX[i][j] == 1:
                if live_neighbors < 2 or live_neighbors > 3:
                    new_matrix[i][j] = 0
                else:
                    new_matrix[i][j] = 1
            else:
                if live_neighbors == 3:
                    new_matrix[i][j] = 1
                else:
                    new_matrix[i][j] = 0
    MATRIX = new_matrix
    time.sleep(0.1)