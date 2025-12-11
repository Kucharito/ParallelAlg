import numpy as np
import math
import matplotlib.pyplot as plt
import time

N = 500 # poÄŤet bodĹŻ
D = 2   # dimenze
data = np.random.rand(N, D) * 100
plt.ion()
fig, ax = plt.subplots()
scat = ax.scatter(data[:,0], data[:,1])
circles = []
for i in range(N):
    color = 'yellow' if i % 2 == 0 else 'blue'
    circle = plt.Circle((data[i,0], data[i,1]),
                        2, color=color, fill=False)
    circles.append(circle)
    ax.add_artist(circle)
for iteration in range(100):
    for i in range(N):
        data[i] += np.random.randn(D) * 2
        circles[i].center = (data[i,0], data[i,1])
    scat.set_offsets(data)
    plt.draw()
    plt.pause(0.01)
    time.sleep(0.01)
plt.ioff()
plt.show()

