import requests
import multiprocessing as mp
import time

def say_hello(_):
    pid = mp.current_process().pid
    x=0
    for i in range(1000000):
        x=+1
    print(f"Hello from process {pid}")
    return pid

def download_page(i, times_spent, lock):
    pid = mp.current_process().pid
    start_time = time.time()
    r = requests.get(f'https://name-service-phi.vercel.app/api/v1/names/{i}.json')
    end_time = time.time()

    time_spent = end_time - start_time
    with lock:
        record = times_spent.get(pid, {"access_count": 0, "total_time": 0.0})
        record["access_count"] += 1
        record["total_time"] += time_spent
        times_spent[pid] = record
    return r.content




if __name__=="__main__":
    with mp.Manager() as manager:
        times_spent = manager.dict()
        lock = manager.Lock()
        with mp.Pool(10) as p:
            result = p.starmap(download_page, [(i, times_spent, lock) for i in range(100)])

        print(result)

        for pid, t in times_spent.items():
            print(f"Process {pid} made {t['access_count']} accesses, total time spent: {t['total_time']:.4f} seconds")
