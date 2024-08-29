import cupy as cp
import itertools
from multiprocessing import Process, Queue
from tqdm import tqdm
from numba import njit, prange

# Define Classes
numbers = list(range(1, 46))
combinations = list(itertools.combinations(numbers, 6))
freqgroup25 = list(itertools.combinations(list(range(1, 26)), 6))
freqgroup30 = list(itertools.combinations(list(range(1, 31)), 6))
freqgroup35 = list(itertools.combinations(list(range(1, 36)), 6))

othersgroup25 = list(set(combinations) - set(freqgroup25))
othersgroup30 = list(set(combinations) - set(freqgroup30))
othersgroup35 = list(set(combinations) - set(freqgroup35))

group25 = [freqgroup25, othersgroup25]
group30 = [freqgroup30, othersgroup30]
group35 = [freqgroup35, othersgroup35]

# Define sampling functions
@njit(parallel=True)
def rticket(batch_size):
    tickets = cp.empty((batch_size, 6), dtype=cp.int32)
    for i in prange(batch_size):
        ticket = cp.random.choice(cp.arange(1, 46), 6, replace=False)
        tickets[i] = cp.sort(ticket)
    return tickets


def nrticket1(c, w, size, N=45, k=6):
    class_prob = [w, 1 - w]
    tickets = []
    for _ in prange(size):
        group = int(cp.random.choice(cp.arange(2), size=1, p=class_prob))
        ticket = cp.sort(cp.random.choice(cp.array(c[group]), k, replace=False))
        tickets.append(ticket)
    return cp.array(tickets)

# Work
B = 1000
batch_size = 10
R = batch_size // 3
NR = batch_size - R
iterations = B // batch_size

w_grid_25 = [1/30, 1/20, 1/10]
w_grid_30 = [1/10, 1/5, 1/2]
w_grid_35 = [1/3, 1/2, 2/3]

def work25(start, end, gpu_id):
    cp.cuda.Device(gpu_id).use()
    for d in range(start, end):
        draw = cp.sort(cp.random.choice(cp.arange(1, 46), 6, replace=False))
        for weight in w_grid_25:
            results = []
            for _ in tqdm(range(iterations)):
                auto = rticket(R)
                manual = nrticket1(group25, w=weight, size=NR).ravel().reshape(NR, -1)
                result = cp.sum(cp.all(draw == auto, axis=1)) + cp.sum(cp.all(draw == manual, axis=1))
                results.append(result)

            print(f"type: 25, d: {d}, weight = {weight:.6f}, result = {cp.sum(results)}")
    return

def work30(start, end, gpu_id):
    cp.cuda.Device(gpu_id).use()
    for d in range(start, end):
        draw = cp.sort(cp.random.choice(cp.arange(1, 46), 6, replace=False))
        for weight in w_grid_30:
            results = []
            for _ in tqdm(range(iterations)):
                auto = rticket(R)
                manual = nrticket1(group30, w=weight, size=NR).ravel().reshape(NR, -1)
                result = cp.sum(cp.all(draw == auto, axis=1)) + cp.sum(cp.all(draw == manual, axis=1))
                results.append(result)

            print(f"type: 30, d: {d}, weight = {weight:.6f}, result = {cp.sum(results)}")
    return

def work35(start, end, gpu_id):
    cp.cuda.Device(gpu_id).use()
    for d in range(start, end):
        draw = cp.sort(cp.random.choice(cp.arange(1, 46), 6, replace=False))
        for weight in w_grid_35:
            results = []
            for _ in tqdm(range(iterations)):
                auto = rticket(R)
                manual = nrticket1(group35, w=weight, size=NR).ravel().reshape(NR, -1)
                result = cp.sum(cp.all(draw == auto, axis=1)) + cp.sum(cp.all(draw == manual, axis=1))
                results.append(result)

            print(f"type: 35, d: {d}, weight = {weight:.6f}, result = {cp.sum(results)}")
    return

# Define Processes

if __name__ == '__main__':
    result = Queue()
    processes = []
    num_gpus = 100  # Assuming 100 GPUs

    for i in range(25):
        gpu_id = i % num_gpus
        p = Process(target=work25, args=(20 * i, 20 * (i + 1), gpu_id))
        processes.append(p)

    for i in range(25):
        gpu_id = (i + 25) % num_gpus
        p = Process(target=work30, args=(20 * i, 20 * (i + 1), gpu_id))
        processes.append(p)

    for i in range(25):
        gpu_id = (i + 50) % num_gpus
        p = Process(target=work35, args=(20 * i, 20 * (i + 1), gpu_id))
        processes.append(p)

    # Start all processes
    for p in processes:
        p.start()

    # Wait for all processes to complete
    for p in processes:
        p.join()

    result.put('STOP')
    total = 0

    while True:
        tmp = result.get()
        if tmp == 'STOP':
            break
        else:
            total += tmp
