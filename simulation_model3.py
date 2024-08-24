import numpy as np
import random
import itertools
from math import factorial
from functools import lru_cache
from multiprocessing import Process, Queue
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

#Define Classes
numbers = list(range(1,46))
combinations = list(itertools.combinations(numbers, 6))

#group_1
temp_1 = list(itertools.combinations(list(range(1,16)),2))
temp_2 = list(itertools.combinations(list(range(16,31)),2))
temp_3 = list(itertools.combinations(list(range(31,46)),2))
group1 = list(itertools.product(temp_1, temp_2, temp_3))
group1 = [x[0]+x[1]+x[2] for x in group1]

#group_2
temp_1 = list(itertools.combinations(list(range(1,16)),1))
temp_2 = list(itertools.combinations(list(range(16,31)),2))
temp_3 = list(itertools.combinations(list(range(31,46)),3))
group_123 = list(itertools.product(temp_1, temp_2, temp_3))
group_123 = [x[0]+x[1]+x[2] for x in group_123]

temp_1 = list(itertools.combinations(list(range(1,16)),1))
temp_2 = list(itertools.combinations(list(range(16,31)),3))
temp_3 = list(itertools.combinations(list(range(31,46)),2))
group_132 = list(itertools.product(temp_1, temp_2, temp_3))
group_132 = [x[0]+x[1]+x[2] for x in group_132]

temp_1 = list(itertools.combinations(list(range(1,16)),2))
temp_2 = list(itertools.combinations(list(range(16,31)),1))
temp_3 = list(itertools.combinations(list(range(31,46)),3))
group_213 = list(itertools.product(temp_1, temp_2, temp_3))
group_213 = [x[0]+x[1]+x[2] for x in group_213]

temp_1 = list(itertools.combinations(list(range(1,16)),2))
temp_2 = list(itertools.combinations(list(range(16,31)),3))
temp_3 = list(itertools.combinations(list(range(31,46)),1))
group_231 = list(itertools.product(temp_1, temp_2, temp_3))
group_231 = [x[0]+x[1]+x[2] for x in group_231]

temp_1 = list(itertools.combinations(list(range(1,16)),3))
temp_2 = list(itertools.combinations(list(range(16,31)),1))
temp_3 = list(itertools.combinations(list(range(31,46)),2))
group_312 = list(itertools.product(temp_1, temp_2, temp_3))
group_312 = [x[0]+x[1]+x[2] for x in group_312]

temp_1 = list(itertools.combinations(list(range(1,16)),3))
temp_2 = list(itertools.combinations(list(range(16,31)),2))
temp_3 = list(itertools.combinations(list(range(31,46)),1))
group_321 = list(itertools.product(temp_1, temp_2, temp_3))
group_321 = [x[0]+x[1]+x[2] for x in group_321]

group2 = group_123 + group_132 + group_213 + group_231 + group_312 + group_321
group3 = list(set(combinations) - set(group1) - set(group2))

group = [group1, group2, group3]

#define sampling functions
def rticket(batch_size):
    return np.array([sorted(random.sample(range(1, 46), 6)) for _ in range(batch_size)])

def comb(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))

@lru_cache(None)
def comb_cached(n, k):
    return comb(n, k)



def nrticket1(c, w, size): 
  tickets = []
  for _ in range(size):
    group = int(np.random.choice(list(range(3)), size=1, p=w))
    ticket = sorted(random.sample(c[group], 1))
    tickets.append(ticket)
  return np.array(tickets)



# work

B = 100000000

batch_size = 1000000
R = batch_size // 3
NR = batch_size - R
iter = B//batch_size

results = []

w_grid = [[0.25, 0.7, 0.05], [0.25, 0.6, 0.15],[0.25,0.5,0.25],[0.2,0.7,0.1],[0.2,0.6,0.2],
          [0.2,0.5,0.3],[0.15,0.55,0.3]]

def work(start, end):
  for d in range(start, end):
    draw = np.sort(random.sample(list(range(1,46)), 6))
    for weight in w_grid:
      results = []
      for _ in tqdm(range(iter)):
        auto = rticket(R)
        manual = nrticket1(group, w = weight, size=NR).ravel().reshape(NR, -1)
        result = np.sum(np.all(draw == auto, axis=1)) + np.sum(np.all(draw == manual, axis=1))
        results.append(result)
      
      print("d : %d, weight_a = %f, weight_b = %f, weight_c = %f, result = %d"%(d, weight[0], weight[1], weight[2], sum(results)))
  return 




#Define Threads.

if __name__ == '__main__':
  result = Queue()
  processes = []
  for i in range(0, 500, 10):
    p = Process(target=work, args=(i, i + 10))
    processes.append(p)

  # 모든 프로세스를 시작합니다.
  for p in processes:
      p.start()

  # 모든 프로세스가 종료될 때까지 기다립니다.
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
