import pandas as pd
import numpy as np
import random
import itertools
from math import factorial
from functools import lru_cache
from multiprocessing import Process, Queue
from tqdm import tqdm

#Define Classes
numbers = list(range(1,46))
combinations = list(itertools.combinations(numbers, 6))

freqgroup1 = list(itertools.combinations(list(range(1,26)), 6))
freqgroup2 = list(itertools.combinations(list(range(5,31)), 6))
freqgroup3 = list(itertools.combinations(list(range(10,35)), 6))

othersgroup1 = list(set(combinations) - set(freqgroup1))
othersgroup2 = list(set(combinations) - set(freqgroup2))
othersgroup3 = list(set(combinations) - set(freqgroup3))

group1 = [freqgroup1, othersgroup1]
group2 = [freqgroup2, othersgroup2]
group3 = [freqgroup3, othersgroup3]

#define sampling functions
def rticket(batch_size):
    return np.array([sorted(random.sample(range(1, 46), 6)) for _ in range(batch_size)])

def comb(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))

@lru_cache(None)
def comb_cached(n, k):
    return comb(n, k)

def nrticket1(c, w, size, N = 45, k = 6): 
  class_prob = [w, 1-w]
  tickets = []
  for _ in range(size):
    group = int(np.random.choice(list(range(2)), size=1, p=class_prob))
    ticket = sorted(random.sample(c[group], 1))
    tickets.append(ticket)
  return np.array(tickets)

# work

B = 100000000

batch_size = 1000000
R = batch_size // 3
NR = batch_size - R
iter = B//batch_size

w_grid_25 = [1/30, 1/20, 1/10]

def work25(start, end):
  for d in range(start, end):
    draw = np.sort(random.sample(list(range(1,46)), 6))
    for weight in w_grid_25:
      results1 = []
      results2 = []
      results3 = []
      for _ in tqdm(range(iter)):
        auto = rticket(R)
        manual1 = nrticket1(group1 , w = weight, size=NR).ravel().reshape(NR, -1)
        manual2 = nrticket1(group2 , w = weight, size=NR).ravel().reshape(NR, -1)
        manual3 = nrticket1(group3 , w = weight, size=NR).ravel().reshape(NR, -1)

        result1 = np.sum(np.all(draw == auto, axis=1)) + np.sum(np.all(draw == manual1, axis=1))
        results1.append(result1)
      
        result2 = np.sum(np.all(draw == auto, axis=1)) + np.sum(np.all(draw == manual2, axis=1))
        results2.append(result2)

        result3 = np.sum(np.all(draw == auto, axis=1)) + np.sum(np.all(draw == manual3, axis=1))
        results3.append(result3) 

      print("group1 : d : %d, weight = %f, result = %d"%(d, weight, sum(results1)))
      print("group2 : d : %d, weight = %f, result = %d"%(d, weight, sum(results2)))
      print("group3 : d : %d, weight = %f, result = %d"%(d, weight, sum(results3)))

  return 

#Define Threads

if __name__ == '__main__':
  result = Queue()
  processes = []
  for i in range(75):
    p = Process(target=work25, args=(10*i, 10*(i+1)))
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