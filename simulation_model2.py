import pandas as pd
import numpy as np
import random
import itertools
from math import factorial
from functools import lru_cache
from multiprocessing import Process, Queue
from tqdm import tqdm
from numba import njit, prange


#Define Classes
numbers = list(range(1,46))
combinations = list(itertools.combinations(numbers, 6))
freqgroup25 = list(itertools.combinations(list(range(1,26)), 6))
freqgroup30 = list(itertools.combinations(list(range(1,31)), 6))
freqgroup35 = list(itertools.combinations(list(range(1,36)), 6))

othersgroup25 = list(set(combinations) - set(freqgroup25))
othersgroup30 = list(set(combinations) - set(freqgroup30))
othersgroup35 = list(set(combinations) - set(freqgroup35))

group25 = [freqgroup25, othersgroup25]
group30 = [freqgroup30, othersgroup30]
group35 = [freqgroup35, othersgroup35]


#define sampling functions
@njit(parallel=True)
def rticket(batch_size):
    tickets = np.empty((batch_size, 6), dtype=np.int32)
    for i in prange(batch_size):
        ticket = np.random.choice(np.arange(1, 46), 6, replace=False)
        tickets[i] = np.sort(ticket)
    return tickets


def comb(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))

@lru_cache(None)
def comb_cached(n, k):
    return comb(n, k)


def nrticket1(c, w, size, N = 45, k = 6): 
  class_prob = [w, 1-w]
  tickets = []
  for _ in prange(size):
    group = int(np.random.choice(list(range(2)), size=1, p=class_prob))
    ticket = sorted(random.sample(c[group], 1))
    tickets.append(ticket)
  return np.array(tickets)



# work

B = 1000

batch_size = 10
R = batch_size // 3
NR = batch_size - R
iter = B//batch_size

w_grid_25 = [1/30, 1/20, 1/10]
w_grid_30 = [1/10, 1/5, 1/2]
w_grid_35 = [1/3, 1/2, 2/3]

def work25(start, end):
  for d in range(start, end):
    draw = np.sort(random.sample(list(range(1,46)), 6))
    for weight in w_grid_25:
      results = []
      for _ in tqdm(range(iter)):
        auto = rticket(R)
        manual = nrticket1(group25 , w = weight, size=NR).ravel().reshape(NR, -1)
        result = np.sum(np.all(draw == auto, axis=1)) + np.sum(np.all(draw == manual, axis=1))
        results.append(result)
      
      print("type : 25, d : %d, weight = %f, result = %d"%(d, weight, sum(results)))
  return 

def work30(start, end):
  for d in range(start, end):
    draw = np.sort(random.sample(list(range(1,46)), 6))
    for weight in w_grid_30:
      results = []
      for _ in tqdm(range(iter)):
        auto = rticket(R)
        manual = nrticket1(group30 , w = weight, size=NR).ravel().reshape(NR, -1)
        result = np.sum(np.all(draw == auto, axis=1)) + np.sum(np.all(draw == manual, axis=1))
        results.append(result)
      
      print("type : 30, d : %d, weight = %f, result = %d"%(d, weight, sum(results)))
  return 

def work35(start, end):
  for d in range(start, end):
    draw = np.sort(random.sample(list(range(1,46)), 6))
    for weight in w_grid_35:
      results = []
      for _ in tqdm(range(iter)):
        auto = rticket(R)
        manual = nrticket1(group35 , w = weight, size=NR).ravel().reshape(NR, -1)
        result = np.sum(np.all(draw == auto, axis=1)) + np.sum(np.all(draw == manual, axis=1))
        results.append(result)
      
      print("type : 35, d : %d, weight = %f, result = %d"%(d, weight, sum(results)))
  return 

#Define Threads

if __name__ == '__main__':
  result = Queue()
  processes = []
  for i in range(25):
    p = Process(target=work25, args=(20*i, 20*(i+1)))
    processes.append(p)

  for i in range(25):
    p = Process(target=work30, args=(20*i, 20*(i+1)))
    processes.append(p)

  for i in range(25):
    p = Process(target=work35, args=(20*i, 20*(i+1)))
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