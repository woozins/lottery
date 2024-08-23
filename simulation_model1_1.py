import pandas as pd
import numpy as np
import random
import itertools
from math import factorial
from functools import lru_cache
from multiprocessing import Process, Queue
from tqdm import tqdm

F = list(range(1, 16)) 

numbers = list(range(1,46))
combinations = list(itertools.combinations(numbers, 6))

c = [[] for _ in range(7)]
for comb in combinations:
  i = len(set(F).intersection(set(comb)))
  c[i].append(comb)


def rticket(batch_size):
    return np.array([sorted(random.sample(range(1, 46), 6)) for _ in range(batch_size)])


def comb(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))

@lru_cache(None)
def comb_cached(n, k):
    return comb(n, k)

def nrticket1(c, F, w, size, N = 45, k = 6): 
  K = len(F)
  freq_prob = np.array([w*K, 1-(w*K)])
  class_prob = [comb_cached(k,i)*((freq_prob[0])**i)*(freq_prob[1])**(k-i) for i in range(7)]


  tickets = []
  for _ in range(size):
    group = int(np.random.choice(list(range(7)), size=1, p=class_prob))
    ticket = sorted(random.sample(c[group], 1))
    tickets.append(ticket)
  return np.array(tickets)

B = 100000000

batch_size = 1000000


R = batch_size // 3
NR = batch_size - R
iter = B//batch_size

F = list(range(1, 16))


w_grid = [1/45, 1/30, 2/45, 1/18]


def work(start, end):
  for d in range(start, end):
    draw = np.sort(random.sample(list(range(1, 46)), 6))
    for weight in w_grid:
      results = []
      for _ in tqdm(range(iter)):
        auto = rticket(R)
        manual = nrticket1(c, F, w = weight, size=NR).ravel().reshape(NR, -1)
        result = np.sum(np.all(draw == auto, axis=1)) + np.sum(np.all(draw == manual, axis=1))
        results.append(result)
      
      print("d : %d, weight = %f, result = %d"%(d, weight, sum(results)))
  return 


if __name__ == '__main__':
  result = Queue()
  th1 = Process(target=work, args=(0, 2))
  th2 = Process(target=work, args=(2, 4))
  th3 = Process(target=work, args=(4, 6))
  th4 = Process(target=work, args=(6, 8))
  th5 = Process(target=work, args=(8, 10))
  th6 = Process(target=work, args=(10, 12))
  th7 = Process(target=work, args=(12, 14))
  th8 = Process(target=work, args=(14, 16))
  th9 = Process(target=work, args=(16, 18))
  th10 = Process(target=work, args=(18, 20))


  th1.start()
  th2.start()
  th3.start()
  th4.start()
  th5.start()
  th6.start()
  th7.start()
  th8.start()
  th9.start()
  th10.start()


  th1.join()
  th2.join()
  th3.join()
  th4.join()
  th5.join()
  th6.join()
  th7.join()
  th8.join()
  th9.join()
  th10.join()

  result.put('STOP')
  total = 0


  while True:
      tmp = result.get()
      
      
      if tmp == 'STOP':
          break
      else:
          total += tmp
