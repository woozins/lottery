import pandas as pd
import numpy as np
import random
import itertools
from math import factorial
from functools import lru_cache
from multiprocessing import Process, Queue
from tqdm import tqdm


#Define Classes

def find_combinations(current_comb, start, result):
    if len(current_comb) == 6:
        result.append(tuple(current_comb))
        return
    
    for i in range(start, 46):
        if not current_comb or i - current_comb[-1] >= 4:
            find_combinations(current_comb + [i], i + 1, result)

result = []
find_combinations([], 1, result)

freqgroup = result
P = 10000
plusfreq = list(map(tuple,[sorted(random.sample(range(1,46), 6)) for _ in range(P)]))

freqgroup = freqgroup + plusfreq


numbers = list(range(1,46))
combinations = list(itertools.combinations(numbers, 6))
othersgroup = list(set(combinations) - set(freqgroup))
group = [freqgroup, othersgroup]




#define sampling functions
def rticket(batch_size):
    return np.array([sorted(random.sample(range(1, 46), 6)) for _ in range(batch_size)])

def comb(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))

@lru_cache(None)
def comb_cached(n, k):
    return comb(n, k)


def nrticket1(c, w, size, N = 45, k = 6): 
  tot = len(c[0]) + len(c[1])
  freqprob = (len(c[0])/tot)*w
  class_prob = [freqprob, 1-freqprob]
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

F = list(range(1, 11))

results = []

w_grid = [200, 400, 600, 800, 1000]

results_data = []

def work(start, end):
  for d in range(start, end):
    draw = np.sort(random.sample(freqgroup, 1))
    for weight in w_grid:
      results = []
      for _ in tqdm(range(iter)):
        auto = rticket(R)
        manual = nrticket1(group, w = weight, size=NR).ravel().reshape(NR, -1)
        result = np.sum(np.all(draw == auto, axis=1)) + np.sum(np.all(draw == manual, axis=1))
        results.append(result)
      
      print("d : %d, weight = %f, result = %d"%(d, weight, sum(results)))
  return 






if __name__ == '__main__':
  result = Queue()
  th1 = Process(target=work, args=(0, 10))
  th2 = Process(target=work, args=(10, 20))
  th3 = Process(target=work, args=(20, 30))
  th4 = Process(target=work, args=(30, 40))
  th5 = Process(target=work, args=(40, 50))
  th6 = Process(target=work, args=(50, 60))
  th7 = Process(target=work, args=(60, 70))
  th8 = Process(target=work, args=(70, 80))
  th9 = Process(target=work, args=(80, 90))
  th10 = Process(target=work, args=(90, 100))


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
