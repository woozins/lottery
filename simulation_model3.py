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

group_1 = []
group_2 = []


def classify(start, end):
  for i in tqdm(range(start, end)):
    comb = combinations[i]
    if np.array_equal(np.histogram(comb, bins = [1,16,31,46]), [2,2,2]):
      group_1.append(comb)
    elif np.array_equal(np.histogram(comb, bins = [1,16,31,46]), [1,2,3]):
      group_2.append(comb)
    elif np.array_equal(np.histogram(comb, bins = [1,16,31,46]), [1,3,2]):
      group_2.append(comb)
    elif np.array_equal(np.histogram(comb, bins = [1,16,31,46]), [2,3,1]):
      group_2.append(comb)
    elif np.array_equal(np.histogram(comb, bins = [1,16,31,46]), [2,1,3]):
      group_2.append(comb) 
    elif np.array_equal(np.histogram(comb, bins = [1,16,31,46]), [3,1,2]):
      group_2.append(comb)
    elif np.array_equal(np.histogram(comb, bins = [1,16,31,46]), [3,2,1]):
      group_2.append(comb) 

#define threads for classification
N = 8154060
if __name__ == '__main__':
  result = Queue()
  processes = []
  for i in range(49):
    p = Process(target=classify, args=((N//50)*i, (N//50)*(i+1)))
    processes.append(p)
  p_50 = Process(target=classify, args=((N//50)*49, N))
  processes.append(p_50)


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

group_3 = list(set(combinations) - set(group_1) - set(group_2))

#check classification
print(len(group_1))
print(len(group_2))
print(len(group_3))



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
  freqprob = (len(c[0])/tot
              )*w
  class_prob = [a, b, 1-a-b]
  tickets = []
  for _ in range(size):
    group = int(np.random.choice(list(range(3)), size=1, p=class_prob))
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
