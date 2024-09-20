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
othersgroup1 = list(set(combinations) - set(freqgroup1))
group1 = [freqgroup1, othersgroup1]

freqgroup2 = list(itertools.combinations(list(range(11,36)), 6))
othersgroup2 = list(set(combinations) - set(freqgroup2))
group2 = [freqgroup2, othersgroup2]

freqgroup3 = list(itertools.combinations(list(range(21,46)), 6))
othersgroup3 = list(set(combinations) - set(freqgroup3))
group3 = [freqgroup3, othersgroup3]


#define sampling functions
def rticket(batch_size):
    tickets = np.empty((batch_size, 6), dtype=np.int32)
    for i in range(batch_size):
        ticket = np.random.choice(np.arange(1, 46), 6, replace=False)
        tickets[i] = np.sort(ticket)
    return tickets


def nrticket1(c, w, size): 
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

groups = [group1, group2, group3]

def work25(start, end):
  data = []
  for d in range(start, end):
    draw = np.sort(random.sample(list(range(1,46)), 6))
    for i, g in enumerate(groups):
        results = []
        for _ in tqdm(range(iter)): #배치 반복 횟수.
            auto = rticket(R)
            manual = nrticket1(g , w = 0.1, size=NR).ravel().reshape(NR, -1)
            result = np.sum(np.all(draw == auto, axis=1)) + np.sum(np.all(draw == manual, axis=1))
            results.append(result)
        print("group : %d, d : %d, weight = %f, result = %d"%(i+1, d, 0.1, sum(results)))
  return


if __name__ == '__main__':
  result = Queue()
  processes = []
  for i in range(50):
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