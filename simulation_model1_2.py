import pandas as pd
import numpy as np
import random
import itertools
from math import factorial
from functools import lru_cache
from multiprocessing import Process, Queue
from tqdm import tqdm

F = list(range(1, 20)) 

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


w_grid = [1/45, 1/36, 1/30, 7/180, 2/45]


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
  th1 = Process(target=work, args=(0, 5))
  th2 = Process(target=work, args=(5, 10))
  th3 = Process(target=work, args=(10, 15))
  th4 = Process(target=work, args=(15, 20))
  th5 = Process(target=work, args=(20, 25))
  th6 = Process(target=work, args=(25, 30))
  th7 = Process(target=work, args=(30, 35))
  th8 = Process(target=work, args=(35, 40))
  th9 = Process(target=work, args=(40, 45))
  th10 = Process(target=work, args=(45, 50))
  th11 = Process(target=work, args=(50, 55))
  th12 = Process(target=work, args=(55, 60))
  th13 = Process(target=work, args=(60, 65))
  th14 = Process(target=work, args=(65, 70))
  th15 = Process(target=work, args=(70, 75))
  th16 = Process(target=work, args=(75, 80))
  th17 = Process(target=work, args=(80, 85))
  th18 = Process(target=work, args=(85, 90))
  th19 = Process(target=work, args=(90, 95))
  th20 = Process(target=work, args=(95, 100))

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
  th11.start()
  th12.start()
  th13.start()
  th14.start()
  th15.start()
  th16.start()
  th17.start()
  th18.start()
  th19.start()
  th20.start()

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
  th11.join()
  th12.join()
  th13.join()
  th14.join()
  th15.join()
  th16.join()
  th17.join()
  th18.join()
  th19.join()
  th20.join()

  result.put('STOP')
  total = 0


  while True:
      tmp = result.get()
      
      
      if tmp == 'STOP':
          break
      else:
          total += tmp
