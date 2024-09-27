import pandas as pd
import numpy as np
import random
import itertools
from math import factorial
from functools import lru_cache
from multiprocessing import Process, Queue
from tqdm import tqdm

#Define Classes : map each combinations to 1 to 8145060
combinations = list(range(8145060))
freqgroup_25 = list(range(177100))
othersgroup_25 = list(set(combinations) - set(freqgroup_25))
group_for_analysis_25 = [freqgroup_25, othersgroup_25]

freqgroup_30 = list(range(593775))
othersgroup_30 = list(set(combinations) - set(freqgroup_30))
group_for_analysis_30 = [freqgroup_30, othersgroup_30]

freqgroup_35 = list(range(1623160))
othersgroup_35 = list(set(combinations) - set(freqgroup_35))
group_for_analysis_35 = [freqgroup_35, othersgroup_35]

#define sampling functions
def rticket(batch_size):
    tickets = np.empty((batch_size, 1), dtype=np.int32)
    for i in range(batch_size):
        ticket = np.random.choice(combinations)
        tickets[i] = ticket
    return tickets


def nrticket1(c, w, size): 
  class_prob = [w, 1-w]
  tickets = []
  for _ in range(size):
    group = int(np.random.choice(list(range(2)), size = 1, p=class_prob))
    ticket = random.sample(c[group], 1)
    tickets.append(ticket)
  return np.array(tickets)

# work
B = 100000000
batch_size = 1000000
R = batch_size // 3
NR = batch_size - R
iter = B//batch_size

w_grid = [0.25, 0.375, 0.5]

def work25(start, end):
  for d in range(start, end):
    draw =  nrticket1(c = group_for_analysis_25, w = 0.3, size = 1).ravel()
    for weight in w_grid:
        results = []
        for _ in tqdm(range(iter)): #배치 반복 횟수.
            auto = rticket(R)
            manual = nrticket1(c = group_for_analysis_25, w = weight, size=NR).ravel().reshape(NR, -1)
            result = np.sum(np.all(draw == auto, axis=1)) + np.sum(np.all(draw == manual, axis=1))
            results.append(result)
        print("d : %d, weight = %f, result = %d"%(d, weight, sum(results)))
  return


def work30(start, end):
  for d in range(start, end):
    draw =  nrticket1(c = group_for_analysis_30, w = 0.3, size = 1).ravel()
    for weight in w_grid:
        results = []
        for _ in tqdm(range(iter)): #배치 반복 횟수.
            auto = rticket(R)
            manual = nrticket1(c = group_for_analysis_30, w = weight, size=NR).ravel().reshape(NR, -1)
            result = np.sum(np.all(draw == auto, axis=1)) + np.sum(np.all(draw == manual, axis=1))
            results.append(result)
        print("d : %d, weight = %f, result = %d"%(d, weight, sum(results)))
  return

def work35(start, end):
  for d in range(start, end):
    draw =  nrticket1(c = group_for_analysis_35, w = 0.3, size = 1).ravel()
    for weight in w_grid:
        results = []
        for _ in tqdm(range(iter)): #배치 반복 횟수.
            auto = rticket(R)
            manual = nrticket1(c = group_for_analysis_35, w = weight, size=NR).ravel().reshape(NR, -1)
            result = np.sum(np.all(draw == auto, axis=1)) + np.sum(np.all(draw == manual, axis=1))
            results.append(result)
        print("d : %d, weight = %f, result = %d"%(d, weight, sum(results)))
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