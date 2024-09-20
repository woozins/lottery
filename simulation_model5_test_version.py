import pandas as pd
import numpy as np
import random
import itertools
from math import factorial
from functools import lru_cache
from multiprocessing import Process, Queue
from tqdm import tqdm

#Define Classes
numbers = list(range(1,11))
combinations = list(itertools.combinations(numbers, 3))

freqgroup1 = list(itertools.combinations(list(range(1,6)), 3))
othersgroup1 = list(set(combinations) - set(freqgroup1))
group1 = [freqgroup1, othersgroup1]

freqgroup2 = list(itertools.combinations(list(range(3,9)), 3))
othersgroup2 = list(set(combinations) - set(freqgroup2))
group2 = [freqgroup2, othersgroup2]

freqgroup3 = list(itertools.combinations(list(range(5,11)), 3))
othersgroup3 = list(set(combinations) - set(freqgroup3))
group3 = [freqgroup3, othersgroup3]


def rticket(batch_size):
    tickets = np.empty((batch_size, 3), dtype=np.int32)
    for i in range(batch_size):
        ticket = np.random.choice(np.arange(1, 11), 3, replace=False)
        tickets[i] = np.sort(ticket)
    return tickets

def nrticket1(c, w, size): 
  class_prob = [w, 1-w]
  tickets = []
  for _ in range(size):
    group = int(np.random.choice(list(range(2)), size=1, p=class_prob))
    ticket = random.sample(c[group], 1)
    tickets.append(ticket)
  return np.array(tickets)


# work
B = 1000
batch_size = 10
R = batch_size // 3
NR = batch_size - R
iter = B//batch_size

groups = [group1, group2, group3]

def work25(start, end):
  data = []
  for d in range(start, end):
    draw = np.sort(random.sample(list(range(1,11)), 3))
    for i, g in enumerate(groups):
        results = []
        for _ in tqdm(range(iter)): #배치 반복 횟수.
            auto = rticket(R)
            manual = nrticket1(g , w = 0.1, size=NR).ravel().reshape(NR, -1)
            result = np.sum(np.all(draw == auto, axis=1)) + np.sum(np.all(draw == manual, axis=1))
            results.append(result)
        data.append({
            'group': i+1,
            'd': d,
            'weight': 0.1,
            'result': sum(results)
        })
  return pd.DataFrame(data)

a = work25(1,100)
print(a.groupby('group').result.plot.hist())
