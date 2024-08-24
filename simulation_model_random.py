import numpy as np
import random
from multiprocessing import Process, Queue
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def rticket(batch_size):
    return np.array([sorted(random.sample(range(1, 46), 6)) for _ in range(batch_size)])

B = 100000000
batch_size = 1000000
iter = B//batch_size


results = []

def work(start, end):
  for d in range(start, end):
    draw = np.sort(random.sample(list(range(1,46)), 6))
    results = []
    for _ in tqdm(range(iter)):
      auto = rticket(batch_size)
      result = np.sum(np.all(draw == auto, axis=1))
      results.append(result)
    
      print("d : %d, random, result = %d"%(d, sum(results)))
  return 

#Define Threads.

if __name__ == '__main__':
  result = Queue()
  processes = []
  for i in range(0, 500, 50):
    p = Process(target=work, args=(i, i + 50))
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
