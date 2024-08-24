import itertools
import random
import numpy as np
from tqdm import tqdm
from functools import lru_cache
from math import factorial
from multiprocessing import Pool

# 사전 설정 작업
numbers = list(range(1, 46))
combinations = set(itertools.combinations(numbers, 6))

# 그룹 1 생성
group1 = set([
    x[0] + x[1] + x[2] for x in itertools.product(
        itertools.combinations(range(1, 16), 2),
        itertools.combinations(range(16, 31), 2),
        itertools.combinations(range(31, 46), 2)
    )
])

# 그룹 2 생성
def generate_group2():
    groups = []
    for a, b, c in [(1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1)]:
        group = set([
            x[0] + x[1] + x[2] for x in itertools.product(
                itertools.combinations(range(1, 16), a),
                itertools.combinations(range(16, 31), b),
                itertools.combinations(range(31, 46), c)
            )
        ])
        groups.append(group)
    return set.union(*groups)

group2 = generate_group2()

# 그룹 3 생성
group3 = combinations - group1 - group2
group = [list(group1), list(group2), list(group3)]

# 함수 정의
def rticket(batch_size):
    return np.array([sorted(random.sample(range(1, 46), 6)) for _ in range(batch_size)])

def nrticket1(c, w, size):
    tickets = []
    for _ in range(size):
        group_idx = np.random.choice(range(3), p=w)
        ticket = random.choice(c[group_idx])
        tickets.append(ticket)
    return np.array(tickets)

# 작업 함수
def work(params):
    draw, weight, iter, R, NR = params
    results = []
    for _ in tqdm(range(iter)):
        auto = rticket(R)
        manual = nrticket1(group, w=weight, size=NR)
        result = np.sum(np.all(auto == draw, axis=1)) + np.sum(np.all(manual == draw, axis=1))
        results.append(result)
    return sum(results)

# 병렬 처리 실행
def parallel_work(start, end):
    B = 100000000
    batch_size = 1000000
    R = batch_size // 3
    NR = batch_size - R
    iter = B // batch_size
    
    w_grid = [
        [0.25, 0.7, 0.05], [0.25, 0.6, 0.15], [0.25, 0.5, 0.25],
        [0.2, 0.7, 0.1], [0.2, 0.6, 0.2], [0.2, 0.5, 0.3],
        [0.15, 0.55, 0.3]
    ]

    results = []

    for d in range(start, end):
        draw = np.sort(np.random.choice(numbers, 6, replace=False))
        params = [(draw, weight, iter, R, NR) for weight in w_grid]
        
        with Pool(processes = 50) as pool:
            result_list = pool.map(work, params)
            
        for weight, result in zip(w_grid, result_list):
            results.append((d, weight, result))
            print(f"d: {d}, weight: {weight}, result: {result}")

    return results


parallel_work(1,501)