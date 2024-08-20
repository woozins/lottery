import itertools
import random


def find_combinations(current_comb, start, result):
    if len(current_comb) == 6:
        result.append(tuple(current_comb))
        return
    
    for i in range(start, 46):
        if not current_comb or i - current_comb[-1] >= 7:
            find_combinations(current_comb + [i], i + 1, result)

result = []
find_combinations([], 1, result)

freqgroup = result 

P = 1000
plusfreq = list(map(tuple,[sorted(random.sample(range(1,46), 6)) for _ in range(P)]))

freqgroup = freqgroup + plusfreq

numbers = list(range(1,46))

combinations = list(itertools.combinations(numbers, 6))

othersgroup = list(set(combinations) - set(freqgroup))

group = [freqgroup, othersgroup]
