import os
import time
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

K = [7, 9, 11]; P = [5, 7, 9]; PAA = [2, 4, 6]

pathDataset = './Dataset/MLTollsStackOverflow.csv'

labels = list(); kapra_time = list()
min = float('inf')

parameters = {'K' : 0, 'P' : 0, 'PAA' : 0}

for k in K:
    for p in P:
        if p >= k:
            continue
        for paa in PAA:
            start = time.time()
            os.system(f'python3 ./kp-anonymity.py {k} {p} {paa} {pathDataset}')
            stop = time.time()
            execTime = stop - start
            if min > execTime:
                min = execTime
                parameters['K'] = k; parameters['P'] = p; parameters['PAA'] = paa
            labels.append(f"K={k}\nP={p}\nPAA={paa}")
            kapra_time.append(execTime)

print(parameters)
plt.figure(figsize=(25, 7))
plt.bar(labels, kapra_time, width = 0.3)
plt.ylabel("Execution Time(s) ")
plt.savefig('graph.png')


