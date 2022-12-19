import os
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

K = [9, 10, 11]; P = [5, 6, 7]; PAA = [3, 6, 11]

pathDataset = './Dataset/MLTollsStackOverflow.csv'

labels = list(); kapra_time = list(); IVLs = list()
min_time = float('inf')

parameters = {'K' : 0, 'P' : 0, 'PAA' : 0}

# Try different value for K and P, skipping case: P >= K
# Save execution time
# Save result appending IVL to a fileTmp
for k in K:
    for p in P:
        if p < k:
            for paa in PAA:
                print("ITERAZIONE: ", k, p, paa)
                start = time.time()
                os.system(f'python3 ./kp-anonymity.py {k} {p} {paa} {pathDataset} | grep IVL >> fileTmp.txt')
                stop = time.time()
                execTime = stop - start
                if min_time > execTime:
                    min_time = execTime
                    parameters['K'] = k; parameters['P'] = p; parameters['PAA'] = paa
                labels.append(f"K={k}\nP={p}\nPAA={paa}")
                kapra_time.append(execTime)


# Read fileTmp with result previously saved inside
lines = []
with open('fileTmp.txt', 'r') as f:
    lines = f.readlines()

os.system('rm ./fileTmp.txt')

# to chart purpose
IVLs = [float(lines[i].split('IVL: ')[1].strip()) for i in range (0, len(lines))]

# Print and plot output
print("Time: ", parameters)
print("IVL: ", labels[IVLs.index(min(IVLs))], min(IVLs))

plt.figure(figsize=(25, 7))
plt.bar(labels, kapra_time, width = 0.3)
plt.ylabel("Execution Time(s) ")
plt.savefig('./Output/graphExecutionTime.png')

plt.figure(figsize=(25, 7))
plt.bar(labels, IVLs, width=0.3)
plt.ylabel("Instan Value Loss ")
plt.savefig('./Output/graphInstantValueLoss.png')

