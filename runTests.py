import os
import time
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

K = [7, 9, 10, 11]; P = [5, 6, 7]; PAA = [3, 6, 11]

pathDataset = './Dataset/MLTollsStackOverflow.csv'

labels = list(); kapra_time = list(); IVLs = list()
min = float('inf')

parameters = {'K' : 0, 'P' : 0, 'PAA' : 0}

for k in K:
    for p in P:
        if p >= k:
            continue
        for paa in PAA:
            start = time.time()
            os.system(f'python3 ./kp-anonymity.py {k} {p} {paa} {pathDataset} | grep IVL >> fileTmp.txt')
            stop = time.time()
            execTime = stop - start
            if min > execTime:
                min = execTime
                parameters['K'] = k; parameters['P'] = p; parameters['PAA'] = paa
            labels.append(f"K={k}\nP={p}\nPAA={paa}")
            kapra_time.append(execTime)

lines = []

with open('fileTmp.txt', 'r') as f:
    lines = f.readlines()

os.system('rm ./fileTmp.txt')

for i in range(0, len(lines)):
    IVLs.append(float(lines[i].split('IVL')[1].strip()))

minVL = float('inf'); minLindex = -1

for i in range(0, len(lines)):
    if float(lines[i].split('IVL')[1].strip()) < minVL:
        minVL = float(lines[i].split('IVL')[1].strip())
        minLindex = i

print("Time:", parameters)
print("IVL:", labels[minLindex].strip(), minVL)

plt.figure(figsize=(25, 7))
plt.bar(labels, kapra_time, width = 0.3)
plt.ylabel("Execution Time(s) ")
plt.savefig('./Output/graphExecutionTime.png')

plt.figure(figsize=(25, 7))
plt.bar(labels, IVLs, width=0.3)
plt.ylabel("Instan Value Loss ")
plt.savefig('./Output/graphInstantValueLoss.png')

