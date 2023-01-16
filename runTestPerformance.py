import os
import time
import matplotlib.pyplot as plt
import pandas as pd

data_slice = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1300, 1500]

k = 9; p = 7; paa = 6

pathDataset = './Dataset/NIFTY50-1_day_with_indicators.csv'

labels = list(); kapra_time = list()
min_time = float('inf')

# Save execution time
for d in data_slice:
    print("ITERAZIONE: ", d)
    df = pd.read_csv(pathDataset)
    if len(df.index) not in data_slice:
        data_slice.append(len(df.index))
    df = df.head(d)
    df.to_csv('./Dataset/tmpDataset.csv', index=False)
    start = time.time()
    os.system(f'python3 ./kp-anonymity.py {k} {p} {paa} ./Dataset/tmpDataset.csv')
    stop = time.time()
    execTime = stop - start
    if min_time > execTime:
        min_time = execTime
    labels.append(f"size={d}")
    kapra_time.append(execTime)
    os.system('rm ./Dataset/tmpDataset.csv')


plt.figure(figsize=(25, 7))
plt.grid()
plt.xlabel("Len of dataset")
plt.ylabel("Execution Time(s) ")
print(labels)
print(kapra_time)
plt.plot(labels, kapra_time)
plt.savefig('./Output/graphPerformance.png')
