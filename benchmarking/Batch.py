import subprocess
from sys import argv
import time
import os

# 11325 batches
start = int(argv[1])
end = int(argv[2])
tic = time.time()
path = "cache/distances_batch/early_benchmarking_crema"
todo = []
for i in range(start, end):
    if not os.path.exists("%s/%i.h5"%(path, i)):
        todo.append(i)
print(len(todo))

for i in todo:
    subprocess.call(["python", "EarlyFusion.py", "-s", "benchmarking_crema", "-d", "../features_benchmark", "-c", "crema", "-i", "%i"%i])
    t = time.time()-tic
    print("Total Elapsed Time: %.3g"%(t))
    print("Avg time per iteration: %.3g"%(t/(i-start+1)))