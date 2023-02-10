from joblib import Parallel, delayed
import time
def process(i):
    return (i-2)**2 + (i-3)**2

s = time.time()
results = Parallel(n_jobs=1)(delayed(process)(i) for i in range(1024))
e = time.time()
print('okay',e-s)  # prints [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
