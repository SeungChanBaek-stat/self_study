# 최송 샘플 튜플 확인
import sys, os
sys.path.append(os.getcwd())
from functions.preprocess import LunaDataset
import time

X = LunaDataset()
t0 = time.time()
total_t = 0.0
for i in range(1_000):
    temp = X[i]
    t1 = time.time()
    total_t += (t1 - t0)
    if i % 100 == 0:
        
        print(f"캐시 이전 {i} 번째 인스턴스 접근 : {total_t:.4f} s elapsed")
    t0 = time.time()
    
    
    
t0 = time.time()
total_t = 0.0
for i in range(1_000):
    temp = X[i]
    t1 = time.time()
    total_t += (t1 - t0)
    if i % 100 == 0:
        
        print(f"캐시 이후 {i} 번째 인스턴스 접근 : {total_t:.4f} s elapsed")
    t0 = time.time()