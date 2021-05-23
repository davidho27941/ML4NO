import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

#總資料組數
N = 1000000

ve_dune = []
vu_dune = []
vebar_dune = []
vubar_dune = []
ve_t2hk = []
vu_t2hk = []
vebar_t2hk = []
vubar_t2hk = []
theta12 = []
theta13 = []
theta23 = []
delta = []
sdm = []
ldm = []
octant = []
cpv = []
mo = []

#Import txt檔
f = open("n1000000.txt")

for i in range(N):
    s = f.readline().split()
    array = []
    for j in range(len(s)) :
        array.append(float(s[j]))
    ve_dune.append(array[0:50])
    vu_dune.append(array[65:115])
    vebar_dune.append(array[130:180])
    vubar_dune.append(array[195:245])
    ve_t2hk.append(array[260:271])
    vu_t2hk.append(array[271:282])
    vebar_t2hk.append(array[282:293])
    vubar_t2hk.append(array[293:304])
    theta12.append(array[304])
    theta13.append(array[305])
    theta23.append(array[306])
    delta.append(array[307])
    sdm.append(array[308])
    ldm.append(array[309])
    octant.append(array[310])
    cpv.append(array[311])
    mo.append(array[312])   

#輸出全部資料為npz檔
np.savez("neutrino0523n1000000",
         ve_dune = ve_dune,
         vu_dune = vu_dune,
         vebar_dune = vebar_dune,
         vubar_dune = vubar_dune,
         ve_t2hk = ve_t2hk,
         vu_t2hk = vu_t2hk,
         vebar_t2hk = vebar_t2hk,
         vubar_t2hk= vubar_t2hk,
         theta12 = theta12,
         theta13 = theta13,   
         theta23 = theta23,   
         delta = delta,
         sdm = sdm,   
         ldm = ldm,  
         octant = octant,
         cpv = cpv,   
         mo = mo  
        )
