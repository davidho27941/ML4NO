# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session






#總資料組數
N = 10000

#Import txt檔
f = open("/kaggle/input/test0322n10000/test0322n10000.txt")

#將txt檔中數列分割
s = f.read().split()    

#將''轉換成數字
array = []
for i in range(len(s)) :
    array.append(float(s[i]))

#將N組data分割
array_s = np.array_split(array, N)


#將每組資料加入標籤，存進data
data = {}
for i in range(N):
    data[i] = {
        've1': array_s[i][0:65],      #65  # Spectrum for electron neutrino      in exp.1
        'vu1': array_s[i][65:130],    #65  # Spectrum for muon neutrino          in exp.1
        've_1': array_s[i][130:195],  #65  # Spectrum for electron anti-neutrino in exp.1
        'vu_1': array_s[i][195:260],  #65  # Spectrum for muon anti-neutrino     in exp.1
        've2': array_s[i][260:271],   #11  # Spectrum for electron neutrino      in exp.2
        'vu2': array_s[i][271:282],   #11  # Spectrum for muon neutrino          in exp.2
        've_2': array_s[i][282:293],  #11  # Spectrum for electron anti-neutrino in exp.2
        'vu_2': array_s[i][293:304],  #11  # Spectrum for muon anti-neutrino     in exp.2
        'para1': array_s[i][304:310], #6   # theta12, theta13, theta23, delta_cp, sdm, ldm
        'para2': array_s[i][310:313]  #3   # OCTANT, CPV, MO
    }


#將dict檔轉換為dataframe檔
df = pd.DataFrame(data)


#將不同組樣本的相同標籤資料取出彙整
ve1 = []
vu1 = []
ve_1 = []
vu_1 = []
ve2 = []
vu2 = []
ve_2 = []
vu_2 = []
theta12 = []
theta13 = []
theta23 = []
delta = []
sdm = []
ldm = []
octant = []
cpv = []
mo = []

for i in range(N):
    ve1.append(df[i]['ve1'])
    vu1.append(df[i]['vu1'])
    ve_1.append(df[i]['ve_1'])
    vu_1.append(df[i]['vu_1'])
    ve2.append(df[i]['ve2'])
    vu2.append(df[i]['vu2'])
    ve_2.append(df[i]['ve_2'])
    vu_2.append(df[i]['vu_2'])    
    theta12.append(df[i]['para1'][0])
    theta13.append(df[i]['para1'][1])
    theta23.append(df[i]['para1'][2])
    delta.append(df[i]['para1'][3])
    sdm.append(df[i]['para1'][4])
    ldm.append(df[i]['para1'][5])   
    octant.append(df[i]['para2'][0])
    cpv.append(df[i]['para2'][1])
    mo.append(df[i]['para2'][2])


#輸出全部資料為npz檔
np.savez("neutrino0323n10000v2",
         ve1 = ve1,
         vu1 = vu1,
         ve_1 = ve_1,
         vu_1 = vu_1,
         ve2 = ve2,
         vu2 = vu2,
         ve_2 = ve_2,
         vu_2 = vu_2,
         theta12 = theta12,
         theta13 = theta13,   
         theta23 = theta23,   
         delta = delta,
         sdm = sdm,   
         ldm = ldm,  
         octant = octant,
         cpv = cpv,   
         mo = mo,  
        )


