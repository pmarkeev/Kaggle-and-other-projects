# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 21:33:29 2021

@author: MarkeevP
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import glob

folder = r'83-88'
data_files = glob.glob(folder+'/*.csv')
text_files = glob.glob(folder+'/*.txt')
#%% read data files and appropriate energy values (x values)
#function to read text file with energy axis 
def read_txt(file):
    data = [pd.read_csv(i, skiprows=16, sep='\s+', usecols=['trajectory_1_1']) for i in file]
    data = [list(i['trajectory_1_1']) for i in data] #make a list instaed of dataframe 
    return data

txt_data = read_txt(text_files) 
txt_data = [i[1:] for i in txt_data] #drop first point, because it's compromised in all files

#func to read data files, creates list of data frames 
def read_data(file):
    data = [pd.read_csv(i) for i in file]
    return data

data = read_data(data_files)
data = [i.iloc[1: , :] for i in data]

#%% Bring data to 0
def bring_to_0(data):
    col = list(data.columns)
    for i in col:
        data[i] = data[i] - data[i].min()
    return data

for i in range(len(data)):
    data[i]=bring_to_0(data[i])    

#%% plot everythin to check if it looks good

def plot_allbyone(data, txt_data, ax):
     ax.plot(data)
     x_pos = list(data.index)[::9] 
     x_val = ["%.1f"%i for i in txt_data[::9]]
     ax.set_xticks(x_pos)
     ax.set_xticklabels(x_val)
     ax.legend(data.columns, loc='upper left')
     #return plt.show()

fig, ax = plt.subplots(nrows=2, ncols=3, sharey='row', figsize=(16,10))
ax = ax.flatten() 
for i in range(len(data)):
    plot_allbyone(data[i], txt_data[i], ax[i])
      
      
plt.show()   
#%% array with results to 4*6 DF with propper indexes
def arraytoDF(array):
    res = pd.DataFrame({0:array[0:4], 1:array[4:8], 2:array[8:12], 3:array[12:16], 
                        4:array[16:20], 5:array[20:]})
    res.index = list(data[0].columns)
    return res
 
#%% find where differencial is maximum
dx = 0.01
def find_diff_pos(data):
    dspectra = np.diff(data)/dx
    co_ind = np.where(dspectra == dspectra.max())
    co_ind = co_ind[0].item()   
    return co_ind

ind_result=[]
for i in range(len(data)):
    for col in data[i]:
        ind_result.append(find_diff_pos(data[i][col]))

ind_result = arraytoDF(ind_result)
#print(ind_result) 

#%% define fitting positions for each set (x,y), fit and save coef(ax), interceptY(bx) and interceptX(x_int)
x_int=[]
ax=[]
bx=[]
def fitting(data, txt_data, ind_result):

    for a, b in zip(data, ind_result):
        x = txt_data[b-1:b+2]
        x = np.array(x).reshape(-1,1)
        y = data[a][b-1:b+2]
        y = np.array(y).reshape(-1,1)
        fit1 = LinearRegression().fit(x, y) 
        ax.append(fit1.coef_.item())
        bx.append(fit1.intercept_.item())
        x_int.append(-fit1.intercept_.item()/fit1.coef_.item())
    return ax, bx, x_int

for i in range(len(data)):
    (ax, bx, x_int) = fitting(data[i], txt_data[i], ind_result[i])

ax_res = arraytoDF(ax)
bx_res = arraytoDF(bx)
x_int_res = arraytoDF(x_int)

#print(ax_res, bx_res, x_int_res)

#%% 
#!!!Probably save ax and bx instead of final result and then plot it and then find x_int
for j in list(range(len(data))):
    fig, ax = plt.subplots()
    ax.plot(txt_data[j], data[j], alpha=0.5)
    for i in list(range(4)):
        fx= np.array(txt_data[j][ind_result[j][i]-8:ind_result[j][i]+8])
        fy= ax_res[j][i]*fx + bx_res[j][i]
        ax.plot(fx, fy, 'r--')
    ax.legend(data[j].columns)
    plt.show()
#%% plot everythin to check if it looks good
# fig, ax = plt.subplots()
# def plot_allbyone(data, txt_data, ax):
#      ax.plot(data)
#      x_pos = list(data.index)[::9] 
#      x_val = ["%.1f"%i for i in txt_data[::9]]
#      ax.set_xticks(x_pos)
#      ax.set_xticklabels(x_val)
#      #return plt.show()

# fig, ax = plt.subplots(nrows=2, ncols=3, sharey='row')
# ax = ax.flatten() 
# for i in range(6):
#     plot_allbyone(data[i], txt_data[i], ax[i])

#%% 
from sklearn import preprocessing
n=2
for i in list(range(len(data))):
    #norm_data = preprocessing.normalize([np.array(list(data[i].iloc[:,n]))])
    col = data[i].columns
    plt.plot(txt_data[i], data[i].iloc[:,n], label = col[n])   
    #plt.plot(txt_data[i], norm_data[0], label = col[n])
plt.xlim(0, 3)
plt.legend()
plt.show()
