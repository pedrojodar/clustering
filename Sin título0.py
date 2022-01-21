# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 15:10:59 2021

@author: Pedro Jesús Jódar 
"""
import numpy as np
import pandas as pd

test=False


res=pd.read_table('Domain_table_FINAL.tab', header=10, sep='\t')
def clustering (res, index):
    value =[]
    value.append(res.keys()[0])
    res=res.drop(value,axis=1)
    if (test):
        matixnp=np.matrix([[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]])
        res= pd.DataFrame(matixnp)
    
    suma = res.transpose().sum().reset_index(drop=True)
    q=np.matrix(res)*np.matrix(res).transpose()
    print ('q', q)
    np.fill_diagonal(q,0)
    q2=np.matrix(res).transpose()*np.matrix(res)
    grade = np.diag(q2)
    c=np.fromfunction(lambda i, j: (q[i, j]-1)/(np.matrix(suma)[0,i]+np.matrix(suma)[0,j]-q[i,j]-1), (q.shape[0],q.shape[1]), dtype=int)
    
    c_df=pd.DataFrame(c).replace(np.nan,0)
    c_df[c_df<0]=0
    resultado = np.diag(np.matrix(res).transpose()*np.matrix(c_df)*np.matrix(res))
    resultado=resultado/(grade*(grade-1))
    DF=pd.DataFrame([grade, resultado]).transpose().sort_values(by=0).reset_index(drop=True)
    DF=DF.replace(np.nan,0)
    
    print (DF)
    base = np.zeros((int(max(DF[0])),))
    count = np.zeros((int(max(DF[0])),))
    
    for i in range (len(DF[0])):
        base[int(DF[0][i])-1]+=DF[1][i]
        count[int(DF[0][i])-1]+=1
    
    k=[]
    mean=[]
    count_clean=[]
    file_name = 'clustering'+str(index)+'.txt'
    f=open (file_name, 'w')
    for i in range (1,len(base)):
        if (count[i]>0):
            k.append(i+1)
            mean.append(base[i]/count[i])
            count_clean.append(count[i])
            aux_write=str(i+1)+'\t'+str(base[i]/count[i])+'\n'
            f.write(aux_write)
    f.close()
    import matplotlib.pyplot as plt
    plt.plot(k, mean)
    plt.xscale('log')
    plt.yscale('log')
    return DF, k, mean, count_clean

DDF, k_2, mean_2, counter_2 = clustering(res,1)
DDF_3, k_3, mean_3, counter_3 = clustering(res.drop(res.keys()[0], axis=1).transpose(),2)
import matplotlib.pyplot as plt
plt.show()
plt.plot(k_2, counter_2)
plt.plot(k_3, counter_3)
plt.xscale('log')
plt.yscale('log')

f2 = open ('genome','w')
for i in range (len(k_2)):
    f2.write(str(k_2[i])+'\t'+str(counter_2[i])+'\n')
f2.close()
f3 = open ('gen.txt','w')
for i in range (len(k_3)):
    f3.write(str(k_3[i])+'\t'+str(counter_3[i])+'\n')
f3.close()
