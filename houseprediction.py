#!/usr/bin/env python
# coding: utf-8

# In[107]:


import numpy as np
import copy,math
import matplotlib.pyplot as plt

np.set_printoptions(precision=2)


# In[73]:


x_train=np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train=np.array([460, 232, 178])
def cost_function(x,y,w,b):
    tot=0.0
    m=x.shape[0]
    for i in range(m):
        f_wb=np.dot(w,x[i])+b
        err=f_wb-y[i]
        sq=err**2
        tot+=sq
    tot=tot/(2*m)
    return tot


# In[74]:


b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
cost = cost_function(x_train, y_train, w_init, b_init)
print(f'Cost at optimal w : {cost}')


# In[75]:


def gradient(x,y,w,b):
    dj_dw=np.zeros(x.shape[1])
    dj_db=0
    m=x.shape[0]
    n=x.shape[1]
    for i in range(m):
        f_wb=np.dot(w,x[i])+b
        err=f_wb-y[i]
        for j in range(n):
            dj_dw[j]=dj_dw[j]+(err*x[i][j])
        dj_db=dj_db+err
    dj_dw=dj_dw/m
    dj_db=dj_db/m
    return dj_dw,dj_db



# In[76]:


tmp_dj_db, tmp_dj_dw = gradient(x_train, y_train, w_init, b_init)
print(f'dj_db at initial w,b: {tmp_dj_db}')
print(f'dj_dw at initial w,b: \n {tmp_dj_dw}')


# In[116]:


def gradient_descent(alpha,gradientt,x,y,w,b,num_iters,compute_cost):
    j_hist=[]
    iterl=[]
    warr=copy.deepcopy(w)
    bloc=b
    for i in range(num_iters):
        dj_dw,dj_db=gradientt(x,y,warr,bloc)
        warr=warr-alpha*dj_dw
        bloc=bloc-alpha*dj_db
        cost=compute_cost(x,y,warr,bloc)
        j_hist.append(cost)
        iterl.append(i)
    return warr,bloc,j_hist,iterl


# In[126]:


def z_score(x):
    mu=np.mean(x,axis=0)
    sigma=np.std(x,axis=0)
    z_sc=(x-mu)/sigma
    return z_sc


# In[136]:


x_train=np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train=np.array([460, 232, 178])
initial_w = np.zeros_like(x_train[0])
initial_b = 0.
z_sc=z_score(x_train)
w_final, b_final,jhist,iterl = gradient_descent(0.01, gradient, z_sc, y_train, initial_w, initial_b, 1000,cost_function)
x_predict=np.array([1200,3,1,40])
y_predict=np.dot(x_predict,w_final)+b_final
print(f"Cost of requested house is:{y_predict}")
plt.plot(iterl,jhist)
#plt.yscale('log')
plt.show()


