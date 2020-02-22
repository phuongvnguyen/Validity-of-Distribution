#!/usr/bin/env python
# coding: utf-8

# $$\small \color{green}{\textbf{Validating the distributions in modelling the Volatilies of financial assets}}$$ 
# 
# 
# $$\large \color{blue}{\textbf{Phuong Van Nguyen}}$$
# $$\small \color{red}{\textbf{ phuong.nguyen@summer.barcelonagse.eu}}$$
# 
# 
# This computer program was written by Phuong V. Nguyen, based on the $\textbf{Anacoda 1.9.7}$ and $\textbf{Python 3.7}$.
# 
# $$\text{1. Issue}$$
# Once modelling the volatilities of financial assets, three distributions are typically used, including 
# 
# 1. The normal distribution.
# 2. The Student’s T distribution.
# 3. The Standardized Skew Student’s T. 
# 
# It is worth noting that financial returns are typically heavy tailed, and a Student’s T distribution is a simple method to capture this feature. However, this mini project attempts to evaluate the fit of these three distributions in modelling the volatilities of financial asset. 
# 
# $$\text{2. Methodology}$$ 
# 
# 
# To this end, I develop and estimate the Threshold ARCH (TARCH) Model using the daily closed price of the Vingroup stock.
# 
# $$\text{3. Dataset}$$ 
# 
# One can download the dataset used to replicate my project at my Repositories on the Github site below
# 
# 
# 
# # Preparing Problem
# 
# ##  Loading Libraries

# In[31]:


import warnings
import itertools
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima_model import ARIMA
#import pmdarima as pm
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
import matplotlib
from arch import arch_model
from arch.univariate import GARCH
from collections import OrderedDict


# ## Defining some varibales for printing the result

# In[2]:


Purple= '\033[95m'
Cyan= '\033[96m'
Darkcyan= '\033[36m'
Blue = '\033[94m'
Green = '\033[92m'
Yellow = '\033[93m'
Red = '\033[91m'
Bold = "\033[1m"
Reset = "\033[0;0m"
Underline= '\033[4m'
End = '\033[0m'


# ##  Loading Dataset

# In[3]:


data = pd.read_excel("data.xlsx")


# # Data Exploration and Preration
# 
# ## Data exploration

# In[4]:


data.head(5)


# ## Computing returns
# ### Picking up the close prices

# In[5]:


closePrice = data[['DATE','CLOSE']]
closePrice.head(5)


# ### Computing the daily returns

# In[6]:


closePrice['Return'] = closePrice['CLOSE'].pct_change()
closePrice.head()


# In[7]:


daily_return=closePrice[['DATE','Return']]
daily_return.head()


# ### Reseting index

# In[8]:


daily_return =daily_return.set_index('DATE')
daily_return.head()


# In[9]:


daily_return = 100 * daily_return.dropna()
daily_return.head()


# In[10]:


daily_return.index


# ### Plotting returns

# In[13]:


sns.set()
fig=plt.figure(figsize=(12,5))
plt.plot(daily_return.Return['2007':'2013'],LineWidth=1)
plt.autoscale(enable=True,axis='both',tight=True)
#plt.grid(linestyle=':',which='both',linewidth=2)
fig.suptitle('The Log Daily Returns', fontsize=15,fontweight='bold')
plt.title('19/09/2007- 31/12/2016',fontsize=10,fontweight='bold',color='k')
plt.ylabel('Return (%)',fontsize=10)
plt.xlabel('Source: The Daily Close Price-based Calculations',fontsize=10,fontweight='normal',color='k')


# ### Checking the Skewness and Kurtosis
# #### Desriptive statistices

# In[20]:


print('The Skewness: {}'.format(round(daily_return.Return.skew(),2)))
print('The Kurtosis: {}'.format(round(daily_return.Return.kurtosis(),2)))


# #### Visualization

# In[23]:


sns.set()
fig=plt.figure(figsize=(12,5))
sns.distplot(closePrice[['Return']],bins=150)
plt.autoscale(enable=True,axis='both',tight=True)
#plt.grid(linestyle=':',which='both',linewidth=2)
fig.suptitle('The Histogram of The Daily Returns', 
             fontsize=15,fontweight='bold',color='b')
plt.title('19/09/2007- 30/12/2016',fontsize=10,fontweight='bold',color='b')
plt.ylabel('Percent (%)',fontsize=10)
#plt.xlabel('Source: The Daily Close Price-based Calculations',fontsize=17,fontweight='normal',color='k')


# $$\textbf{Comments:}$$
# 
# - First, the distribution of the return shows a negative (left) skewness.
# 
# - Second, the distribution of the return shows a positive kurtusis. Accordingly, the peak is stepper and/or the tails are much fatter.
# 
# It implies either the Student’s T distribution or the Standardized Skew Student’s T captures this fact.
# 
# # Modelling TARCH model
# 
# $$\text{Mean equation:}$$
# $$r_{t}=\mu + \epsilon_{t}$$
# 
# $$\text{Conditional Volatility (Variance) equation:}$$
# $$\sigma^{k}_{t}= \omega + \alpha |\epsilon_{t}|^{k} + \gamma |\epsilon_{t-1}|^{k} \mathbf{I}_{[\epsilon_{t-1}<0]}+\beta\sigma^{k}_{t-1}$$
# 
# $$\text{where:}$$
# 
# 
# $$\epsilon_{t}= \sigma_{t} e_{t}$$
# 
# $$e_{t} \sim N(0,1)$$
# 
# $$\mathbf{I} $$ $$\text{is an indicator function that takes the value 1 when its argument is true}$$
# 
# ### The normal distribution

# In[43]:


tarch_norm= arch_model(daily_return, p=1, o=1, q=1, power=1.0, dist='normal')
results_norm = tarch_norm.fit(update_freq=1, disp='on')
print(results_norm.summary())


# ## The Standard Student’s T distribution

# In[27]:


tarch_standT= arch_model(daily_return, p=1, o=1, q=1, power=1.0, dist='t')
results_standT = tarch_standT.fit(update_freq=1, disp='on')
print(results_standT.summary())


# ## The Standardized Skew Student’s T distribution

# In[29]:


tarch_standST= arch_model(daily_return, p=1, o=1, q=1, power=1.0, dist='skewt')
results_standST = tarch_standST.fit(update_freq=1, disp='on')
print(results_standST.summary())


# # Comparing the Log-likelihood

# In[42]:


lls = pd.Series(
    OrderedDict((('normal', results_norm.loglikelihood),
                 ('t', results_standT.loglikelihood), ('skewt',
                                              results_standST.loglikelihood))))
print('--------------------------------------')
print(Bold+'The optimal log-likelihood'+End)
print('--------------------------------------')
print(lls)
params = pd.DataFrame(
    OrderedDict((('normal', results_norm.params), ('t', results_standT.params),
                 ('skewt', results_standST.params))))
print('--------------------------------------')
print(Bold+'The parameters'+End)
print('--------------------------------------')
print(params)
print('--------------------------------------')


# $$\textbf{Comments:}$$
# 
# As analyzing before, we find that the normal has a much lower log-likelihood (-5390.66) than either the Standard Student’s T or the Standardized Skew Student’s T – however, these two are fairly close (-4939.55 and -4939.42, respectively). The closeness of the T and the Skew T indicate that returns are not heavily skewed.
# 
# # Comparing the residuals and conditional variances
# ## The residuals

# In[51]:


fig=plt.figure(figsize=(10,5))
plt.plot(results_norm.resid,LineWidth=1,label='normal')
plt.plot(results_standT.resid,LineWidth=1,label='student t ')
plt.plot(results_standST.resid,LineWidth=1,label='student t skew ')
plt.autoscale(enable=True,axis='both',tight=True)
#plt.grid(linestyle=':',which='both',linewidth=2)
fig.suptitle('The Residuals of Mean Equation', 
             fontsize=15,fontweight='bold',
            color='b')
plt.title('19/09/2007- 31/12/2013',fontsize=10,
          fontweight='bold',color='b')
plt.ylabel('Residuals',fontsize=10,color='k')
plt.legend()


# ## Conditional volatility

# In[50]:


fig=plt.figure(figsize=(10,5))
plt.plot(results_norm.conditional_volatility,LineWidth=1,
        label='normal')
plt.plot(results_standT.conditional_volatility,LineWidth=1,
        label='student t ')
plt.plot(results_standST.conditional_volatility,LineWidth=1,
        label='student t skew ')
plt.autoscale(enable=True,axis='both',tight=True)
#plt.grid(linestyle=':',which='both',linewidth=2)
fig.suptitle('The TARCH-based Conditional Volatilities', 
             fontsize=15,fontweight='bold',
            color='b')
plt.title('19/09/2007- 31/12/2016',fontsize=10,
          fontweight='bold',color='b')
plt.ylabel('Volatilies',fontsize=10,color='b')
plt.legend()


# ## Residuals standardized by conditional volatility

# In[52]:


fig=plt.figure(figsize=(10,5))
plt.plot(results_norm.std_resid,LineWidth=1,label='normal')
plt.plot(results_standT.std_resid,LineWidth=1,label='student t ')
plt.plot(results_standST.std_resid,LineWidth=1,label='student t skew ')
plt.autoscale(enable=True,axis='both',tight=True)
#plt.grid(linestyle=':',which='both',linewidth=2)
fig.suptitle('The TARCH-based Residuals standardized by conditional volatility', 
             fontsize=15,fontweight='bold',
            color='b')
plt.title('19/09/2007- 31/12/2013',fontsize=10,
          fontweight='bold',color='b')
plt.ylabel('Volatilies',fontsize=10,color='b')
plt.legend()

