#!/usr/bin/env python
# coding: utf-8

# In[50]:


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import quandl as q
import seaborn as sns


# In[51]:


q.ApiConfig.api_key = "SzKZrxMAKxcEgThrWNFp"


# In[52]:


q


# In[53]:


df = q.get("EOD/AAPL")


# In[54]:


df


# In[55]:


df['HL_PCT']=(df["Adj_High"])-df["Adj_Low"]/(df["Adj_Close"])*100


# In[56]:


df['PCT_Change']=(df["Adj_Close"])-df["Adj_Open"]/(df["Adj_Open"])*100


# In[57]:


df.head()


# In[58]:


corr =df.corr()


# In[59]:


mask =np.triu(np.ones_like(corr, dtype=bool))
f, ax=plt.subplots(figsize=(11,9))
cmap=sns.diverging_palette(230,20,as_cmap=True)
sns.heatmap(corr,mask=mask,cmap=cmap,vmax=.3,center=0,square=True,linewidth=.5,cbar_kws={"shrink":.5})


# In[60]:


df


# In[61]:


df=df[["Adj_Volume","HL_PCT","PCT_Change","Adj_Close"]]


# In[62]:


df.corr()


# In[63]:


df['Adj_Close'].PCT_Change().plot(figsize =(10,8),grid=True)


# In[70]:


print(df.shape)
label =df["Adj_Close"].shift(-10)


# In[71]:


print(label.isnull().sum())
print(label.shape)
print(df["Adj_Close"])
print(label)     
label.dropna(inplace=True)
y=np.array(label)
print(y.shape)


# In[72]:


X=(df[["Adj_Volume","HL_PCT","PCT_Change"]])
X_lately=X[-10:]
X=X[:-10]



print(X.shape)
print(X_lately.shape)


# In[73]:



X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3,random_state=41)
from sklearn.preprocessing import MinMaxScaler
norm =MinMaxScaler().fit(X_train)
X_train =norm.transform(X_train)
X_test =norm.transform(X_test)
X_lately=norm.transform(X_lately)


# In[83]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit( X_train, y_train )
score =reg.score( X_test, y_test)


# In[84]:


score


# In[92]:


print = (reg.intercept_)
print =(len(reg.coef_))


# In[99]:


coefdf =pd.DataFrame(zip(X.columns,reg.coef_))
coefdf


# In[100]:


forcast=[]
forcast=reg.predict(X_lately)
y_pred=reg.predict(X_test)


# In[101]:


forcast


# In[102]:


y_pred


# In[106]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test,y_pred)
mse


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




