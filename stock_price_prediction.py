#!/usr/bin/env python
# coding: utf-8

# In[146]:


import pandas_datareader as pdr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import plot

from plotly.offline import download_plotlyjs, init_notebook_mode,plot,iplot
init_notebook_mode(connected=True)

import warnings
warnings.filterwarnings('ignore')

key ='paste your key here (read readme file for more info)'


# In[147]:


df = pdr.get_data_tiingo('AAPL',api_key=key)


# In[148]:


df.to_csv('AAPL.csv')


# In[149]:


df= pd.read_csv('AAPL.csv')


# In[150]:


df.tail()


# In[151]:


dfclose = df.reset_index()['close']


# In[152]:


dfclose.shape


# In[153]:


plt.plot(dfclose)


# In[154]:


from sklearn.preprocessing import MinMaxScaler


# In[155]:


scaler = MinMaxScaler(feature_range=(0,1))


# In[156]:


dfclose=scaler.fit_transform(np.array(dfclose).reshape(-1,1))


# In[157]:


print(dfclose)


# In[158]:


## spliting dataset into train and test 
training_size= int (len(df.close)*0.65)
test_size = int (len(dfclose)-training_size)

train_data,test_data = dfclose[0:training_size,:],dfclose[training_size:len(dfclose),:1]


# In[159]:


def create_dataset(dataset,timestep=1):
    #converting array of values into a dataset matrix 
    dataX, dataY =[], []
    for i in range (len(dataset)-timestep-1):
        a= dataset[i:(i+timestep),0]
        dataX.append(a)
        dataY.append(dataset[i+timestep,0])
    return np.array(dataX) , np.array(dataY)


# In[160]:


#reshape into X=t1,t+1,t+2 .... and Y= t+4
timestep =100

X_train, Y_train = create_dataset(train_data,timestep)
X_test,Y_test = create_dataset(test_data,timestep)


# In[161]:


print(X_train)


# In[ ]:





# In[162]:


print(X_test)


# In[163]:


#converting input to be [sample timestep,features] required for LTSM
X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],1)
X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],1)


# In[164]:


# creating stack LTSM model 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


# In[165]:


model = Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')


# In[166]:


model.summary()


# In[167]:


model.fit(X_train,Y_train,validation_data=(X_test,Y_test),epochs=100,batch_size=64,verbose=1)


# In[168]:


#prediction and check performance matrix
train_predict= model.predict(X_train)
test_predict=model.predict(X_test)


# In[169]:


train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)


# In[170]:


import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(Y_train,train_predict))


# In[171]:


math.sqrt(mean_squared_error(Y_test,test_predict))


# In[172]:


look_back=100
trainpredictplot=np.empty_like(dfclose)
trainpredictplot[:,:]=np.nan
trainpredictplot[look_back:len(train_predict)+look_back,:]=train_predict

testpredictplot = np.empty_like(dfclose)
testpredictplot[:,:]=np.nan
testpredictplot[len(train_predict)+(look_back*2)+1:len(dfclose)-1,:]=test_predict

plt.plot(scaler.inverse_transform(dfclose))
plt.plot(trainpredictplot)
plt.plot(testpredictplot)
plt.show()


# In[173]:


x_input = test_data[len(test_data)-100:].reshape(1,-1)
x_input.shape


# In[174]:


temp_input =list(x_input)
temp_input=temp_input[0].tolist()


# In[175]:


temp_input


# In[176]:


from numpy import array


# In[177]:


listout= []
steps=100
i=0
while (i<30):
    if(len(temp_input)>100):
        x_input=np.array(temp_input[1:])
        print("{}day input {}".format(i,x_input))
        x_input = x_input.reshape(1,-1)
        x_input=x_input.reshape((1,steps,1))
        
        yhat=model.predict(x_input,verbose=0)
        print("{}day input {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input = temp_input[1:]
        listout.extend(yhat.tolist())
        i=i+1
        
    else:
        x_input=x_input.reshape((1,steps,1))
        yhat=model.predict(x_input,verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        listout.extend(yhat.tolist())
        i=i+1
        
print(listout)
        


# In[178]:


day_new = np.arange(1,101)
day_pred=np.arange(101,131)


# In[179]:


len(dfclose)


# In[180]:


df2=dfclose.tolist()
df2.extend(listout)


# In[181]:


plt.plot(day_new,scaler.inverse_transform(dfclose[len(dfclose)-100:]))
plt.plot(day_pred,scaler.inverse_transform(listout))


# In[182]:



plt.plot(day_new,scaler.inverse_transform(dfclose[len(dfclose)-100:]))
plt.plot(day_pred,scaler.inverse_transform(listout))


# In[ ]:




