#!/usr/bin/env python
# coding: utf-8

# In[1]:



import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
 
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from typing import List
import pandas as pd
import numpy as np
import math ,time

import math, time
from sklearn.metrics import mean_squared_error


# In[2]:


dfa4= pd.read_csv(r"../../../../Taipei_14.csv")
dfa5= pd.read_csv(r"../../../../Taipei_15.csv")
dfa6= pd.read_csv(r"../../../../Taipei_16.csv")
dfa7= pd.read_csv(r"../../../../Taipei_17.csv")
dfa8= pd.read_csv(r"../../../../Taipei_18.csv")


# In[12]:


a4=dfa4[[ 'AMB_TEMP', 'CH4',
       'CO', 'NMHC', 'NO', 'NO2', 'NOx', 'O3', 'PH_RAIN', 'PM10', 'PM2.5',
       'RAINFALL', 'RAIN_COND', 'RH', 'SO2', 'THC', 'UVB', 'WD_HR',
       'WIND_DIREC', 'WIND_SPEED', 'WIND_cos', 'WIND_sin', 'WS_HR', 'W_HR_cos',
       'W_HR_sin']]
a5=dfa5[['AMB_TEMP', 'CH4',
       'CO', 'NMHC', 'NO', 'NO2', 'NOx', 'O3', 'PH_RAIN', 'PM10', 'PM2.5',
       'RAINFALL', 'RAIN_COND', 'RH', 'SO2', 'THC', 'UVB', 'WD_HR',
       'WIND_DIREC', 'WIND_SPEED', 'WIND_cos', 'WIND_sin', 'WS_HR', 'W_HR_cos',
       'W_HR_sin']]
a6=dfa6[[ 'AMB_TEMP', 'CH4',
       'CO', 'NMHC', 'NO', 'NO2', 'NOx', 'O3', 'PH_RAIN', 'PM10', 'PM2.5',
       'RAINFALL', 'RAIN_COND', 'RH', 'SO2', 'THC', 'UVB', 'WD_HR',
       'WIND_DIREC', 'WIND_SPEED', 'WIND_cos', 'WIND_sin', 'WS_HR', 'W_HR_cos',
       'W_HR_sin']]
a7=dfa7[['AMB_TEMP', 'CH4',
       'CO', 'NMHC', 'NO', 'NO2', 'NOx', 'O3', 'PH_RAIN', 'PM10', 'PM2.5',
       'RAINFALL', 'RAIN_COND', 'RH', 'SO2', 'THC', 'UVB', 'WD_HR',
       'WIND_DIREC', 'WIND_SPEED', 'WIND_cos', 'WIND_sin', 'WS_HR', 'W_HR_cos',
       'W_HR_sin']]
a8=dfa8[['AMB_TEMP', 'CH4',
       'CO', 'NMHC', 'NO', 'NO2', 'NOx', 'O3', 'PH_RAIN', 'PM10', 'PM2.5',
       'RAINFALL', 'RAIN_COND', 'RH', 'SO2', 'THC', 'UVB', 'WD_HR',
       'WIND_DIREC', 'WIND_SPEED', 'WIND_cos', 'WIND_sin', 'WS_HR', 'W_HR_cos',
       'W_HR_sin']]

r14=(dfa4[dfa4.SiteEngName =='Songshan'])
r15=(dfa5[dfa5.SiteEngName =='Songshan'])
r16=(dfa6[dfa6.SiteEngName =='Songshan'])
r17=(dfa7[dfa7.SiteEngName =='Songshan'])
r18=(dfa8[dfa8.SiteEngName =='Songshan'])

######----------------------------------------------------

y14=r14[['AMB_TEMP', 'CO', 'NO', 'NO2', 'NOx', 'O3', 'PM10', 'PM2.5','RAINFALL', 'RH']]
y15=r15[['AMB_TEMP', 'CO', 'NO', 'NO2', 'NOx', 'O3', 'PM10', 'PM2.5','RAINFALL', 'RH']]
y16=r16[['AMB_TEMP', 'CO', 'NO', 'NO2', 'NOx', 'O3', 'PM10', 'PM2.5','RAINFALL', 'RH']]
y17=r17[['AMB_TEMP', 'CO', 'NO', 'NO2', 'NOx', 'O3', 'PM10', 'PM2.5','RAINFALL', 'RH']]
y18=r18[['AMB_TEMP', 'CO', 'NO', 'NO2', 'NOx', 'O3', 'PM10', 'PM2.5','RAINFALL', 'RH']]

# In[14]:
# Year 14

###################

features_14=y14[['AMB_TEMP', 'CO', 'NO', 'NO2', 'NOx', 'O3', 'PM10', 'PM2.5','RAINFALL', 'RH']]
print(features_14.shape)
reshape_14=features_14.values.reshape((-1,10*1))
data_14=np.array(reshape_14)

#print(data_14.shape)
# Year 15

##################################

features_15=y15[['AMB_TEMP', 'CO', 'NO', 'NO2', 'NOx', 'O3', 'PM10', 'PM2.5','RAINFALL', 'RH']]
print(features_15.shape)
reshape_15=features_15.values.reshape((-1,10*1))
data_15=np.array(reshape_15)
#print(data_15.shape)
# Year 16

############################
features_16=y16[['AMB_TEMP', 'CO', 'NO', 'NO2', 'NOx', 'O3', 'PM10', 'PM2.5','RAINFALL', 'RH']]
#print(features_16.shape)
reshape_16=features_16.values.reshape((-1,10*1))
data_16=np.array(reshape_16)
#print(data_16.shape)
# Year 17
##################################

features_17=y17[['AMB_TEMP', 'CO', 'NO', 'NO2', 'NOx', 'O3', 'PM10', 'PM2.5','RAINFALL', 'RH']]
#print(features_17.shape)
reshape_17=features_17.values.reshape((-1,10*1))
data_17=np.array(reshape_17)
#print(data_17.shape)



###Concatenate 4 years data features together 
c4i=np.concatenate((data_14,data_15,data_16,data_17),axis=0)
print(c4i.shape)


# Put timesteps together
x=c4i
timestep =40
x_build = []

for i in range(x.shape[0] - timestep * 2 ):
    x_build.append(x[i:i+timestep])
    #print(i+timestep,i+timestep+timestep)
train_X=np.array(x_build)

print(train_X.shape)

###############################################
out_14=y14[['PM2.5']]
shape_14=out_14.values.reshape((-1, 1))
d_14=np.array(shape_14)
#print(d_14.shape)
###################################################
out_15=y15[['PM2.5']]
shape_15=out_15.values.reshape((-1, 1))
d_15=np.array(shape_15)
#print(d_15.shape)
#######
out_16=y16[['PM2.5']]
shape_16=out_16.values.reshape((-1, 1))
d_16=np.array(shape_16)
#print(d_16.shape)
########
out_17=y17[['PM2.5']]
shape_17=out_17.values.reshape((-1, 1))
d_17=np.array(shape_17)
#print(d_17.shape)
########

##########################################################
c4o=np.concatenate((d_14,d_15,d_16,d_17),axis=0)
#print(c4o.shape)
print("!!")
x1=c4o
x1_build = []

for i in range(timestep, x1.shape[0] - timestep ):
    x1_build.append(x1[i:i+timestep])
    #print(i+timestep,i+timestep+timestep)
train_y=np.array(x1_build)
#print(train_y.shape)

#####################################################################

# In[28]:

print('train_X.shape, train_y.shape')

print(train_X.shape, train_y.shape)

########################################################################
features_18=y18[['AMB_TEMP', 'CO', 'NO', 'NO2', 'NOx', 'O3', 'PM10', 'PM2.5','RAINFALL', 'RH']]
#print(features_18.shape)
reshape_18=features_18.values.reshape((-1,10*1))
data_18=np.array(reshape_18)
#print(data_18.shape)
x2=data_18
y_build = []
for i in range(x2.shape[0] - timestep - timestep):
    y_build.append(x2[i:i+timestep])
   # print(i+timestep,i+timestep+timestep)
test_X=np.array(y_build)
test_X.shape

############################################

out_18=y18[['PM2.5']]
shape_18=out_18.values.reshape((-1, 1))
d_18=np.array(shape_18)
print(d_18.shape)

x3=d_18
y1_build = []

for i in range(timestep, x3.shape[0] - timestep):
    y1_build.append(x3[i:i+timestep])
    #print(i+timestep,i+timestep+timestep)
test_y=np.array(y1_build)
test_y.shape
##################################################

cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda')


x_train = torch.FloatTensor(train_X)
x_test=torch.FloatTensor(test_X)
y_train = torch.FloatTensor(train_y)
y_test=torch.FloatTensor(test_y)

x_train =x_train.to(device)
x_test=x_test.to(device)
y_train =y_train.to(device)
y_test=y_test.to(device)

print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self, ):
        return self.x.shape[0]
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

train_dataset = MyDataset(x_train, y_train)
teset_dataset = MyDataset(x_test, y_test)

train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_dataloader = DataLoader(teset_dataset, batch_size=10, shuffle=False)

############
dropout=0.05

print('x_train.shape,x_test.shape,y_train.shape,y_test.shape')

print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self, ):
        return self.x.shape[0]
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

train_dataset = MyDataset(x_train, y_train)
teset_dataset = MyDataset(x_test, y_test)

train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_dataloader = DataLoader(teset_dataset, batch_size=10, shuffle=False)


# In[3]:


import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(in_dim, in_dim)
        self.key = nn.Linear(in_dim, in_dim)
        self.value = nn.Linear(in_dim, in_dim)

    def forward(self, x):
        # Assuming x has shape (batch_size, sequence_length, feature_dim)

        # Linear transformations for query, key, and value
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        # Scaled Dot-Product Attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / (query.size(-1) ** 0.5)
        attention_weights = nn.functional.softmax(scores, dim=-1)
        attended_values = torch.matmul(attention_weights, value)

        return attended_values

class TimeSeriesSelfAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TimeSeriesSelfAttentionModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, output_size)
        self.attention = SelfAttention(in_dim=input_size)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Assuming x has shape (batch_size, sequence_length, input_size)
        
        # Self-Attention
        attended_values = self.attention(x)

        # Global Average Pooling to summarize the attended values
       # attended_values = torch.mean(attended_values, dim=1)

        # Fully connected layers
        x = self.fc1(attended_values)
        x = self.relu(x)
        x = self.fc2(x)

        return x

# Example usage:
input_size = 10  # Replace with the actual feature dimension of your time series data
hidden_size = 64  # Adjust as needed
output_size = 1  # Adjust based on your task (e.g., regression or classification)

model = TimeSeriesSelfAttentionModel(input_size, hidden_size, output_size).to(device)


# In[4]:



optimiser = torch.optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 50


criterion = torch.nn.MSELoss(reduction='mean')
optimiser = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser,  patience=500,factor =0.5 ,min_lr=1e-7, eps=1e-08)

import math, time
from sklearn.metrics import mean_squared_error
best_loss = 1e9
#################################
hist = np.zeros(num_epochs)
val=np.zeros(num_epochs)
start_time = time.time()
######################################
num_epochs = 50##########################

for t in range(num_epochs):
    model.train()
    for idx, data in enumerate(train_dataloader):
        x, y_true = data
        y_pred = model(x).to(device)
        print(y_pred.shape,y_true.shape)

        loss = criterion(y_pred, y_true)
        hist[t] += loss.item()
        #wrap_loss = wrap(loss)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    hist[t] /= len(train_dataloader)
    model.eval()
    for idx, data in enumerate(test_dataloader):
        with torch.no_grad():
            x, y_true = data
            y_pred = model(x).to(device)

            loss = criterion(y_pred, y_true)
            val[t] += loss.item()
    val[t] /= len(test_dataloader)
    if best_loss > val[t]:
        best_loss = val[t]
        # TODO: Save model 
        torch.save(model.state_dict(),'Songshan32.pt')
#     scheduler.step(vall_loss)
    #print("Epoch:, loss: %1.5f valid loss:  %1.5f "%(loss.item(),vall_loss.item()))
    print("Epoch ", t, "MSE: ", hist[t].item(),t,"Valid loss",val[t].item())

training_time = time.time()-start_time    
print("Training time: {}".format(training_time))


# In[5]:




model.load_state_dict(torch.load('Songshan32.pt'))
#####################
predict_ary = model(x_test)
#model.to(device)
print (predict_ary.shape)
#print (test_y.shape)
from sklearn.metrics import mean_absolute_percentage_error

def mape(actual, pred): 
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100
from sklearn.metrics import mean_absolute_percentage_error

#print (X_valid.shape)
rmse_score = np.sqrt(np.mean(np.square(predict_ary.cpu().detach().numpy() - y_test.cpu().detach().numpy())))
mae_score = np.mean(np.abs(predict_ary.cpu().detach().numpy() - y_test.cpu().detach().numpy()))
mape_score = mape(predict_ary.cpu().detach().numpy(),y_test.cpu().detach().numpy())
#mae2 = mean_absolute_error(predict_ary, validation_Y[:-3])
print('this is rmse ',rmse_score)
print('this is mape ',mape_score)
print('this is mae ',mae_score)
#####################################
sum([param.nelement() for param in model.parameters()])


# In[ ]:



sum([param.nelement() for param in model.parameters()])


# In[ ]:



import csv
data =[[40,rmse_score,mae_score,mape_score,"Songshan"]]
file = open('Songshan.csv', 'a+', newline ='')

# writing the data into the file
with file:    
    write = csv.writer(file)
    write.writerows(data)
dfa47= pd.read_csv("Songshan.csv")
dfa47



# In[ ]:




