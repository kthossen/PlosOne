#!/usr/bin/env python
# coding: utf-8

# In[3]:



import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2" 

# In[13]:



import torch
import torch.nn as nn
import torch.nn.functional as F
#from torchsummary import summary
from torch.optim import Adam
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from typing import List
import pandas as pd
import numpy as np
import math ,time

import math, time
from sklearn.metrics import mean_squared_error

# In[13]:


dfa4= pd.read_csv(r"../../../../Taipei_14.csv")
dfa5= pd.read_csv(r"../../../../Taipei_15.csv")
dfa6= pd.read_csv(r"../../../../Taipei_16.csv")
dfa7= pd.read_csv(r"../../../../Taipei_17.csv")
dfa8= pd.read_csv(r"../../../../Taipei_18.csv")


# In[4]:


# dfa4= pd.read_csv(r"C:\Users\Khalid\Downloads\Taipei_14.csv")
# dfa5= pd.read_csv(r"C:\Users\Khalid\Downloads\Taipei_15.csv")
# dfa6= pd.read_csv(r"C:\Users\Khalid\Downloads\Taipei_16.csv")
# dfa7= pd.read_csv(r"C:\Users\Khalid\Downloads\Taipei_17.csv")
# dfa8= pd.read_csv(r"C:\Users\Khalid\Downloads\Taipei_18.csv")


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

r14=(dfa4[dfa4.SiteEngName =='Cailiao'])
r15=(dfa5[dfa5.SiteEngName =='Cailiao'])
r16=(dfa6[dfa6.SiteEngName =='Cailiao'])
r17=(dfa7[dfa7.SiteEngName =='Cailiao'])
r18=(dfa8[dfa8.SiteEngName =='Cailiao'])

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
timestep =16
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


# In[ ]:





# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

class TimeSeriesModel(nn.Module):
    def __init__(self, input_size, cnn_out_channels, lstm_hidden_size, lstm_num_layers, num_heads, num_classes):
        super(TimeSeriesModel, self).__init__()

        # 1D CNN layer
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, cnn_out_channels, kernel_size=1)
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=8)
        )

        # Bidirectional LSTM layer
        self.lstm = nn.LSTM(cnn_out_channels, lstm_hidden_size, lstm_num_layers, batch_first=True, bidirectional=True)

        # Self-Attention layer
        #self.self_attention = nn.MultiheadAttention(embed_dim=lstm_hidden_size * 2, num_heads=num_heads)
        self.self_attention = nn.MultiheadAttention(embed_dim=lstm_hidden_size*2, num_heads=num_heads)

        # Fully connected layer
        self.fc = nn.Linear(lstm_hidden_size*2 , num_classes)  # Multiply by 2 for bidirectional

    def forward(self, x):
        # Assuming x has shape (batch_size, sequence_length, input_size)

        # Apply 1D CNN
        x = self.cnn(x.transpose(1, 2)).transpose(1, 2)  # Transpose for CNN operation
        # = self.cnn(x)  # Transpose for CNN operation
        # Apply Bidirectional LSTM
        lstm_out, _ = self.lstm(x)

        # Apply Self-Attention
        attention_out, _ = self.self_attention(lstm_out.transpose(0, 1), lstm_out.transpose(0, 1), lstm_out.transpose(0, 1))
       # attended_values = attention_out.transpose(0, 1)
        attended_values = attention_out.transpose(0,1)
        attended_values=torch.mean(attended_values,1)
        # Take the output from the last time step for classification
        #attended_values = attended_values[-1]
        # Fully connected layer
        out = self.fc(attended_values).unsqueeze(-1)
        return out

    
input_size=10

batch_size = 16
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# Instantiate the model, loss function, and optimizer


cnn_out_channels =8 # Adjust as needed

lstm_hidden_size = 10  # Adjust as needed
lstm_num_layers = 8  # Adjust as needed
num_heads = 4  # Number of heads in the self-attention layer
output_size =16 # Adjust based on your task (e.g., binary classification)



model = TimeSeriesModel(input_size, cnn_out_channels, lstm_hidden_size, lstm_num_layers, num_heads, output_size).to(device)
criterion = torch.nn.MSELoss(reduction='mean')
optimizer = optim.Adam(model.parameters(), lr=0.001)


# In[ ]:


print(model)


# In[ ]:



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
#num_epochs =50
##########################

for t in range(num_epochs):
    model.train()
    for idx, data in enumerate(train_dataloader):
        x, y_true = data
        y_pred = model(x).to(device)
      #  print(y_pred.shape,y_true.shape)

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
        torch.save(model.state_dict(),'Cailiao8.pt')
#     scheduler.step(vall_loss)
    #print("Epoch:, loss: %1.5f valid loss:  %1.5f "%(loss.item(),vall_loss.item()))
    print("Epoch ", t, "MSE: ", hist[t].item(),t,"Valid loss",val[t].item())

training_time = time.time()-start_time    
print("Training time: {}".format(training_time))


# In[ ]:




model.load_state_dict(torch.load('Cailiao8.pt'))
#####################


sum([param.nelement() for param in model.parameters()])


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

print('this is rmse ',rmse_score)
print('this is mape ',mape_score)
print('this is mae ',mae_score)

#####################################

import csv
data =[[16,rmse_score,mae_score,mape_score,"Cailiao"]]
file = open('Cailiao.csv', 'a+', newline ='')

# writing the data into the file
with file:    
    write = csv.writer(file)
    write.writerows(data)
dfa47= pd.read_csv("Cailiao.csv")
dfa47


# In[ ]:




