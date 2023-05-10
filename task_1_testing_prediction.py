# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 09:12:48 2023

@author: Abdul Qayyum
"""
#%% trained model
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.init as init
class Channel_Attention(nn.Module):

    def __init__(self,ratio=16,in_channel=128):

        super(Channel_Attention, self).__init__()
        self.ratio = ratio
        self.activate = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                      nn.Conv2d(in_channel,in_channel//ratio,kernel_size = 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(in_channel // ratio,in_channel,kernel_size = 1),
                                      nn.Sigmoid())


    def forward(self,x):
        actition = self.activate(x)
        out = torch.mul(x,actition)

        return out

class Spatial_Attention(nn.Module):
    def __init__(self, in_channel):
        super(Spatial_Attention, self).__init__()
        self.activate = nn.Sequential(nn.Conv2d(in_channel, 1,kernel_size = 1),
                                      )

    def forward(self, x):
        actition = self.activate(x)
        out = torch.mul(x, actition)

        return out

# torch.Size([1, 128, 256, 160])
# model=Spatial_Attention(128)
# inp=torch.rand(1,128,256,160)
# out=model(inp)
# print(out.shape)

def init_conv(conv):
    init.xavier_uniform_(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()

class Self_Attention(nn.Module):
    def __init__(self, in_channel):
        super(Self_Attention, self).__init__()
        self.chanel_in = in_channel

        self.f = nn.Conv2d(in_channels=in_channel, out_channels=in_channel // 8, kernel_size=1)
        self.g = nn.Conv2d(in_channels=in_channel, out_channels=in_channel // 8, kernel_size=1)
        self.h = nn.Conv2d(in_channels=in_channel, out_channels=in_channel//8, kernel_size=1)
        self.v = nn.Conv2d(in_channels=in_channel//8, out_channels=in_channel, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

        init_conv(self.f)
        init_conv(self.g)
        init_conv(self.h)

    def forward(self, x):
        m_batchsize, C, width, height = x.size()

        f = self.f(x).view(m_batchsize, -1, width * height)  # B * (C//8) * (W * H)
        g = self.g(x).view(m_batchsize, -1, width * height)  # B * (C//8) * (W * H)
        h = self.h(x).view(m_batchsize, -1, width * height)  # B * (C//8) * (W * H)

        attention = torch.bmm(f.permute(0, 2, 1), g)  # B * (W * H) * (W * H)
        attention = self.softmax(attention)

        self_attetion = torch.bmm(h, attention)  # B * (C//8) * (W * H)
        self_attetion = self_attetion.view(m_batchsize, -1, width, height)  # B * (C//8) * W * H

        self_attetion = self.v(self_attetion)   # B * C * W * H

        out = self.gamma * self_attetion   #############  +x

        return out

# model=Self_Attention(128)
# inp=torch.rand(1,128,256,160)
# out=model(inp)
# print(out.shape)
class hybrid_attention(nn.Module):
    def __init__(self,in_channel):
        super(hybrid_attention,self).__init__()
        self.spatial_att=Spatial_Attention(in_channel)
        self.selfattention=Self_Attention(in_channel)
        
    def forward(self,x):
        x=self.spatial_att(x)
        x=self.selfattention(x)
        
        return x
        
# modle=hybrid_attention(128)

# #model=Self_Attention(128)
# inp=torch.rand(1,128,256,160)
# out=modle(inp)
# print(out.shape)

###################################################### design unet in pytorch ####################################
# Sample Model Used
# In this tutorial, we'll use a simple 2D U-net to combine the transients, 
# considering the transients as a 2D "image" (spectral_points x transients). ### 2048x60 2D shape

# We'll declare our model as a class
class UNET(nn.Module):

    # initializing the weights for the convolution layers
    def __init__(self,transient_count):
        super(UNET,self).__init__()
        ############ 1-16
        self.down_conv_1_1 = nn.Conv2d(1,16,kernel_size=(5,1),padding="same")
        self.down_conv_1_2 = nn.Conv2d(16,16,kernel_size=(3,3),padding="same")
        #####################16-32
        self.down_conv_2_1 = nn.Conv2d(16,32,kernel_size=(3,3),padding="same")
        self.down_conv_2_2 = nn.Conv2d(32,32,kernel_size=(3,3),padding="same")
        #####################32-64
        self.down_conv_3_1 = nn.Conv2d(32,64,kernel_size=(3,3),padding="same")
        self.down_conv_3_2 = nn.Conv2d(64,64,kernel_size=(3,3),padding="same")
        #################### 64-128
        self.up_conv_1_1 = nn.Conv2d(64,128,kernel_size=(3,3),padding="same")
        self.up_conv_1_2 = nn.Conv2d(128,128,kernel_size=(3,3),padding="same")
        ###########################128-64
        self.up_conv_2_1 = nn.Conv2d(192,64,kernel_size=(3,3),padding="same")
        self.up_conv_2_2 = nn.Conv2d(64,64,kernel_size=(3,3),padding="same")
        #########################64-32
        self.up_conv_3_1 = nn.Conv2d(96,32,kernel_size=(3,3),padding="same")
        self.up_conv_3_2 = nn.Conv2d(32,32,kernel_size=(3,3),padding="same")
        ############################ end layers
        self.end_conv_1_1 = nn.Conv2d(48,128,kernel_size=(1,transient_count))
        self.end_conv_1_2 = nn.Conv2d(128,1,kernel_size=(5,5),padding="same")
        self.attention=Spatial_Attention(128)
        self.attention_s=Self_Attention(128)
        self.channel_a=Channel_Attention(128)
        #self.hybrid_attention=hybrid_attention(128)
    
    # defining forward pass
    def forward(self,x):
        #print(x.shape)  ### 1x2048x40x1
        
        # changing order of dimensions, as in torch the filters come first
        y = x.transpose(1,3)  
        #print(y.shape)    ### 1x1x40x2048
        y = y.transpose(2,3)
       # print(y.shape)  ### 1x1x2048x40
        
        
        ##### first downsample block convert 1-16,2048x40
        y = F.relu(self.down_conv_1_1(y))
        #print('first_downsample:',y.shape)     ### 1x16x2048x40
        y_skip1 = F.relu(self.down_conv_1_2(y))
        ##### first downsample block convert 16-32,2048x40
        #y_skip1 = F.relu(self.down_conv_1_2(y))
        
        ##### first downsample block convert 16-32,1024x40
        y = F.max_pool2d(y_skip1,(2,1)) #### 2048-1024
        y = F.relu(self.down_conv_2_1(y))
        #print('second_downsample:',y.shape)  ### 1x32x2048x40
        y_skip2 = F.relu(self.down_conv_2_2(y))

        y = F.max_pool2d(y_skip2,(2,1))  #### 1024-512
        ##### first downsample block convert 64,512x40
        y = F.relu(self.down_conv_3_1(y))
        #print('third_downsample:',y.shape)    ### 1x64x512x40
        y_skip3 = F.relu(self.down_conv_3_2(y))

        y = F.max_pool2d(y_skip3,(2,1))   #### 512-256

        y = F.relu(self.up_conv_1_1(y))
        y = F.relu(self.up_conv_1_2(y))
        #print('fourth_downsample:',y.shape)  ### 1x128x256x40
         ######################## 1x128x256x40#########
        #y=self.attention(y)
        #y=self.attention_s(y)
        y=self.channel_a(y)
        
        # print('attention_downsample:',y.shape)
        
        #y=self.hybrid_attention(y)
        
        #print('hybrid_downsample:',y.shape)
        
        
        
        y = F.upsample(y,scale_factor=(2,1)) ### 64*512

        y = torch.cat([y,y_skip3],axis=1)

        y = F.relu(self.up_conv_2_1(y))
        y = F.relu(self.up_conv_2_2(y))
        #print('first_upsample:',y.shape)  ### 1x64x512x40

        y = F.upsample(y,scale_factor=(2,1)) #### 32*1024

        y = torch.cat([y,y_skip2],axis=1)

        y = F.relu(self.up_conv_3_1(y))
        y = F.relu(self.up_conv_3_2(y))
        #print('second_upsample:',y.shape)   ### 1x32x1024x40

        y = F.upsample(y,scale_factor=(2,1)) ### 1*2048

        y = torch.cat([y,y_skip1],axis=1)

        y = F.relu(self.end_conv_1_1(y))
        y = self.end_conv_1_2(y)
        #print('third_upsample:',y.shape) ### 1x1x2048x1

        # converting the order of layers back to the original format

        y = y.transpose(1,3)
        y = y.transpose(1,2)
        #print(y.shape)   #1*2048x1*1
        #print(y.view(y.shape[0],-1).shape)

        # flattening result to only have 2 dimensions
        return y.view(y.shape[0],-1) #1x2048
##### we can chnage 40,60,160 etc   
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNET(40).float()
model.load_state_dict(torch.load("torch_weights_task1_base_CHA.pth"))
model.eval()
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model.to(device)
#%latest model
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.init as init
class Channel_Attention(nn.Module):

    def __init__(self,ratio=16,in_channel=128):

        super(Channel_Attention, self).__init__()
        self.ratio = ratio
        self.activate = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                      nn.Conv2d(in_channel,in_channel//ratio,kernel_size = 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(in_channel // ratio,in_channel,kernel_size = 1),
                                      nn.Sigmoid())


    def forward(self,x):
        actition = self.activate(x)
        out = torch.mul(x,actition)

        return out

class Spatial_Attention(nn.Module):
    def __init__(self, in_channel):
        super(Spatial_Attention, self).__init__()
        self.activate = nn.Sequential(nn.Conv2d(in_channel, 1,kernel_size = 1),
                                      )

    def forward(self, x):
        actition = self.activate(x)
        out = torch.mul(x, actition)

        return out

# torch.Size([1, 128, 256, 160])
# model=Spatial_Attention(128)
# inp=torch.rand(1,128,256,160)
# out=model(inp)
# print(out.shape)

def init_conv(conv):
    init.xavier_uniform_(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()

class Self_Attention(nn.Module):
    def __init__(self, in_channel):
        super(Self_Attention, self).__init__()
        self.chanel_in = in_channel

        self.f = nn.Conv2d(in_channels=in_channel, out_channels=in_channel // 8, kernel_size=1)
        self.g = nn.Conv2d(in_channels=in_channel, out_channels=in_channel // 8, kernel_size=1)
        self.h = nn.Conv2d(in_channels=in_channel, out_channels=in_channel//8, kernel_size=1)
        self.v = nn.Conv2d(in_channels=in_channel//8, out_channels=in_channel, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

        init_conv(self.f)
        init_conv(self.g)
        init_conv(self.h)

    def forward(self, x):
        m_batchsize, C, width, height = x.size()

        f = self.f(x).view(m_batchsize, -1, width * height)  # B * (C//8) * (W * H)
        g = self.g(x).view(m_batchsize, -1, width * height)  # B * (C//8) * (W * H)
        h = self.h(x).view(m_batchsize, -1, width * height)  # B * (C//8) * (W * H)

        attention = torch.bmm(f.permute(0, 2, 1), g)  # B * (W * H) * (W * H)
        attention = self.softmax(attention)

        self_attetion = torch.bmm(h, attention)  # B * (C//8) * (W * H)
        self_attetion = self_attetion.view(m_batchsize, -1, width, height)  # B * (C//8) * W * H

        self_attetion = self.v(self_attetion)   # B * C * W * H

        out = self.gamma * self_attetion   #############  +x

        return out

# model=Self_Attention(128)
# inp=torch.rand(1,128,256,160)
# out=model(inp)
# print(out.shape)
class hybrid_attention(nn.Module):
    def __init__(self,in_channel):
        super(hybrid_attention,self).__init__()
        self.spatial_att=Spatial_Attention(in_channel)
        self.selfattention=Self_Attention(in_channel)
        
    def forward(self,x):
        x=self.spatial_att(x)
        x=self.selfattention(x)
        
        return x
        
# modle=hybrid_attention(128)

# #model=Self_Attention(128)
# inp=torch.rand(1,128,256,160)
# out=modle(inp)
# print(out.shape)

###################################################### design unet in pytorch ####################################
# Sample Model Used
# In this tutorial, we'll use a simple 2D U-net to combine the transients, 
# considering the transients as a 2D "image" (spectral_points x transients). ### 2048x60 2D shape

# We'll declare our model as a class
class UNET(nn.Module):

    # initializing the weights for the convolution layers
    def __init__(self,transient_count):
        super(UNET,self).__init__()
        ############ 1-16
        self.down_conv_1_1 = nn.Conv2d(1,16,kernel_size=(5,1),padding="same")
        self.down_conv_1_2 = nn.Conv2d(16,16,kernel_size=(3,3),padding="same")
        #####################16-32
        self.down_conv_2_1 = nn.Conv2d(16,32,kernel_size=(3,3),padding="same")
        self.down_conv_2_2 = nn.Conv2d(32,32,kernel_size=(3,3),padding="same")
        #####################32-64
        self.down_conv_3_1 = nn.Conv2d(32,64,kernel_size=(3,3),padding="same")
        self.down_conv_3_2 = nn.Conv2d(64,64,kernel_size=(3,3),padding="same")
        #################### 64-128
        self.up_conv_1_1 = nn.Conv2d(64,128,kernel_size=(3,3),padding="same")
        self.up_conv_1_2 = nn.Conv2d(128,128,kernel_size=(3,3),padding="same")
        ###########################128-64
        self.up_conv_2_1 = nn.Conv2d(192,64,kernel_size=(3,3),padding="same")
        self.up_conv_2_2 = nn.Conv2d(64,64,kernel_size=(3,3),padding="same")
        #########################64-32
        self.up_conv_3_1 = nn.Conv2d(96,32,kernel_size=(3,3),padding="same")
        self.up_conv_3_2 = nn.Conv2d(32,32,kernel_size=(3,3),padding="same")
        ############################ end layers
        self.end_conv_1_1 = nn.Conv2d(48,128,kernel_size=(1,transient_count))
        self.end_conv_1_2 = nn.Conv2d(128,1,kernel_size=(5,5),padding="same")
        self.attention=Spatial_Attention(128)
        self.attention_s=Self_Attention(128)
        self.channel_a=Channel_Attention(128)
        #self.hybrid_attention=hybrid_attention(128)
    
    # defining forward pass
    def forward(self,x):
        #print(x.shape)  ### 1x2048x40x1
        
        # changing order of dimensions, as in torch the filters come first
        y = x.transpose(1,3)  
        #print(y.shape)    ### 1x1x40x2048
        y = y.transpose(2,3)
       # print(y.shape)  ### 1x1x2048x40
        
        
        ##### first downsample block convert 1-16,2048x40
        y = F.relu(self.down_conv_1_1(y))
        #print('first_downsample:',y.shape)     ### 1x16x2048x40
        y_skip1 = F.relu(self.down_conv_1_2(y))
        ##### first downsample block convert 16-32,2048x40
        #y_skip1 = F.relu(self.down_conv_1_2(y))
        
        ##### first downsample block convert 16-32,1024x40
        y = F.max_pool2d(y_skip1,(2,1)) #### 2048-1024
        y = F.relu(self.down_conv_2_1(y))
        #print('second_downsample:',y.shape)  ### 1x32x2048x40
        y_skip2 = F.relu(self.down_conv_2_2(y))

        y = F.max_pool2d(y_skip2,(2,1))  #### 1024-512
        ##### first downsample block convert 64,512x40
        y = F.relu(self.down_conv_3_1(y))
        #print('third_downsample:',y.shape)    ### 1x64x512x40
        y_skip3 = F.relu(self.down_conv_3_2(y))

        y = F.max_pool2d(y_skip3,(2,1))   #### 512-256

        y = F.relu(self.up_conv_1_1(y))
        y = F.relu(self.up_conv_1_2(y))
        #print('fourth_downsample:',y.shape)  ### 1x128x256x40
         ######################## 1x128x256x40#########
        #y=self.attention(y)
        #y=self.attention_s(y)
        y=self.channel_a(y)
        
        # print('attention_downsample:',y.shape)
        
        #y=self.hybrid_attention(y)
        
        #print('hybrid_downsample:',y.shape)
        
        
        
        y = F.upsample(y,scale_factor=(2,1)) ### 64*512

        y = torch.cat([y,y_skip3],axis=1)

        y = F.relu(self.up_conv_2_1(y))
        y = F.relu(self.up_conv_2_2(y))
        #print('first_upsample:',y.shape)  ### 1x64x512x40

        y = F.upsample(y,scale_factor=(2,1)) #### 32*1024

        y = torch.cat([y,y_skip2],axis=1)

        y = F.relu(self.up_conv_3_1(y))
        y = F.relu(self.up_conv_3_2(y))
        #print('second_upsample:',y.shape)   ### 1x32x1024x40

        y = F.upsample(y,scale_factor=(2,1)) ### 1*2048

        y = torch.cat([y,y_skip1],axis=1)

        y = F.relu(self.end_conv_1_1(y))
        y = self.end_conv_1_2(y)
        #print('third_upsample:',y.shape) ### 1x1x2048x1

        # converting the order of layers back to the original format

        y = y.transpose(1,3)
        y = y.transpose(1,2)
        #print(y.shape)   #1*2048x1*1
        #print(y.view(y.shape[0],-1).shape)

        # flattening result to only have 2 dimensions
        return y.view(y.shape[0],-1) #1x2048
##### we can chnage 40,60,160 etc   
model = UNET(40).float()
# test_inp=torch.rand(1,2048,40,1)
model.load_state_dict(torch.load("torch_weights_task1_base_combinee_new_finallastch.pth"))
model.eval()
#%% test dataset for task1
# Changing path to access scripts
import sys
sys.path.insert(0,"..")

import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F
import h5py
import math
import matplotlib.pyplot as plt
import numpy as np
# Verify cuda
torch.cuda.is_available()
###############################################################dataset ###########################

###################################################### Data Loading ################################################
# The first thing we'll need to do is loading the data, which is stored as ground truth 
# fids in a numpy file. 
# We'll then add noise to the fids, using the same code as in the noise_adding_tutorial,
#  generating our noisy transients this way.

# After obtaining the transients, 
# we'll normalize both the x and y data and divide them 
# into training and testing datasets.


############################################# load sample data #######################################
with h5py.File("D:/reconsrtcionMRI/task1/task1/track_01_test_data.h5") as hf:
    gt_fids = hf["transients"][()]
    ppm = hf["ppm"][()]
    t = hf["t"][()]
    
# # Transforming time domain noisy transients into frequency domain difference transients
noise_spec_a = np.fft.fftshift(np.fft.ifft(gt_fids,axis=1),axes=1)
noise_diff_spec_a = noise_spec_a[:,:,1,:]-noise_spec_a[:,:,0,:]

# initial definition of input and target data
x_a = np.real(noise_diff_spec_a)  #200,2048,40

def normalize_data(x):
    # normalizing using minimum of region of most interest and total maximum.
    x_max = x.max(axis=(1,2),keepdims=True)
    x_mean = x[:,900:1100].min(axis=(1,2),keepdims=True)
    
    x = (x-x_mean)/(x_max-x_mean)
    # expanding dimensions for compatibility with U-NET
    x = np.expand_dims(x,axis=3) #200,2048,160,1
    #print(x.shape)
    return x
x_test=normalize_data(x_a)

x_test = torch.from_numpy(x_test)
#ppm_train = torch.from_numpy(ppm)
ppm_test = torch.from_numpy(ppm)

x_test.float()

class BasicDataset(Dataset):
    def __init__(self,x1):
        super(BasicDataset,self).__init__()

        self.x1 = x1
       

    def __len__(self):
        return int(self.x1.shape[0])

    def __getitem__(self,idx):
        x_test=self.x1[idx]
        return x_test
# train_dataset=BasicDataset(x_test)
# train_dataloader = DataLoader(train_dataset,1,shuffle=False)
# for i,d in enumerate(train_dataloader):
#     print(i)
#     print(d)
#%%     ################# predictions
train_dataset=BasicDataset(x_test.float())
train_dataloader = DataLoader(train_dataset,1,shuffle=False,num_workers = 0)
predf=[]
for i,d in enumerate(train_dataloader):
    #print(i)
    #print
    pred = model.cpu()((d.float())).to("cpu")
    print(pred.shape)
    pred_n=np.squeeze(pred.detach().numpy(),axis=0)
    predf.append(pred_n)
    #del d
    #del pred
    #break
pred=np.array(predf)
#%% ####################### submission
import h5py
import numpy as np


def save_submission(result_spectra,ppm,filename):
    '''
    Save the results in the submission format
    Parameters:
        - results_spectra (np.array): Resulting predictions from test in format scan x spectral points
        - ppm (np.array): ppm values associataed with results, in same format
        - filename (str): name of the file to save results in, should end in .h5
    
    '''

    with h5py.File(filename,"w") as hf:
        hf.create_dataset("result_spectra",result_spectra.shape,dtype=float,data=result_spectra)
        hf.create_dataset("ppm",ppm.shape,dtype=float,data=ppm)

save_submission(pred,ppm_test.numpy(),"track01_simulated_final.H5")




