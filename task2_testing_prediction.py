# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 00:58:58 2023

@author: cse
"""

#%%
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
        self.hybrid_attention=hybrid_attention(128)
    
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
        #y=self.hybrid_attention(y)
        #y=self.attention_s(y)
        y=self.channel_a(y)
        
        # print('attention_downsample:',y.shape)
        
        #y=likeself.hybrid_attention(y)
        
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
# out=model(test_inp)

# test_inp=torch.rand(1,2048,40,1)
# out,out1=model(test_inp)
# print(out.shape)   
    
####################################################################### loss function class ########################
# We'll use a mean average loss applied to a specific range for a loss function
class RangeMAELoss(nn.Module):

    def __init__(self):
        super(RangeMAELoss,self).__init__()

    
    # for the forward pass, a 1d ppm array must be passed and it's assumed that
    # it's valid for all sets
    def forward(self,x,y,ppm):
        
        # defining indexes of boundaries
        min_ind = torch.argmax(ppm[ppm<=4])
        max_ind = torch.argmin(ppm[ppm>=2.5])

        # selecting part of arrays pertaining to region of interest
        loss_x = x[:,min_ind:max_ind]
        loss_y = y[:,min_ind:max_ind]

        #calculate absolute loss mean value
        loss = torch.abs(loss_x-loss_y).mean(dim=1).mean(axis=0)

        return loss
    
#lssbase=RangeMAELoss()
total_loss=0
import numpy as np
import cv2
import torch
from tqdm import tqdm
"""
Loss Functions
""" 
    
def dy_huber_loss(inputs, targets, beta):
    """
    Dynamic Huber loss function
    
    """
    n = torch.abs(inputs - targets)
    cond = n <= beta
    loss = torch.where(cond, 0.5 * n ** 2, beta*n - 0.5 * beta**2)

    return loss.mean()

def dy_smooth_l1_loss(inputs, targets, beta):
    """
    Dynamic ParamSmoothL1 loss function
    
    """
    n = torch.abs(inputs - targets)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2, n + 0.5 * beta**2 - beta)

    return loss.mean()

def dy_tukey_loss(input, target, c):
    """
    Dynamic Tukey loss function
    
    """    
        
    n = torch.abs(input - target)
    cond = n <= c
    loss = torch.where(cond, ((c** 2)/6) * (1- (1 - (n /c)**2) **3 )  , torch.tensor((c** 2)/6).to('cuda'))

    return loss.mean()


"""
Evaluation Calculations
"""
models_save_path='/content/drive/MyDrive/brats2020_survival/save_model'
def MAE_distance(preds, labels):
    return torch.sum(torch.abs(preds - labels))

def PC_mine(preds, labels):
    dem = np.sum((preds - np.mean(preds))*(labels - np.mean(labels)))
    mina = (np.sqrt(np.sum((preds - np.mean(preds))**2)))*(np.sqrt(np.sum((labels - np.mean(labels))**2)))
    return dem/mina 

criterion = dy_smooth_l1_loss  
sigma_max = 0.7
sigma_min = 0.3
train_MAE = []
train_RMSE = []
train_PC = []
test_MAE = []    
test_RMSE = []
test_PC = []
epoch_count = []
pc_best = -2
Nepochs=50
sigma_min=0.2
sigma_max=0.7
total_loss_val=0
c=2
#sigma = sigma_min + (1/2)* (sigma_max - sigma_min ) * (1+ np.cos (np.pi * ((epoch+1)/Nepochs)))
from torch.utils.data import Dataset,DataLoader
    
############################ Simple dataset used to iterate over data #########################
class BasicDataset(Dataset):
    def __init__(self,x,y,ppm):
        super(BasicDataset,self).__init__()

        self.x = x
        self.y = y
        self.ppm = ppm

    def __len__(self):
        return int(self.x.shape[0])

    def __getitem__(self,idx):

        return self.x[idx],self.y[idx],self.ppm[idx]
    

class BasicDataset_my(Dataset):
    def __init__(self,x_a,y_a,ppm_a,x_f,y_f,ppm_f,x_p,y_p,ppm_p):
        super(BasicDataset_my,self).__init__()

        self.x_a = x_a
        self.y_a = y_a
        self.ppm_a = ppm_a
        
        self.x_f = x_f
        self.y_f = y_f
        self.ppm_f = ppm_f
        
        self.x_p = x_p
        self.y_p = y_p
        self.ppm_p = ppm_p

    def __len__(self):
        return int(self.x_a.shape[0])

    def __getitem__(self,idx):
        
        x_a=self.x_a[idx]
        y_a=self.y_a[idx]
        ppm_a=self.ppm_a[idx]
        
        x_f=self.x_f[idx]
        y_f=self.y_f[idx]
        ppm_f=self.ppm_f[idx]
        
        x_p=self.x_p[idx]
        y_p=self.y_p[idx]
        ppm_p=self.ppm_p[idx]

        return x_a,y_a,ppm_a,x_f,y_f,ppm_f,x_p,y_p,ppm_p

# train_dataset_new = BasicDataset_my(x_train_a,y_train_a,ppm_train_f,x_train_f,y_train_f,ppm_train_f,
#                                 x_train_p,y_train_p,ppm_train_p)
# train_dataloader = DataLoader(train_dataset_new,10,shuffle=True)
# a,b,c,d,e,f,g,h,i=train_dataset_new[0]


# for x_a,y_a,x_ppm_a,x_f,y_f,x_ppm_f ,x_p,y_p,x_ppm_p in train_dataloader:
#     x_a=x_a
#     y_a=y_a
#     x_ppm_a=x_ppm_a
    
#     x_f=x_f
#     y_f=y_f
#     x_ppm_f=x_ppm_f
    
    
#     x_p=x_p
#     y_p=y_p
#     x_ppm_p=x_ppm_p
#     print(x_a)
#     pred = model(x.float())
#     break


################################################## Setting Up model ###########################################
#We'll instantiate the model using the same number of transients 
#we've been using for our data, 40 transients.

# instantiate model and make it use float values for avoiding incompatibilities
model = UNET(40).float()
model.load_state_dict(torch.load("torch_weights_task2_base_combinee_task2ch_finallastmore.pth"))
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
with h5py.File("D:/reconsrtcionMRI/task2/task2/track_02_test_data.h5") as hf:
    gt_fids = hf["transient_fids"][()]
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
    x_mean = x.min(axis=(1,2),keepdims=True)
    
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
#%%
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
pred_task2=np.array(predf)
#%%
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

save_submission(pred_task2,ppm_test.numpy(),"track02_invivo_homogeneous1.H5")