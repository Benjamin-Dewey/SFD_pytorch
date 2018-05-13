
# coding: utf-8

# In[1]:


from __future__ import print_function

import torch
import torch.optim as optim

from torch.utils.data.dataset import Dataset
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.autograd import Variable
torch.backends.cudnn.bencmark = True
import torchvision
import torchvision.transforms as transforms

import os,sys,cv2,random,datetime,time,math
import argparse
import numpy as np

# from net_s3fd import *
from s3fd import *
from bbox import *
from sklearn.preprocessing import MultiLabelBinarizer
from PIL import Image
from copy import deepcopy

# In[2]:



class CelebDataset(Dataset):
    """Dataset wrapping images and target labels
    Arguments:
        A CSV file path
        Path to image folder
        Extension of images
        PIL transforms
    """

    def __init__(self, csv_path, img_path, img_ext, transform=None):

        tmp_df = pd.read_csv(csv_path)
        assert tmp_df['Image_Name'].apply(lambda x: os.path.isfile(img_path + x + img_ext)).all(), "Some images referenced in the CSV file were not found"

        self.mlb = MultiLabelBinarizer()
        self.img_path = img_path
        self.img_ext = img_ext
        self.transform = transform

        self.X_train = tmp_df['Image_Name']
        self.y_train = self.mlb.fit_transform(tmp_df['Gender'].str.split()).astype(np.float32)

    def __getitem__(self, index):
        img = cv2.imread(self.img_path + self.X_train[index] + self.img_ext)
        img = cv2.resize(img, (256,256))
        img = img - np.array([104,117,123])
        img = img.transpose(2, 0, 1)

        #img = img.reshape((1,)+img.shape)
        img = torch.from_numpy(img).float()
        #img = Variable(torch.from_numpy(img).float(),volatile=True)

        #if self.transform is not None:
        #    img = self.transform(img)

        label = torch.from_numpy(self.y_train[index])
        return img, label

    def __len__(self):
        return len(self.X_train.index)


# In[3]:


transformations = transforms.Compose(
    [
     transforms.ToTensor()

     #transforms.Normalize(mean=[104,117,123])
     ])


# In[4]:


train_data = "index.csv"
img_path = "data/Celeb_Small_Dataset/"
img_ext = ".jpg"
dset = CelebDataset(train_data,img_path,img_ext,transformations)
train_loader = DataLoader(dset,
                          batch_size=1,
                          shuffle=True,
                          num_workers=1 # 1 for CUDA
                         # pin_memory=True # CUDA only
                         )


# In[5]:


def save(model, optimizer, loss, filename):
    save_dict = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.data[0]
        }
    torch.save(save_dict, filename)


# In[6]:


def train_model(model, criterion, optimizer, num_classes, num_epochs = 100):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        model.train()
        running_loss = 0.0

        for i,(img,label) in enumerate(train_loader):
            img = img.view((1,)+img.shape[1:])
            if use_cuda:
                data, target = Variable(img.cuda()), Variable(torch.Tensor(label).cuda())
            else:
                data, target = Variable(img), Variable(torch.Tensor(label))
            target = target.view(1, num_classes)

            optimizer.zero_grad()
            olist = model(data)
            genList = []
            for j in range(len(olist)): olist[j] = F.softmax(olist[j])
            for j in range(len(olist)//2):
                ocls,ogen = olist[j*2].data.cpu(),olist[j*2+1]
                FB,FC,FH,FW = ocls.size() # feature map size
                stride = 2**(j+2)    # 4,8,16,32,64,128
                anchor = stride*4
                for Findex in range(FH*FW):
                    windex,hindex = Findex%FW,Findex//FW
                    axc,ayc = stride/2+windex*stride,stride/2+hindex*stride
                    score = ocls[0,1,hindex,windex]
                    if score<0.05: continue
                    genScore = ogen[0,:,hindex,windex].contiguous().view(1,2)
                    genList.append(genScore)

            losses = []
            for gen in genList:
                loss = criterion(gen, target)
                losses.append(loss)

            if i%50==0: print("Reached iteration ",i)

            loss = sum(losses)
            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]
        if epoch % 10 == 0:
            save(model, optimizer, loss, 'faceRecog.saved.model')
        print(running_loss)


# In[7]:


num_classes = 2
myModel = s3fd_original()


loadedModel = torch.load('s3fd_convert.pth')
newModel = myModel.state_dict()
pretrained_dict = {k: v for k, v in loadedModel.items() if k in newModel}
newModel.update(pretrained_dict)
myModel.load_state_dict(newModel)


# In[8]:


use_cuda = True
myModel.eval()


# In[ ]:


criterion = nn.BCELoss()

for param in myModel.parameters(): param.requires_grad = False

copyModel = deepcopy(myModel)

myModel.conv4_3_norm_gender    = nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1) # gender layer
a = torch.Tensor(copyModel.conv4_3_norm_mbox_conf.weight[0])
a = torch.cat((a, a), 0)
myModel.conv4_3_norm_gender.weight = a

myModel.conv5_3_norm_gender    = nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1) # gender layer
myModel.conv5_3_norm_gender.weight[0] = copyModel.conv5_3_norm_mbox_conf.weight[0]
myModel.conv5_3_norm_gender.weight[1] = copyModel.conv5_3_norm_mbox_conf.weight[0]

myModel.fc7_gender        = nn.Conv2d(1024, 2, kernel_size=3, stride=1, padding=1) # gender layer
myModel.fc7_gender.weight[0] = copyModel.fc7_mbox_conf.weight[0]
myModel.fc7_gender.weight[1] = copyModel.fc7_mbox_conf.weight[0]

myModel.conv6_2_gender    = nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1) # gender layer
myModel.conv6_2_gender.weight[0] = copyModel.conv6_2_mbox_conf.weight[0]
myModel.conv6_2_gender.weight[1] = copyModel.conv6_2_mbox_conf.weight[0]

optimizer = optim.SGD(filter(lambda p: p.requires_grad,myModel.parameters()), lr=0.0001, momentum=0.9)

if use_cuda: myModel = myModel.cuda()

model_ft = train_model(myModel, criterion, optimizer, num_classes, num_epochs=100)


# In[ ]:


def transform(img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (256,256))
        img = img - np.array([104,117,123])
        img = img.transpose(2, 0, 1)

        img = img.reshape((1,)+img.shape)
        img = torch.from_numpy(img).float()

        return Variable(img.cuda())

myModel = myModel.cuda()

testImage1 = transform('data/Test/TestCeleb_4/25-FaceId-0.jpg')
testImage2 = transform('data/Test/TestCeleb_4/26-FaceId-0.jpg')
testImage3 = transform('data/Test/TestCeleb_4/27-FaceId-0.jpg')
testImage4 = transform('data/Test/TestCeleb_10/25-FaceId-0.jpg')
testImage5 = transform('data/Test/TestCeleb_10/26-FaceId-0.jpg')
testImage6 = transform('data/Test/TestCeleb_10/24-FaceId-0.jpg')

output1 = myModel(testImage1)
output2 = myModel(testImage2)
output3 = myModel(testImage2)
output4 = myModel(testImage4)
output5 = myModel(testImage5)
output6 = myModel(testImage6)
print("testImage1 - ",output1)
print("testImage2 - ",output2)
print("testImage3 - ",output3)
print("testImage1 - ",output4)
print("testImage2 - ",output5)
print("testImage3 - ",output6)
