import sys
sys.path.append('../')
import torch
import numpy as np
import torch.utils.data as data
from os import listdir
from os.path import join
import os
from PIL import Image
import random
import math
from utils import util
import scipy.ndimage


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def default_loader(path):
    return Image.open(path).convert('RGB')

def ToTensor(pic):
    """Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """
    if isinstance(pic, np.ndarray):
        # handle numpy array
        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        # backard compability
        return img.float().div(255)
    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float().div(255)
    else:
        return img


# You should build custom dataset as below.
class multiPIE(data.Dataset):
    def __init__(self,dataPath='/home/shumao/dr-gan/Data_new_realigned2/setting2/train/',loadSize=100,fineSize=96,pose_num=9,light_num=20,labelPath='/home/shumao/dr-gan/Data_new_realigned2/setting2/Facedata/',multiview=0):
        super(multiPIE, self).__init__()
        # list all images into a list
        self.image_list = [x for x in listdir(dataPath) if is_image_file(x)]
        self.dataPath = dataPath
        self.loadSize = loadSize
        self.fineSize = fineSize
        self.poses = {80:0,130:1,140:2,51:3,50:4,41:5,190:6,90:7,200:8}
        self.poses_inv= {0:'080',1:'130',2:'140',3:'051',4:'050',5:'041',6:'190',7:'090',8:'200'}
        self.pose_num = pose_num
        self.light_num = light_num
        self.labelPath = labelPath
        self.multiview = multiview

    def __getitem__(self, index):
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        dataName = os.path.join(self.dataPath,self.image_list[index])

        if self.multiview == 1:
            pose_index = np.random.randint(0,self.pose_num)
        else:
            pose_index = 3
        light_index = np.random.randint(0,self.light_num)

        labelNamePart = self.image_list[index]
        labelNamePart = labelNamePart[:10]+self.poses_inv[pose_index]+labelNamePart[13:]
        labelNamePart = labelNamePart[:14]+str(light_index).zfill(2)+labelNamePart[16:]
        labelName = os.path.join(self.labelPath,labelNamePart)
        identity, pose_angle,light = util.ParseImgName(self.image_list[index])
        

        identity = identity - 1
        pose = self.poses[pose_angle]
        img = default_loader(dataName)
        w,h = img.size
        img_label = default_loader(labelName)

        if(h != self.loadSize):
            img = img.resize((self.loadSize, self.loadSize), Image.BILINEAR)

        if(self.loadSize != self.fineSize):
            p = random.random()
            if p < 0.5:
                # random crop
                x1 = random.randint(0, self.loadSize - self.fineSize)
                y1 = random.randint(0, self.loadSize - self.fineSize)
                img = img.crop((x1, y1, x1 + self.fineSize, y1 + self.fineSize))
            else:
                # random shift
                x1 = random.randint(0, self.loadSize - self.fineSize)
                y1 = random.randint(0, self.loadSize - self.fineSize)
                array = np.asarray(img)
                array = scipy.ndimage.interpolation.shift(img,(x1,y1,0),mode='nearest')
                img = Image.fromarray(array)

                img = img.resize((self.fineSize, self.fineSize), Image.BILINEAR)
        img_input = img
        img_label = img_label.resize((self.fineSize, self.fineSize), Image.BILINEAR)


        img = ToTensor(img) # 3 x 256 x 256
        img_input = ToTensor(img_input)
        img_label = ToTensor(img_label)
        img = img.mul_(2).add_(-1)
        img_input = img_input.mul_(2).add_(-1)
        img_label = img_label.mul_(2).add_(-1)
        return img, img_input, identity, pose, light, img_label, pose_index, light_index

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.image_list)

# You should build custom dataset as below.
class multiPIE_test(data.Dataset):
    def __init__(self,dataPath='data/crop/',loadSize=100,fineSize=96,flip=0):
        super(multiPIE_test, self).__init__()
        # list all images into a list
        self.image_list = [x for x in listdir(dataPath) if is_image_file(x)]
        self.dataPath = dataPath
        self.loadSize = loadSize
        self.fineSize = fineSize
        self.flip = flip
        self.poses = {80:0,130:1,140:2,51:3,50:4,41:5,190:6,90:7,200:8}

    def __getitem__(self, index):
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        path = os.path.join(self.dataPath,self.image_list[index])
        #nums = self.image_list[index].split('_')
        #pose = self.poses[int(nums[3])]
        #identity = int(nums[0])
        identity, pose_angle,light = util.ParseImgName(self.image_list[index])
        identity = identity - 1
        pose = self.poses[pose_angle]
        img = default_loader(path)
        w,h = img.size

        if(h != self.loadSize):
            img = img.resize((self.loadSize, self.loadSize), Image.BILINEAR)

        if(self.loadSize != self.fineSize):
            p = random.random()
            if p < 0.5:
                # random crop
                x1 = random.randint(0, self.loadSize - self.fineSize)
                y1 = random.randint(0, self.loadSize - self.fineSize)
                #x1 = math.floor((self.loadSize - self.fineSize)/2)
                #y1 = math.floor((self.loadSize - self.fineSize)/2)
                img = img.crop((x1, y1, x1 + self.fineSize, y1 + self.fineSize))
            else:
                # random shift
                x1 = random.randint(0, self.loadSize - self.fineSize)
                y1 = random.randint(0, self.loadSize - self.fineSize)
                #x1 = math.floor((self.loadSize - self.fineSize)/2)
                #y1 = math.floor((self.loadSize - self.fineSize)/2)
                array = np.asarray(img)
                array = scipy.ndimage.interpolation.shift(img,(x1,y1,0),mode='nearest')
                img = Image.fromarray(array)
                img = img.resize((self.fineSize, self.fineSize), Image.BILINEAR)
            #else:   #random rotate
            #    angle = random.uniform(-5,5)    
            #    array = np.asarray(img)
            #    array = scipy.ndimage.interpolation.rotate(img,angle)
            #    img = Image.fromarray(array)
            #    img = img.resize((self.fineSize, self.fineSize), Image.BILINEAR)

        img = ToTensor(img) # 3 x 256 x 256

        img = img.mul_(2).add_(-1)
        # 3. Return a data pair (e.g. image and label).
        return img, identity, pose, light

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.image_list)