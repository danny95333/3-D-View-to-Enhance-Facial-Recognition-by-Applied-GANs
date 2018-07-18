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



    """Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """
def ToTensor(pic):

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
    def __init__(self,dataPath='/home/shumao/dr-gan/Data_new_realigned2/setting2/train/',loadSize=100,fineSize=96,flip=0,occlusion=0,masksize=0,supervision=1,pose_num=9,light_num=20,labelPath='/home/shumao/dr-gan/Data_new_realigned2/setting2/train/',multiview=0):
        super(multiPIE, self).__init__()
        # list all images into a list
        self.image_list = [x for x in listdir(dataPath) if is_image_file(x)]
        self.dataPath = dataPath
        self.loadSize = loadSize
        self.fineSize = fineSize
        self.flip = flip
        self.occlusion = occlusion
        self.masksize = masksize

        self.poses = {80:0,130:1,140:2,51:3,50:4,41:5,190:6,90:7,200:8}
        self.poses_inv= {0:'080',1:'130',2:'140',3:'051',4:'050',5:'041',6:'190',7:'090',8:'200'}
        self.pose_num = pose_num
        self.light_num = light_num
        self.labelPath = labelPath
        self.multiview = multiview

    def __getitem__(self, index):
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        dataName = os.path.join(self.dataPath,self.image_list[index])
        lise_len = len(self.image_list) #111780 for train/9 for test

        if self.multiview == 1:
            pose_index = np.random.randint(0,self.pose_num)
        else:
            pose_index = 3


        light_index = np.random.randint(0,self.light_num)

        labelNamePart = self.image_list[index]
        labelNamePart = labelNamePart[:10]+self.poses_inv[pose_index]+labelNamePart[13:]
        labelNamePart = labelNamePart[:14]+str(light_index).zfill(2)+labelNamePart[16:]
        labelName = os.path.join(self.labelPath,labelNamePart)
#---------get data's pose and light index-------------
        dataPose = self.image_list[index]
        dataPose = dataPose[10:13]
        dataPose_index = self.poses[int(dataPose)] # poses[51]->3
        #get 51 from '051',138_01_01_051_07.png
        dataLight = self.image_list[index]
        dataLight = dataLight[14:16]
        dataLight_index = int(dataLight)
        #get identity of image_1
        dataIden = self.image_list[index]
        dataIden = dataIden[0:3]
        dataIden_index = int(dataIden)




        threshold = random.random()
        # threshold = 0~1.0
        if threshold > 0.5:
            #if bigger than 5.0 by_image is same light different pose
            rdm_pose = random.random()
            if rdm_pose > 0.5:
                #by_image pose is positive
                if dataPose_index == 8:
                    by_poseName_index = 7
                else:
                    by_poseName_index = dataPose_index + 1
                by_lightName = dataLight_index
                by_IdenName = dataIden_index

            else:
                #by_image pose is negative
                if dataPose_index == 0:
                    by_poseName_index = 1
                else:
                    by_poseName_index = dataPose_index - 1
                by_lightName = dataLight_index
                by_IdenName = dataIden_index

            by_labelName_Part = self.image_list[index]
            by_labelName_Part = str(by_IdenName).zfill(3)+by_labelName_Part[3:]
            by_labelName_Part = by_labelName_Part[:10]+self.poses_inv[by_poseName_index]+by_labelName_Part[13:]
            by_labelName_Part = by_labelName_Part[:14]+str(by_lightName).zfill(2)+by_labelName_Part[16:]
            by_labelName = os.path.join(self.labelPath,by_labelName_Part)
        else:
            # totally random pose and light
            rdm_index = np.random.randint(0,lise_len) # image number in /train/
            by_labelName_Part = self.image_list[rdm_index]

            # IdenName = self.image_list[rdm_index]
            IdenName = by_labelName_Part[0:3]
            by_IdenName = int(IdenName)
            by_lightName_index = by_labelName_Part[14:16]
            by_lightName = int(by_lightName_index)
            by_poseName_index = by_labelName_Part[10:13]
            by_poseName_index = self.poses[int(by_poseName_index)] # poses[51]->3

            by_labelName = os.path.join(self.labelPath, by_labelName_Part)





        #for contrastive loss-- same_pose,same_light,same_iden
        if by_poseName_index == dataPose_index:
            same_pose = 1
        else:
            same_pose = 0
        if by_lightName == dataLight_index:
            same_light = 1
        else:
            same_light = 0
        if by_IdenName == dataIden_index:
            same_iden = 1
        else:
            same_iden = 0
        # print("------------------")
        # print(threshold)
        # print(same_pose)
        # print(same_light)
        # print(same_iden)
        # no problem

#-----------------------------------------------------


        #nums = self.image_list[index].split('_')
        #pose = self.poses[int(nums[3])]
        #identity = int(nums[0])
        identity, pose_angle,light = util.ParseImgName(self.image_list[index])
        

        # identity = identity - 1
        pose = self.poses[pose_angle]
        img = default_loader(dataName)
        w,h = img.size
        img_label = default_loader(labelName)
        img_by = default_loader(by_labelName)

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
        img_input = img
        img_label = img_label.resize((self.fineSize, self.fineSize), Image.BILINEAR)
        img_by = img_by.resize((self.fineSize, self.fineSize), Image.BILINEAR)

        # if(self.occlusion == 1):
        #     margin_x = self.fineSize - self.masksize;
        #     margin_y = self.fineSize - self.masksize;

        #     rand_x = random.randint(0,margin_x); 
        #     rand_y = random.randint(0,margin_y); 

        #     array2 = np.asarray(img)
        #     array2 = array2/255.0
        #     noise = np.random.rand(self.masksize, self.masksize, 3)
        #     array2[rand_x:rand_x+self.masksize, rand_y:rand_y+self.masksize,:] = noise;
        #     array2 = array2 * 255.0
        #     array2 = np.uint8(array2)
        #     img_input = Image.fromarray(array2)

        # if(self.flip == 1):
        #     if random.random() < 0.5:
        #         img_input = img_input.transpose(Image.FLIP_LEFT_RIGHT)
        #         pose = len(self.poses) - 1 - pose


        img = ToTensor(img) # 3 x 256 x 256
        img_input = ToTensor(img_input)
        img_label = ToTensor(img_label)
        img_by = ToTensor(img_by)
        img = img.mul_(2).add_(-1)
        img_input = img_input.mul_(2).add_(-1)
        img_label = img_label.mul_(2).add_(-1)
        img_by = img_by.mul_(2).add_(-1)
        # 3. Return a data pair (e.g. image, label, targeted pose and light label).
        # return img, identity, pose, light, img_label, img_by, same_pose, same_light, same_iden
        return img, pose, light, img_label, img_by, same_pose, same_light, same_iden

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


        if(self.flip == 1):
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                pose = len(self.poses) - 1 - pose


        img = ToTensor(img) # 3 x 256 x 256

        img = img.mul_(2).add_(-1)
        # 3. Return a data pair (e.g. image and label).
        return img, identity, pose, light

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.image_list)