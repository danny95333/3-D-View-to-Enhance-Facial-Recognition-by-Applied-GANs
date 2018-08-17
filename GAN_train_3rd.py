from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchnet.meter as meter
import torchvision.utils as vutils
from torch.autograd import Variable
from GAN_dataset_3rd import multiPIE
from GAN_model_3rd import Discriminator
from GAN_model_3rd import Generator
import numpy
# from pycrayon import CrayonClient

#for plotting loss
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import time,math
from logger import Logger

logger = Logger('./log');

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--loadSize', type=int, default=100, help='the height / width of the input image to network')
parser.add_argument('--fineSize', type=int, default=96, help='the height / width of the input image to network')
parser.add_argument('--noise_dim', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=32)
parser.add_argument('--ndf', type=int, default=32)
parser.add_argument('--channel_num', type=int, default=3, help='input image channel')
parser.add_argument('--id_num', type=int, default=200, help='Total training identity.')
parser.add_argument('--pose_num', type=int, default=9, help='Total training pose.')
parser.add_argument('--light_num', type=int, default=20, help='Total training lightmination.')
parser.add_argument('--niter', type=int, default=50000, help='number of iterations to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--outf', default='test', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--dataPath', default='/home/shumao/dr-gan/Data_new_realigned2/setting2/train/', help='which dataset to train on')

parser.add_argument('--save_step', type=int, default=10000, help='save weights every 10000 iterations ')
parser.add_argument('--hidden_size', type=int, default=320, help='bottleneck dimension of Discriminator')
parser.add_argument('--labelPath', default='/home/shumao/dr-gan/Data_new_realigned2/setting2/Facedata/', help='which dataset to train on')
parser.add_argument('--multiview', type=int, default=1, help='generate multiview images')



opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

###############   DATASET   ##################
dataset = multiPIE(opt.dataPath,opt.loadSize,opt.fineSize,opt.pose_num,opt.light_num,opt.labelPath,opt.multiview)
dataset_test = multiPIE('/home/shumao/dr-gan/comparison/',opt.loadSize,opt.fineSize,opt.pose_num,opt.light_num,opt.labelPath,opt.multiview)
loader_ = torch.utils.data.DataLoader(dataset=dataset,
                                           batch_size=opt.batchSize,
                                           shuffle=True,
                                           num_workers=2)
loader_test_ = torch.utils.data.DataLoader(dataset=dataset_test,
                                           batch_size=9,
                                           shuffle=False,
                                           num_workers=2)
loader = iter(loader_)
loader_test = iter(loader_test_)

###############   MODEL   ####################
ndf = opt.ndf
ngf = opt.ngf
nc = opt.channel_num
nf = opt.hidden_size
nd = opt.id_num
np = opt.pose_num
ni = opt.light_num
nz = opt.noise_dim

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

netD = Discriminator(nc,ndf,nf,nd,np,ni)
netG = Generator(nc,nf,ngf,nz,np,ni)
netD.apply(weights_init)
netG.apply(weights_init)

if(opt.cuda):
    netD.cuda()
    netG.cuda()

###########   LOSS & OPTIMIZER   ##########
optimizerD = torch.optim.Adam(netD.parameters(),lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(),lr=opt.lr, betas=(opt.beta1, 0.999))

GANCriterion = nn.BCELoss()
idCriterion = nn.CrossEntropyLoss()
poseCriterion = nn.CrossEntropyLoss()
lightCriterion = nn.CrossEntropyLoss()
GeneratorCriterion = nn.L1Loss()  
parsingCriterion = nn.L1Loss()

##########   GLOBAL VARIABLES   ###########
noise = torch.FloatTensor(opt.batchSize, opt.noise_dim)
input_pose_label = torch.LongTensor(opt.batchSize)
target_pose_label = torch.LongTensor(opt.batchSize)
target_pose_code = torch.FloatTensor(opt.batchSize, opt.pose_num)

input_light_label = torch.LongTensor(opt.batchSize)
target_light_label = torch.LongTensor(opt.batchSize)
target_light_code = torch.FloatTensor(opt.batchSize, opt.light_num)

identity = torch.LongTensor(opt.batchSize, opt.id_num)
label = torch.FloatTensor(1)
inputImg = torch.FloatTensor(opt.batchSize, nc, opt.fineSize, opt.fineSize)
Img = torch.FloatTensor(opt.batchSize, nc, opt.fineSize, opt.fineSize)
labelImg = torch.FloatTensor(opt.batchSize, nc, opt.fineSize, opt.fineSize)
real_label = 1
fake_label = 0

noise = Variable(noise)
input_pose_label = Variable(input_pose_label)
target_pose_code = Variable(target_pose_code)
target_pose_label = Variable(target_pose_label)

input_light_label = Variable(input_light_label)
target_light_code = Variable(target_light_code)
target_light_label = Variable(target_light_label)

identity = Variable(identity)
label = Variable(label)
inputImg = Variable(inputImg)
Img = Variable(Img)
labelImg = Variable(labelImg)

mtr = meter.ConfusionMeter(k=opt.id_num)
pose_mtr = meter.ConfusionMeter(k=opt.pose_num)
light_mtr = meter.ConfusionMeter(k=opt.light_num)

if(opt.cuda):
    noise = noise.cuda()
    input_pose_label = input_pose_label.cuda()
    target_pose_code = target_pose_code.cuda()
    target_pose_label = target_pose_label.cuda()

    input_light_label = input_light_label.cuda()
    target_light_code = target_light_code.cuda()
    target_light_label = target_light_label.cuda()

    identity = identity.cuda()
    label = label.cuda()
    inputImg = inputImg.cuda()
    Img = Img.cuda()
    labelImg = labelImg.cuda()
    GANCriterion.cuda()
    idCriterion.cuda()
    poseCriterion.cuda()
    lightCriterion.cuda()
    GeneratorCriterion.cuda()
    parsingCriterion.cuda()

########### Training   ###########
def sample_noise_pose(batchSize,pose_dim,noise_dim,light_dim,pose_index,light_index):
    # noise is a (batchSize x noise_dim) vector
    # pose is a (batchSize x pose_dim) one_hot vector
    noise = torch.Tensor(batchSize, noise_dim).uniform_(-1, 1)


    pose = torch.zeros(batchSize,pose_dim)

    light = torch.zeros(batchSize,light_dim)
    for i in range(batchSize): pose[i][pose_index[i]] = 1
    for i in range(batchSize): light[i][light_index[i]] = 1
    return noise, pose, torch.LongTensor(pose_index), light, torch.LongTensor(light_index)
#k = 0

errD_avg = 0
errG_avg = 0
errD_GAN_avg = 0
errG_GAN_avg = 0
D_losses = []
G_losses = []
D_GAN_losses = []
G_GAN_losses = []
id_acc = []
pose_acc = []
light_acc = []

def test(iteration,loader_test,loader_test_,noise_dim,pose_dim,light_dim):
    try:
        images,images_input,iden,po,il,images_label,pose_index,light_index = loader_test.next()
    except StopIteration:
        loader_test = iter(loader_test_)
        images,images_input,iden,po,il,images_label,pose_index,light_index = loader_test.next()
    Img.data.resize_(images.size()).copy_(images)
    inputImg.data.resize_(images_input.size()).copy_(images_input)
    labelImg.data.resize_(images_label.size()).copy_(images_label)

    z = torch.FloatTensor(9, noise_dim).uniform_(-1, 1)
    noise.data.resize_(z.size()).copy_(z)
    pose = torch.zeros(9,pose_dim)
    for i in range(9):
        pose[i][3] = 1
    ii = torch.zeros(9,light_dim)
    for i in range(9):
        ii[i][7] = 1
    target_pose_code.data.resize_(pose.size()).copy_(pose)
    target_light_code.data.resize_(ii.size()).copy_(ii)
    fake = netG(inputImg,noise,target_pose_code,target_light_code)
    vutils.save_image(fake.data,
        '%s/fake_samples_iteration_%03d.png' % (opt.outf, iteration), normalize=True)
    vutils.save_image(inputImg.data,
            '%s/input_samples_iteration_%03d.png' % (opt.outf, iteration), normalize=True)
    return

for iteration in range(1,opt.niter+1):
    D_corrects = 0
    G_corrects = 0
    try:
        images,images_input,iden,po,il,images_label,pose_index,light_index = loader.next()
    except StopIteration:
        loader = iter(loader_)
        images,images_input,iden,po,il,images_label,pose_index,light_index = loader.next()
    z,p,p_label,ii,i_label = sample_noise_pose(images.size(0),opt.pose_num,opt.noise_dim,opt.light_num,pose_index,light_index)

    inputImg.data.resize_(images_input.size()).copy_(images_input)
    Img.data.resize_(images.size()).copy_(images)
    noise.data.resize_(z.size()).copy_(z)
    labelImg.data.resize_(images_label.size()).copy_(images_label)

    #-------------test-----------------------------

    target_pose_code.data.resize_(p.size()).copy_(p)   #one-hot vector
    target_pose_label.data.resize_(p_label.size()).copy_(p_label)
    input_pose_label.data.resize_(po.size()).copy_(po)

    target_light_code.data.resize_(ii.size()).copy_(ii)
    target_light_label.data.resize_(i_label.size()).copy_(i_label)
    input_light_label.data.resize_(il.size()).copy_(il)

    identity.data.resize_(iden.size()).copy_(iden)

    ########### fDx ###########
    netD.zero_grad()
    # train with real data
    label.data.resize_(images.size(0)).fill_(real_label)
    id_output, pose_output, gan_output, light_output= netD(labelImg)
    D_corrects += sum(gan_output>0.5)


    errD_id = idCriterion(id_output,identity)
    errD_pose = poseCriterion(pose_output,input_pose_label)
    errD_light = lightCriterion(light_output,input_light_label)
    errD_gan = GANCriterion(gan_output,label)
    #------------2*real/fake------------
    errD_gan = 2*errD_gan
    #errD_real = errD_id + errD_pose + errD_gan + errD_light
    errD_real = errD_id + errD_gan + errD_light
    errD_real.backward()

    # train with fake data
    label.data.fill_(fake_label)
    fake = netG(inputImg,noise,target_pose_code,target_light_code)
    fake_id_output, fake_pose_output, fake_gan_output, fake_light_output = netD(fake.detach())
    D_corrects += sum(fake_gan_output<0.5)
    errD_fake = GANCriterion(fake_gan_output,label)
    #--------------2*real/fake
    errD_fake = errD_fake * 2
    errD_fake.backward()
    errD_GAN_avg += errD_gan.data[0] + errD_fake.data[0]

    errD = errD_fake + errD_real
    errD_avg += errD.data[0] - errD_gan.data[0] - errD_fake.data[0] 
    if(iteration < 5000):
        if(iteration % 2 == 0):
            optimizerD.step()
    else:
        if(iteration % 4 == 0):
            optimizerD.step()


    ########### fGx ###########
    netG.zero_grad()
    label.data.fill_(real_label)
    fake_id_output, fake_pose_output, fake_gan_output, fake_light_output = netD(fake)
    G_corrects += sum(fake_gan_output > 0.5)
    errG_gan = GANCriterion(fake_gan_output,label)
    errG_id = idCriterion(fake_id_output,identity)
    errG_pose = poseCriterion(fake_pose_output,target_pose_label)
    errG_light = lightCriterion(fake_light_output,target_light_label)
    errG_gen = GeneratorCriterion(fake,labelImg)
    errG_gen = 100*errG_gen
\
    errG = errG_gan + errG_id + errG_light + errG_gen
    errG_avg += errG.data[0] - errG_gan.data[0]
    errG_GAN_avg += errG_gan.data[0]
    errG.backward()

    optimizerG.step()
    
    

    ########## Visualize #########
    if(iteration % 100 == 0):
        print(id_output.data.size(),identity.data.size())
        mtr.add(id_output.data, identity.data)
        trainacc = mtr.value().diagonal().sum()*1.0/opt.batchSize
        #id_acc.append(trainacc)
        mtr.reset()

        pose_mtr.add(pose_output.data, input_pose_label.data)
        pose_trainacc = pose_mtr.value().diagonal().sum()*1.0/opt.batchSize
        #pose_acc.append(pose_trainacc)
        pose_mtr.reset()

        light_mtr.add(light_output.data, input_light_label.data)
        light_trainacc = light_mtr.value().diagonal().sum()*1.0/opt.batchSize
        #light_acc.append(light_trainacc)
        light_mtr.reset()


        #############tenserboard#####################
        D_corrects = D_corrects.data[0]
        G_corrects = G_corrects.data[0]
        info = {
            'GAN_acc/D_acc': D_corrects*1.0/(images.size(0)*2),
            'GAN_acc/G_acc': G_corrects*1.0/images.size(0),
        }

        for tag, value in info.items():
            logger.scalar_summary(tag, value, step=iteration)

        test(iteration,loader_test,loader_test_,opt.noise_dim,opt.pose_num,opt.light_num)

    ########### Logging #########
    print('[%d/%d] Loss_D: %.4f Loss_G: %.4f Loss_G_l1: %.4f Loss_G_gan: %.4f'
              % (iteration, opt.niter,
                 errD.data[0], errG.data[0], errG_gen.data[0], errG_gan.data[0]))
    if(iteration % opt.save_step == 0):
        torch.save(netG.state_dict(), '%s/netG_%d.pth' % (opt.outf,iteration))
        torch.save(netD.state_dict(), '%s/netD_%d.pth' % (opt.outf,iteration))
