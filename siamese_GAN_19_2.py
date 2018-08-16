#from __future__ import print_function
import cv2
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchnet.meter as meter
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from dataset import multiPIE
from siamese_model_GAN import Siamese2nd
from siamese_model_GAN import LinearDiscriminator
from contrastive import ContrastiveLoss
import torch.nn.init as weight_init
import torch.nn.functional as F
from torch.optim import lr_scheduler
import numpy as np
# import cv2
#from pycrayon import CrayonClient
 
#for plotting loss
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time,math
from logger import Logger
# from models_Parsing import ParseNet
saveFile = open('/home/shumao/wyw_files/siamese_GAN_19_2/record.txt', 'w')
saveFile.write("niter:" + str(70000) + "\n")
saveFile.write("---lr_s:" + str(0.0002) + "decay[2w, 4w]" + "\n")
saveFile.write("---lr_d:" + str(0.0001) + "decay[1w, 4w]" + "\n")
saveFile.write("D update [4:6]" + "\n")
saveFile.write("beta1:" + str(0.5) + "\n")
saveFile.write("W:0.1gan-1-1-1-1-1-5L1" + "\n")
saveFile.write("use light_index=9 GT as real img" + "\n")
saveFile.write("use usual_normal" + "\n")
saveFile.write("no mask" + "\n")
logger = Logger('./log_GAN_1_3');

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=50, help='input batch size')
parser.add_argument('--loadSize', type=int, default=100, help='the height / width of the input image to network')
parser.add_argument('--fineSize', type=int, default=96, help='the height / width of the input image to network')
parser.add_argument('--id_num', type=int, default=200, help='Total training identity.')
parser.add_argument('--pose_num', type=int, default=9, help='Total training pose.')
parser.add_argument('--light_num', type=int, default=20, help='Total training light.')
parser.add_argument('--niter', type=int, default=70000, help='number of iterations to train for')
parser.add_argument('--lr_s', type=float, default=0.0002, help='learning rate, default=0.0001')
parser.add_argument('--lr_d', type=float, default=0.0001, help='learning rate, default=0.0001')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.7')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--outf', default='/home/shumao/wyw_files/siamese_GAN_19_2', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--dataPath', default='/home/shumao/dr-gan/Data_new_realigned2/setting2/train/', help='which dataset to train on')
parser.add_argument('--modelPath_S', default='/home/shumao/wyw_files/siamese_GAN_16/netS_70000.pth', help='which model based on')
parser.add_argument('--modelPath_D', default='/home/shumao/wyw_files/siamese_GAN_16/netD_70000.pth', help='which model based on')
parser.add_argument('--save_step', type=int, default=200, help='save weights every 400 iterations ')
parser.add_argument('--labelPath', default='/home/shumao/dr-gan/Data_new_realigned2/setting2/Facedata/', help='which dataset to train on')


opt = parser.parse_args()
print(opt) # print every parser arguments
# print(opt.niter)


try:
    os.makedirs(opt.outf)
except OSError:
    pass

# w_r = 1
# w_cL = 0.02
# w_cP = 0.02
# w_cI = 0.02
# w_P = 0.02
# w_L = 0.02

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True
#---------------------Load Mask-------------------
# mask = np.load('mask_20.npy')
# mask = mask.astype(np.float32)
# M = torch.from_numpy(mask.transpose((2, 0, 1)))
# FinalMask = M.expand(opt.batchSize,3,96,96)
# print m.size()
# 3x96x96


#---------------------Load DATA-------------------------
dataset_1 = multiPIE(opt.dataPath,opt.loadSize,opt.fineSize,labelPath = opt.labelPath)
# dataset_2 = multiPIE(opt.dataPath,opt.loadSize,opt.fineSize,opt.labelPath)
dataset_test = multiPIE('/home/shumao/dr-gan/comparison/',opt.loadSize,opt.fineSize,labelPath = opt.labelPath)
loader_train_1 = torch.utils.data.DataLoader(dataset=dataset_1,
                                      batch_size = opt.batchSize,
                                      shuffle=True,
                                      num_workers=4,
                                      drop_last = True)
# loader_train_2 = torch.utils.data.Dataloader(dataset=dataset_1,
#                                       batch_size = opt.batchSize,
#                                       shuffle=True,
#                                       num_workers=4)


loader_test = torch.utils.data.DataLoader(dataset=dataset_test,
                                          batch_size = 9,
                                          shuffle=False,
                                          num_workers=4)
data_train_1 = iter(loader_train_1)
# data_train_2 = iter(loader_train_2)
data_test = iter(loader_test)


#----------------------Parameters-----------------------
# num_pose = opt.pose_num
# num_light = opt.light_num
# num_iden = opt.id_num


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') !=-1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') !=-1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


netD = LinearDiscriminator()
# netD = netD.apply(weights_init)
netD.load_state_dict(torch.load(opt.modelPath_D))

netS = Siamese2nd()
# netS = netS.apply(weights_init)
netS.load_state_dict(torch.load(opt.modelPath_S))

#-----------------params freeze-----------------
# for param in netS.conv11.parameters():
#     param.requires_grad = False
# for param in netS.conv1r.parameters():
#     param.requires_grad = False
# for param in netS.conv12.parameters():
#     param.requires_grad = False
# for param in netS.conv21.parameters():
#     param.requires_grad = False
# for param in netS.conv22.parameters():
#     param.requires_grad = False
# for param in netS.conv23.parameters():
#     param.requires_grad = False
# for param in netS.conv31.parameters():
#     param.requires_grad = False
# for param in netS.conv32.parameters():
#     param.requires_grad = False
# for param in netS.conv33.parameters():
#     param.requires_grad = False
# for param in netS.conv41.parameters():
#     param.requires_grad = False
# for param in netS.conv42.parameters():
#     param.requires_grad = False
# for param in netS.conv43.parameters():
#     param.requires_grad = False
# for param in netS.conv51.parameters():
#     param.requires_grad = False
# for param in netS.conv52.parameters():
#     param.requires_grad = False
# for param in netS.conv53.parameters():
#     param.requires_grad = False
# for param in netS.convfc.parameters():
#     param.requires_grad = False
#-----------------freeze D
# for param in netD.parameters():
#     param.requires_grad = False


#-----------------params freeze-----------------
if(opt.cuda):
    netS.cuda()
    netD.cuda()
#-------------------Loss & Optimization
optimizerD = torch.optim.Adam(netD.parameters(),lr=opt.lr_d, betas=(opt.beta1, 0.999))
optimizerS = torch.optim.Adam(netS.parameters(),lr=opt.lr_s, betas=(opt.beta1, 0.999))
# optimizerD = torch.optim.Adam(filter(lambda p: p.requires_grad, netD.parameters()),lr=opt.lr_d, betas=(opt.beta1, 0.999))

S_schedular = torch.optim.lr_scheduler.MultiStepLR(optimizerS, [20000, 40000], gamma = 0.5)
D_schedular = torch.optim.lr_scheduler.MultiStepLR(optimizerD, [10000, 40000], gamma = 0.5)


poss_contrastive_loss = ContrastiveLoss() # load from the begining
light_contrastive_loss = ContrastiveLoss()
identity_contrastive_loss = ContrastiveLoss()
reconstructe_loss = nn.MSELoss()
pose_class_loss = nn.CrossEntropyLoss()
light_class_loss = nn.CrossEntropyLoss()
GANCriterion = nn.BCELoss()
#------------------ Global Variables------------------
input_pose_1 = torch.LongTensor(opt.batchSize)
input_light_1 = torch.LongTensor(opt.batchSize)
# input_pose_2 = torch.LongTensor(opt.batchSize)
# input_light_2 = torch.LongTensor(opt.batchSize)

inputImg_1 = torch.FloatTensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
inputImg_2 = torch.FloatTensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
GT = torch.FloatTensor(opt.batchSize, 3,opt.fineSize, opt.fineSize)
same_pose = torch.FloatTensor(opt.batchSize)
same_iden = torch.FloatTensor(opt.batchSize)
same_light = torch.FloatTensor(opt.batchSize)

target = torch.FloatTensor(opt.batchSize)

real_label = 1
fake_label = 0
# w_1 = torch.FloatTensor(1)
# w_2 = torch.FloatTensor(20)
# w_3 = torch.FloatTensor(10)
# w_4 = torch.FloatTensor(10)
# w_5 = torch.FloatTensor(10)
# w_6 = torch.FloatTensor(20)
# output_pose_1_label = torch.LongTensor(opt.batchSize)
# output_pose_2_label = torch.LongTensor(opt.batchSize)
# output_light_1_label = torch.LongTensor(opt.batchSize)
# output_light_2_label = torch.LongTensor(opt.batchSize)

input_pose_1 = Variable(input_pose_1)
# input_pose_2 = Variable(input_pose_2)
input_light_1 = Variable(input_light_1)
# input_light_2 = Variable(input_light_2)

inputImg_1 = Variable(inputImg_1)
inputImg_2 = Variable(inputImg_2)
GT = Variable(GT)
same_pose = Variable(same_pose)
same_iden = Variable(same_iden)
same_light = Variable(same_light)
target = Variable(target)

# FinalMask = Variable(FinalMask)

# w_1 = Variable(w_1, requires_grad = False)
# w_2 = Variable(w_2, requires_grad = False)
# w_3 = Variable(w_3, requires_grad = False)
# w_4 = Variable(w_4, requires_grad = False)
# w_5 = Variable(w_5, requires_grad = False)
# w_6 = Variable(w_6, requires_grad = False)


pose_mtr = meter.ConfusionMeter(k=opt.pose_num)
light_mtr = meter.ConfusionMeter(k=opt.light_num)


if(opt.cuda):

    input_pose_1 = input_pose_1.cuda()
    # input_pose_2 = input_pose_2.cuda()
    input_light_1 = input_light_1.cuda()
    # input_light_2 = input_light_2.cuda()
    inputImg_1 = inputImg_1.cuda()
    inputImg_2 = inputImg_2.cuda()
    GT = GT.cuda()
    same_pose = same_pose.cuda()
    same_light = same_light.cuda()
    same_iden = same_iden.cuda()
    target = target.cuda()

    # FinalMask = FinalMask.cuda()

    # w_1 = w_1.cuda()
    # w_2 = w_1.cuda()
    # w_3 = w_1.cuda()
    # w_4 = w_1.cuda()
    # w_5 = w_1.cuda()
    # w_6 = w_1.cuda()
    # poss_contrastive_loss.cuda()
    # light_contrastive_loss.cuda()
    # identity_contrastive_loss.cuda()
    pose_class_loss.cuda()
    light_class_loss.cuda()
    reconstructe_loss.cuda()
    GANCriterion.cuda()

#------------------test---------

# k = 0 # for meter

err_total = 0
err_recon = 0
err_contraL = 0
err_contraP = 0
err_contraI = 0
err_classP = 0
err_classL = 0

def test(iteration, data_test, loader_test):
    try:
        images_1,po_1,li_1,GT_1,by_image,same_po,same_li,same_id, GT_pose, GT_light = data_test.next()
    except StopIteration:
        data_test = iter(loader_test)
        images_1,po_1,li_1,GT_1,by_image,same_po,same_li,same_id, GT_pose, GT_light = data_test.next()

    GT.data.resize_(GT_1.size()).copy_(GT_1)
    inputImg_1.data.resize_(images_1.size()).copy_(images_1)
    inputImg_2.data.resize_(by_image.size()).copy_(by_image)
    input_pose_1.data.resize_(po_1.size()).copy_(po_1)
    input_light_1.data.resize_(li_1.size()).copy_(li_1)


    output_pose_1, output_pose_2, output_light_1, output_light_2, out_f_1, out_f_2, out = netS(inputImg_1, inputImg_2)
    vutils.save_image(out.data,
        '%s/fake_samples_iteration_%03d.png' % (opt.outf, iteration), normalize=True)
    vutils.save_image(inputImg_1.data,
            '%s/input_samples_iteration_%03d.png' % (opt.outf, iteration), normalize=True)
    # output = F.cosine_similarity(out, GT)
    # cos = torch.mean(output)
    # # print out_f_1.data.type()
    # # print output.data.type()
    # print('----------------------------')
    # print('test_cos: %.4f ' %(cos.data[0]))
    # print('----------------------------')
    # saveFile.write('[%d/%d] test_cos: %.4f ' %(iteration, opt.niter, cos.data[0]) + "\n")


#-------------------train----------------------
for iteration in range(1,opt.niter+1):
    running_corrects = 0
    running_corrects_light = 0
    D_corrects = 0
    S_corrects = 0

    S_schedular.step()
    D_schedular.step()

    try:
        images_1,po_1,li_1,GT_1,by_image,same_po,same_li,same_id, GT_pose, GT_light = data_train_1.next()
    except StopIteration:
        data_train_1 = iter(loader_train_1)
        images_1,po_1,li_1,GT_1,by_image,same_po,same_li,same_id, GT_pose, GT_light = data_train_1.next()

    GT.data.resize_(GT_1.size()).copy_(GT_1)



    inputImg_1.data.resize_(images_1.size()).copy_(images_1)
    inputImg_2.data.resize_(by_image.size()).copy_(by_image)

    input_pose_1.data.resize_(po_1.size()).copy_(po_1)
    input_light_1.data.resize_(li_1.size()).copy_(li_1)

    same_pose.data.resize_(same_po.size()).copy_(same_po)
    same_light.data.resize_(same_li.size()).copy_(same_li)
    same_iden.data.resize_(same_id.size()).copy_(same_id)
    #-----------------add D-------------
    for p in netD.parameters():
        p.requires_grad = True

    netD.zero_grad()
    # # train with real images
    target.data.resize_(images_1.size(0)).fill_(real_label)
    real_gan_output = netD(GT)

    D_corrects += sum(real_gan_output > 0.5)

    errD_real = GANCriterion(real_gan_output, target)
    # errD_real = 1 * errD_real
    errD_real.backward()
    
    # train with fake images
    target.data.fill_(fake_label)
    output_pose_1, output_pose_2, output_light_1, output_light_2, out_f_1, out_f_2, out = netS(inputImg_1, inputImg_2)

    fake_gan_output = netD(out.detach())
    D_corrects += sum(fake_gan_output < 0.5)
    errD_fake = GANCriterion(fake_gan_output, target)
    # errD_fake = 1 * errD_fake

    errD_fake.backward()

    # in order to make D is not that good
    # if iteration > 5000:
    #     if iteration % 4 == 0:
    #         optimizerD.step()
    # else:
    if iteration % 6 == 0:
        optimizerD.step()
    #-------------modified S---------------

    for p in netD.parameters():
        p.requires_grad = False

    netS.zero_grad()
    target.data.fill_(real_label)
    # netS target is to generate a real-like image
    fake_gan_output = netD(out)
    S_corrects += sum(fake_gan_output > 0.5)
    errS_GAN = GANCriterion(fake_gan_output, target)
    err_contraI = identity_contrastive_loss(out_f_1, out_f_2, same_iden)
    err_contraP = poss_contrastive_loss(output_pose_1, output_pose_2, same_pose)
    err_contraL = light_contrastive_loss(output_light_1,output_light_2, same_light)
    err_classL = light_class_loss(output_light_1, input_light_1)
    err_classP = pose_class_loss(output_pose_1, input_pose_1)

    # # if iteration > 40000:
    # out = FinalMask * out
    # GT = FinalMask * GT
    err_recon = reconstructe_loss(out, GT)
    

    errS_GAN = 0.1 * errS_GAN
    err_contraI = 1 *err_contraI
    err_contraP = 1*err_contraP
    err_contraL = 1*err_contraL
    err_classL = 1*err_classL
    err_classP = 1*err_classP
    err_recon = 5 * err_recon

    errS = errS_GAN + err_contraI + err_contraP + err_contraL + err_classL + err_classP + err_recon
    errS.backward()
    optimizerS.step()





    # output_pose_1, output_pose_2, output_light_1, output_light_2, out_f_1, out_f_2, out = netS(inputImg_1, inputImg_2)
    #-----------------mask test area-----------------------------
    # print out.data.type()
    # print GT.data.type()
    # print FinalMask.data.type() same
    # print FinalMask.data.size() 64x3x96x96
    # Final_out = FinalMask * out
    # Final_GT = FinalMask * GT





    #-----------------mask test area-----------------------------
    # f_1 & f_2 variable
    # same_iden variable
    # print(err_recon.data.size())
    # print(err_contraL.data.size())
    # print(err_classP.data.size())
    # modify the contrastive loss function to make contrastive loss be 1Lx1L 
    # contrastive loss and Softmax and Loss1 are all requires_grad
    # err_total = 1 * err_recon + 10 * err_contraP + 10 * err_contraI + 10 * err_classP + 20 * err_classL
    # err_total = err_recon + err_contraI + err_contraP + err_contraL + err_classL + err_classP
    # err_total = w_r * err_recon
    # err_total.backward()
    # optimizerS.step()

    #----------------------Visualize-----------
    if(iteration % 200 == 0):

        pose_mtr.add(output_pose_1.data, input_pose_1.data)
        pose_trainacc = pose_mtr.value().diagonal().sum()*1.0/opt.batchSize
        pose_mtr.reset()

        light_mtr.add(output_light_1.data, input_light_1.data)
        light_trainacc = light_mtr.value().diagonal().sum()*1.0/opt.batchSize
        light_mtr.reset()
        #-----------------------------------------
        D_corrects = D_corrects.data[0]
        S_corrects = S_corrects.data[0]
        info = {
            'GAN_acc/D_acc': D_corrects*1.0/(images_1.size(0)*2),
            'GAN_acc/G_acc': S_corrects*1.0/images_1.size(0),
        }

        for tag, value in info.items():
            logger.scalar_summary(tag, value, step=iteration)


        test(iteration, data_test, loader_test)

    #record the first loss situtation
    if(iteration == 0 or iteration == 1 or iteration == 2 or iteration == 3):
        saveFile.write('[%d/%d] errD_real: %.4f ' %(iteration, opt.niter, errD_real.data[0]) + "\n")
        saveFile.write('[%d/%d] errD_fake: %.4f ' %(iteration, opt.niter, errD_fake.data[0]) + "\n")
        saveFile.write('[%d/%d] errS_GAN: %.4f ' %(iteration, opt.niter, errS_GAN.data[0]*10) + "\n")
        saveFile.write("+++++++++++++++++++++++++++++" + "\n")

    if (iteration % 1000 ==0):
        saveFile.write('[%d/%d] errD_real: %.4f ' %(iteration, opt.niter, errD_real.data[0]) + "\n")
        saveFile.write('[%d/%d] errD_fake: %.4f ' %(iteration, opt.niter, errD_fake.data[0]) + "\n")
        saveFile.write('[%d/%d] errS_GAN: %.4f ' %(iteration, opt.niter, errS_GAN.data[0]*10) + "\n")
        saveFile.write("+++++++++++++++++++++++++++++" + "\n")





    # #pose prediction

    # preds_pose = torch.max(output_pose_1.data, 1)
    # running_corrects += torch.sum(preds == input_pose_1)
    # print('pose_accuracy: %.2f' 
    #         % (running_corrects * 1.0/images.size(0)))
    
    # #light prediction
    # preds_light = torch.max(output_light_1.data, 1)
    # running_corrects_light += torch.sum(preds_light == input_light_1)
    # print('light_accuracy: %.2f' 
    #         % (running_corrects_light * 1.0/images.size(0)))
    print('----------------------------------------')
    print('[%d/%d] errD_real: %.4f ' %(iteration, opt.niter, errD_real.data[0]))
    print('        errD_fake: %.4f ' %(errD_fake.data[0]))
    print('        errS_GAN: %.4f ' %(errS_GAN.data[0]*10))
    print('        Reco_S: %.4f ' %(err_recon.data[0]/5))
    print('        conL_S: %.4f ' %(err_contraL.data[0]))
    print('        conP_S: %.4f ' %(err_contraP.data[0]))
    print('        conI_S: %.4f ' %(err_contraI.data[0]))
    print('        Clas_P: %.4f ' %(err_classP.data[0]))
    print('        Clas_L: %.4f ' %(err_classL.data[0]))



    if(iteration % opt.save_step == 0):
        torch.save(netS.state_dict(), '%s/netS_%d.pth' % (opt.outf,iteration))
        torch.save(netD.state_dict(), '%s/netD_%d.pth' % (opt.outf,iteration))

