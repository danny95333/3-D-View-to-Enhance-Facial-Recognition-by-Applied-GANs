import torch.nn as nn
import torch
import torch.nn.functional as f
class Siamese(nn.Module):
    def __init__(self):
        super(Siamese,self).__init__()
        # 3x96x96
        self.conv11 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ELU(True))
        # 3x96x96 conv1r has RES
        self.conv1r = nn.Sequential(nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0))
        # 64x96x96 
        self.conv12 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(64))
        # sum with residual, then pass the ELU block
        self.elu12 = nn.ELU(True)
        #-------------------------------------------1st layer
        # 64x96x96 conv21 has RES
        self.conv21 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ELU(True))
        # 128x48x48
        self.conv22 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ELU(True))
        # 64x48x48
        self.conv23 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(128))
        # sum with residual, then pass to ELU block
        self.elu23 = nn.ELU(True)
        #-------------------------------------------2nd layer
        # 128x48x48 conv31 has RES
        self.conv31 = nn.Sequential(nn.Conv2d(128, 192, kernel_size=3, stride=2, padding=1),
                                    nn.BatchNorm2d(192),
                                    nn.ELU(True))
        # 192x24x24
        self.conv32 = nn.Sequential(nn.Conv2d(192, 96, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(96),
                                    nn.ELU(True))
        # 96x24x24
        self.conv33 = nn.Sequential(nn.Conv2d(96, 192, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(192))
        # sum with residual, then pass to ELU block
        self.elu33 = nn.ELU(True)
        #-------------------------------------------3rd layer
        #192x24x24 conv41 has RES
        self.conv41 = nn.Sequential(nn.Conv2d(192, 256, kernel_size=3, stride=2, padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.ELU(True))
        #256x12x12
        self.conv42 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ELU(True))
        #128x12x12
        self.conv43 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(256))
        # sum with residual, then pass to ELU block
        self.elu43 = nn.ELU(True)
        #-------------------------------------------4th layer
        #256x12x12
        self.conv51 = nn.Sequential(nn.Conv2d(256, 320, kernel_size=3, stride=2, padding=1),
                                    nn.BatchNorm2d(320),
                                    nn.ELU(True))
        #320x6x6
        self.conv52 = nn.Sequential(nn.Conv2d(320, 160, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(160),
                                    nn.ELU(True))
        #160x6x6
        self.conv53 = nn.Sequential(nn.Conv2d(160, 320, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(320))
        # sum with residual, then pass to ELU block
        self.elu53 = nn.ELU(True)
        #output size 320x6x6
        #-------------------------------------------5th layer
        #-------------------------------------------decomposition
        #320x6x6
        # view->240 + 80
        #
        #240x6x6 Image_1 and Image_2 compute contrastive loss

        # 80 pas through conv to be 27
        # 80x6x6
        self.convfc = nn.Conv2d(80, 29, kernel_size=6, stride=1, padding=0)
        #27x1x1
        #27 slice into 20x1x1 + 7x1x1
        # self.fc_pose = nn.Linear(27 , 20)
        # self.fc_light = nn.Linear(27, 7)
        #compute contractiveLoss
        #compute Softmax of pose and light
        #predict Label of pose and light
        #-------------------------------------------decoder-------------------------------------------------------------------
        #240x6x6
        self.dconv52 = nn.Sequential(nn.ConvTranspose2d(240, 160, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(160),
                                     nn.ELU(True))
        #160x6x6
        self.dconv51 = nn.Sequential(nn.ConvTranspose2d(160, 256, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(256),
                                     nn.ELU(True))
        #256x6x6 bilinear interpolation upsamping
        #self.upsampling43 = nn.UpsamplingBilinear2d(scale_factor=2)
        #256x12x12
        self.dconv43 = nn.Sequential(nn.ConvTranspose2d(256 , 256, kernel_size = 3, stride=2, padding=1, output_padding=1),
                                     nn.BatchNorm2d(256),
                                     nn.ELU(True))
        #256x12x12
        self.dconv42 = nn.Sequential(nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(128),
                                     nn.ELU(True))
        #128x12x12
        self.dconv41 = nn.Sequential(nn.ConvTranspose2d(128, 192, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(192),
                                     nn.ELU(True))
        #192x12x12 bilinear interpolation upsamping
        #self.upsampling33 = nn.UpsamplingBilinear2d(scale_factor=2)
        #192x24x24
        self.dconv33 = nn.Sequential(nn.ConvTranspose2d(192 , 192, kernel_size = 3, stride=2, padding=1, output_padding=1),
                                     nn.BatchNorm2d(192),
                                     nn.ELU(True))
        #192x24x24
        self.dconv32 = nn.Sequential(nn.ConvTranspose2d(192, 96, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(96),
                                     nn.ELU(True))
        #96x24x24
        self.dconv31 = nn.Sequential(nn.ConvTranspose2d(96, 128, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(128),
                                     nn.ELU(True))
        #128x24x24 bilinear interpolation upsamping
        #self.upsampling23 = nn.UpsamplingBilinear2d(scale_factor=2)
        #128x48x48
        self.dconv23 = nn.Sequential(nn.ConvTranspose2d(128 , 128, kernel_size = 3, stride=2, padding=1, output_padding=1),
                                     nn.BatchNorm2d(128),
                                     nn.ELU(True))
        #128x48x48
        self.dconv22 = nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(64),
                                     nn.ELU(True))
        #64x48x48
        self.dconv21 = nn.Sequential(nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(64),
                                     nn.ELU(True))
        #64x48x48 bilinear interpolation upsamping
        #self.upsampling13 = nn.UpsamplingBilinear2d(scale_factor=2)
        #64x96x96
        self.dconv12 = nn.Sequential(nn.ConvTranspose2d(64 , 64, kernel_size = 3, stride=2, padding=1, output_padding=1),
                                     nn.BatchNorm2d(64),
                                     nn.ELU(True))
        #64x96x96
        self.dconv11 = nn.Sequential(nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(32),
                                     nn.ELU(True))
        #32x96x96
        self.output = nn.Sequential(nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1),
                                    nn.Tanh())
        #3x96x96
        #self.tanh = nn.tanh()
        #compute the L1 loss(Lrecon) between GT and reconstruction 
    def forward(self, x, x_2):
        #-----layer 1------------------------------
        out_res_1 = self.conv1r(x)
        out = self.conv11(x)
        out = self.conv12(out)
        out = out + out_res_1
        out = self.elu12(out)

        out_res_1_2 = self.conv1r(x_2)
        out_2 = self.conv11(x_2)
        out_2 = self.conv12(out_2)
        out_2 = out_2 + out_res_1_2
        out_2 = self.elu12(out_2)
        #-----layer 2------------------------------
        out_res_2 = self.conv21(out)
        out = self.conv22(out_res_2)
        out = self.conv23(out)
        out = out + out_res_2
        out = self.elu23(out)

        out_res_2_2 = self.conv21(out_2)
        out_2 = self.conv22(out_res_2_2)
        out_2 = self.conv23(out_2)
        out_2 = out_2 + out_res_2_2
        out_2 = self.elu23(out_2)
        #-----layer 3-------------------------------
        out_res_3 = self.conv31(out)
        out = self.conv32(out_res_3)
        out = self.conv33(out)
        out = out + out_res_3
        out = self.elu33(out)

        out_res_3_2 = self.conv31(out_2)
        out_2 = self.conv32(out_res_3_2)
        out_2 = self.conv33(out_2)
        out_2 = out_2 + out_res_3_2
        out_2 = self.elu33(out_2)
        #-----layer 4--------------------------------
        out_res_4 = self.conv41(out)
        out = self.conv42(out_res_4)
        out = self.conv43(out)
        out = out + out_res_4
        out = self.elu43(out)

        out_res_4_2 = self.conv41(out_2)
        out_2 = self.conv42(out_res_4_2)
        out_2 = self.conv43(out_2)
        out_2 = out_2 + out_res_4_2
        out_2 = self.elu43(out_2)
        #-----layer 5--------------------------------
        out_res_5 = self.conv51(out)
        out = self.conv52(out_res_5)
        out = self.conv53(out)
        out = out + out_res_5
        out = self.elu53(out)

        out_res_5_2 = self.conv51(out_2)
        out_2 = self.conv52(out_res_5_2)
        out_2 = self.conv53(out_2)
        out_2 = out_2 + out_res_5_2
        out_2 = self.elu53(out_2)
        #----------------------------------------decomposition----------------------------------------------
        out_240 = out.narrow(1,0,240) #second dimension[channels]slices from 0 to 239
        out_80 = out.narrow(1,240,80) #second dimension[channels]slices from 240 to 319
        out_240_2 = out_2.narrow(1, 0, 240)
        out_80_2 = out_2.narrow(1, 240, 80)

        out_29 = self.convfc(out_80)
        out_29_2 = self.convfc(out_80_2)
        #29x1x1
        out_pose = out_29.narrow(1,0,9) #second dimension[channels]slices from 0 to 6 for pose

        #9x1x1
        out_pose_2 = out_29_2.narrow(1, 0, 9)


        out_light = out_29.narrow(1,9,20) #second dimension[channels]slices from 7 to 26 for light
        #20x1x1

        out_light_2 = out_29_2.narrow(1, 9 , 20)


        #-----------------------------------------decoder---------------------------------------------------
        out = self.dconv52(out_240)
        out = self.dconv51(out)
        
        # out = f.upsample(out, scale_factor = 2, mode='bilinear')
        out = self.dconv43(out)
        out = self.dconv42(out)
        out = self.dconv41(out)
        
        # out = f.upsample(out, scale_factor = 2, mode='bilinear')
        out = self.dconv33(out)
        out = self.dconv32(out)
        out = self.dconv31(out)
        
        # out = f.upsample(out, scale_factor = 2, mode='bilinear')
        out = self.dconv23(out)
        out = self.dconv22(out)
        out = self.dconv21(out)

        # out = f.upsample(out, scale_factor = 2, mode='bilinear')
        out = self.dconv12(out)
        out = self.dconv11(out)
        out = self.output(out)
        # out = out.tanh()

        out_pose = out_pose.contiguous()
        out_pose_2 = out_pose_2.contiguous()
        out_light = out_light.contiguous()
        out_light_2 = out_light_2.contiguous()

        out_pose = out_pose.view(out_pose.size(0),-1)
        out_pose_2 = out_pose_2.view(out_pose_2.size(0),-1)
        out_light = out_light.view(out_light.size(0),-1)
        out_light_2 = out_light_2.view(out_light_2.size(0),-1)

        out_240 = out_240.contiguous()
        out_240 = out_240.view(64, -1)
        # 64x8640
        out_240_2 = out_240_2.contiguous()
        out_240_2 = out_240_2.view(64, -1)
        #--------------------------------------------LOSS
        # contractive loss between gt and image_1's identity
        return out_pose, out_pose_2, out_light, out_light_2, out_240, out_240_2, out


class Siamese2nd(nn.Module):
    def __init__(self):
        super(Siamese2nd,self).__init__()
        # 3x96x96
        self.conv11 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(32),
                                    nn.ELU(True))
        # 32x96x96 
        self.conv12 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ELU(True))

        #-------------------------------------------1st layer
        # 64x96x96
        self.conv21 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ELU(True))
        # 64x48x48
        self.conv22 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ELU(True))
        # 64x48x48
        self.conv23 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ELU(True))
        #-------------------------------------------2nd layer
        # 128x48x48 conv31 has RES
        self.conv31 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ELU(True))
        # 128x24x24
        self.conv32 = nn.Sequential(nn.Conv2d(128, 96, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(96),
                                    nn.ELU(True))
        # 96x24x24
        self.conv33 = nn.Sequential(nn.Conv2d(96, 192, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(192),
                                    nn.ELU(True))
        #-------------------------------------------3rd layer
        #192x24x24 conv41 has RES
        self.conv41 = nn.Sequential(nn.Conv2d(192, 192, kernel_size=3, stride=2, padding=1),
                                    nn.BatchNorm2d(192),
                                    nn.ELU(True))
        #192x12x12
        self.conv42 = nn.Sequential(nn.Conv2d(192, 128, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ELU(True))
        #128x12x12
        self.conv43 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.ELU(True))
        #-------------------------------------------4th layer
        #256x12x12
        self.conv51 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.ELU(True))
        #256x6x6
        self.conv52 = nn.Sequential(nn.Conv2d(256, 160, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(160),
                                    nn.ELU(True))
        #160x6x6
        self.conv53 = nn.Sequential(nn.Conv2d(160, 320, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(320),
                                    nn.ELU(True))
        #output size 320x6x6
        #-------------------------------------------5th layer
        #-------------------------------------------decomposition
        #320x6x6
        # view->240 + 80
        #
        #240x6x6 Image_1 and Image_2 compute contrastive loss

        # 80 pas through conv to be 27
        # 80x6x6
        self.convfc = nn.Conv2d(80, 29, kernel_size=6, stride=1, padding=0)
        #27x1x1
        #27 slice into 20x1x1 + 7x1x1
        # self.fc_pose = nn.Linear(27 , 20)
        # self.fc_light = nn.Linear(27, 7)
        #compute contractiveLoss
        #compute Softmax of pose and light
        #predict Label of pose and light
        #-------------------------------------------decoder-------------------------------------------------------------------
        #240x6x6
        self.dconv52 = nn.Sequential(nn.ConvTranspose2d(240, 160, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(160),
                                     nn.ELU(True))
        #160x6x6
        self.dconv51 = nn.Sequential(nn.ConvTranspose2d(160, 256, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(256),
                                     nn.ELU(True))
        #256x6x6
        self.dconv43 = nn.Sequential(nn.ConvTranspose2d(256 , 256, kernel_size = 3, stride=2, padding=1, output_padding=1),
                                     nn.BatchNorm2d(256),
                                     nn.ELU(True))
        #256x12x12
        self.dconv42 = nn.Sequential(nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(128),
                                     nn.ELU(True))
        #128x12x12
        self.dconv41 = nn.Sequential(nn.ConvTranspose2d(128, 192, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(192),
                                     nn.ELU(True))
        #192x12x12
        self.dconv33 = nn.Sequential(nn.ConvTranspose2d(192 , 192, kernel_size = 3, stride=2, padding=1, output_padding=1),
                                     nn.BatchNorm2d(192),
                                     nn.ELU(True))
        #192x24x24
        self.dconv32 = nn.Sequential(nn.ConvTranspose2d(192, 96, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(96),
                                     nn.ELU(True))
        #96x24x24
        self.dconv31 = nn.Sequential(nn.ConvTranspose2d(96, 128, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(128),
                                     nn.ELU(True))
        #128x24x24
        self.dconv23 = nn.Sequential(nn.ConvTranspose2d(128 , 128, kernel_size = 3, stride=2, padding=1, output_padding=1),
                                     nn.BatchNorm2d(128),
                                     nn.ELU(True))
        #128x48x48
        self.dconv22 = nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(64),
                                     nn.ELU(True))
        #64x48x48
        self.dconv21 = nn.Sequential(nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(64),
                                     nn.ELU(True))
        #64x48x48
        self.dconv12 = nn.Sequential(nn.ConvTranspose2d(64 , 64, kernel_size = 3, stride=2, padding=1, output_padding=1),
                                     nn.BatchNorm2d(64),
                                     nn.ELU(True))
        #64x96x96
        self.dconv11 = nn.Sequential(nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(32),
                                     nn.ELU(True))
        #32x96x96
        self.output = nn.Sequential(nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1),
                                    nn.Tanh())
        #3x96x96
    def forward(self, x, x_2):
        #-----layer 1------------------------------
        out = self.conv11(x)
        out = self.conv12(out)

        out_2 = self.conv11(x_2)
        out_2 = self.conv12(out_2)
        #-----layer 2------------------------------
        out = self.conv21(out)
        out = self.conv22(out)
        out = self.conv23(out)

        out_2 = self.conv21(out_2)
        out_2 = self.conv22(out_2)
        out_2 = self.conv23(out_2)

        #-----layer 3-------------------------------
        out = self.conv31(out)
        out = self.conv32(out)
        out = self.conv33(out)

        out_2 = self.conv31(out_2)
        out_2 = self.conv32(out_2)
        out_2 = self.conv33(out_2)

        #-----layer 4--------------------------------
        out = self.conv41(out)
        out = self.conv42(out)
        out = self.conv43(out)

        out_2 = self.conv41(out_2)
        out_2 = self.conv42(out_2)
        out_2 = self.conv43(out_2)

        #-----layer 5--------------------------------
        out = self.conv51(out)
        out = self.conv52(out)
        out = self.conv53(out)

        out_2 = self.conv51(out_2)
        out_2 = self.conv52(out_2)
        out_2 = self.conv53(out_2)

        #----------------------------------------decomposition----------------------------------------------
        out_240 = out.narrow(1,0,240) #second dimension[channels]slices from 0 to 239
        out_80 = out.narrow(1,240,80) #second dimension[channels]slices from 240 to 319
        out_240_2 = out_2.narrow(1, 0, 240)
        out_80_2 = out_2.narrow(1, 240, 80)

        out_29 = self.convfc(out_80)
        out_29_2 = self.convfc(out_80_2)
        #29x1x1
        out_pose = out_29.narrow(1,0,9) #second dimension[channels]slices from 0 to 6 for pose

        #9x1x1
        out_pose_2 = out_29_2.narrow(1, 0, 9)


        out_light = out_29.narrow(1,9,20) #second dimension[channels]slices from 7 to 26 for light
        #20x1x1

        out_light_2 = out_29_2.narrow(1, 9 , 20)


        #-----------------------------------------decoder---------------------------------------------------
        out = self.dconv52(out_240)
        out = self.dconv51(out)
        
        # out = f.upsample(out, scale_factor = 2, mode='bilinear')
        out = self.dconv43(out)
        out = self.dconv42(out)
        out = self.dconv41(out)
        
        # out = f.upsample(out, scale_factor = 2, mode='bilinear')
        out = self.dconv33(out)
        out = self.dconv32(out)
        out = self.dconv31(out)
        
        # out = f.upsample(out, scale_factor = 2, mode='bilinear')
        out = self.dconv23(out)
        out = self.dconv22(out)
        out = self.dconv21(out)

        # out = f.upsample(out, scale_factor = 2, mode='bilinear')
        out = self.dconv12(out)
        out = self.dconv11(out)
        out = self.output(out)

        out_pose = out_pose.contiguous()
        out_pose_2 = out_pose_2.contiguous()
        out_light = out_light.contiguous()
        out_light_2 = out_light_2.contiguous()

        out_pose = out_pose.view(out_pose.size(0),-1)
        out_pose_2 = out_pose_2.view(out_pose_2.size(0),-1)
        out_light = out_light.view(out_light.size(0),-1)
        out_light_2 = out_light_2.view(out_light_2.size(0),-1)

        out_240 = out_240.contiguous()
        out_240 = out_240.view(64, -1)
        # 64x8640
        out_240_2 = out_240_2.contiguous()
        out_240_2 = out_240_2.view(64, -1)
        #--------------------------------------------LOSS
        # contractive loss between gt and image_1's identity
        return out_pose, out_pose_2, out_light, out_light_2, out_240, out_240_2, out



class LinearDiscriminator(nn.Module):
    def __init__(self):
        super(LinearDiscriminator,self).__init__()
        #3x96x96
        self.conv11 = nn.Sequential(nn.Conv2d(3, 32, kernel_size = 3, stride = 1, padding = 1),
                                    nn.BatchNorm2d(32),
                                    nn.ELU(True))
        #32x96x96
        self.conv12 = nn.Sequential(nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1),
                                    nn.BatchNorm2d(64),
                                    nn.ELU(True))
        #64x96x96
        self.conv21 = nn.Sequential(nn.Conv2d(64, 64, kernel_size = 3, stride = 2, padding = 1),
                                    nn.BatchNorm2d(64),
                                    nn.ELU(True))
        #64x48x48
        self.conv22 = nn.Sequential(nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1),
                                    nn.BatchNorm2d(64),
                                    nn.ELU(True))
        #64x48x48
        self.conv23 = nn.Sequential(nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
                                    nn.BatchNorm2d(128),
                                    nn.ELU(True))
        #128x48x48
        self.conv31 = nn.Sequential(nn.Conv2d(128, 128, kernel_size = 3, stride = 2, padding = 1),
                                    nn.BatchNorm2d(128),
                                    nn.ELU(True))
        #128x24x24
        self.conv32 = nn.Sequential(nn.Conv2d(128, 96, kernel_size = 3, stride = 1, padding = 1),
                                    nn.BatchNorm2d(96),
                                    nn.ELU(True))
        #96x24x24
        self.conv33 = nn.Sequential(nn.Conv2d(96, 192, kernel_size = 3, stride = 1, padding = 1),
                                    nn.BatchNorm2d(192),
                                    nn.ELU(True))
        #192x24x24
        self.conv41 = nn.Sequential(nn.Conv2d(192, 192, kernel_size = 3, stride = 2, padding = 1),
                                    nn.BatchNorm2d(192),
                                    nn.ELU(True))
        #192x12x12
        self.conv42 = nn.Sequential(nn.Conv2d(192, 128, kernel_size = 3, stride = 1, padding = 1),
                                    nn.BatchNorm2d(128),
                                    nn.ELU(True))
        #128x12x12
        self.conv43 = nn.Sequential(nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1),
                                    nn.BatchNorm2d(256),
                                    nn.ELU(True))
        #256x12x12
        self.conv51 = nn.Sequential(nn.Conv2d(256, 256, kernel_size = 3, stride = 2, padding = 1),
                                    nn.BatchNorm2d(256),
                                    nn.ELU(True))
        #256x6x6
        self.conv52 = nn.Sequential(nn.Conv2d(256, 160, kernel_size = 3, stride = 1, padding = 1),
                                    nn.BatchNorm2d(160),
                                    nn.ELU(True))
        #160x6x6
        self.conv53 = nn.Sequential(nn.Conv2d(160, 320, kernel_size = 3, stride = 1, padding = 1),
                                    nn.BatchNorm2d(320),
                                    nn.ELU(True))
        #320x6x6

        #1.conv + linear
        self.avgpool = nn.AvgPool2d(6,1)
        #320x1x1
        self.gan = nn.Sequential(nn.Linear(320, 1),
                                 nn.Sigmoid())
        #1x1x1
    def forward(self,x):
        out = self.conv11(x)
        out = self.conv12(out)

        out = self.conv21(out)
        out = self.conv22(out)
        out = self.conv23(out)

        out = self.conv31(out)
        out = self.conv32(out)
        out = self.conv33(out)

        out = self.conv41(out)
        out = self.conv42(out)
        out = self.conv43(out)

        out = self.conv51(out)
        out = self.conv52(out)
        out = self.conv53(out)
        
        out = self.avgpool(out)
        #BSx320x1x1
        out = out.contiguous()
        out = out.view(out.size(0), -1)
        out = self.gan(out)
        return out



class ConDiscriminator(nn.Module):
    def __init__(self):
        super(ConDiscriminator,self).__init__()
        #3x96x96
        self.conv11 = nn.Sequential(nn.Conv2d(3, 32, kernel_size = 3, stride = 1, padding = 1),
                                    nn.BatchNorm2d(32),
                                    nn.ELU(True))
        #32x96x96
        self.conv12 = nn.Sequential(nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1),
                                    nn.BatchNorm2d(64),
                                    nn.ELU(True))
        #64x96x96
        self.conv21 = nn.Sequential(nn.Conv2d(64, 64, kernel_size = 3, stride = 2, padding = 1),
                                    nn.BatchNorm2d(64),
                                    nn.ELU(True))
        #64x48x48
        self.conv22 = nn.Sequential(nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1),
                                    nn.BatchNorm2d(64),
                                    nn.ELU(True))
        #64x48x48
        self.conv23 = nn.Sequential(nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
                                    nn.BatchNorm2d(128),
                                    nn.ELU(True))
        #128x48x48
        self.conv31 = nn.Sequential(nn.Conv2d(128, 128, kernel_size = 3, stride = 2, padding = 1),
                                    nn.BatchNorm2d(128),
                                    nn.ELU(True))
        #128x24x24
        self.conv32 = nn.Sequential(nn.Conv2d(128, 96, kernel_size = 3, stride = 1, padding = 1),
                                    nn.BatchNorm2d(96),
                                    nn.ELU(True))
        #96x24x24
        self.conv33 = nn.Sequential(nn.Conv2d(96, 192, kernel_size = 3, stride = 1, padding = 1),
                                    nn.BatchNorm2d(192),
                                    nn.ELU(True))
        #192x24x24
        self.conv41 = nn.Sequential(nn.Conv2d(192, 192, kernel_size = 3, stride = 2, padding = 1),
                                    nn.BatchNorm2d(192),
                                    nn.ELU(True))
        #192x12x12
        self.conv42 = nn.Sequential(nn.Conv2d(192, 128, kernel_size = 3, stride = 1, padding = 1),
                                    nn.BatchNorm2d(128),
                                    nn.ELU(True))
        #128x12x12
        self.conv43 = nn.Sequential(nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1),
                                    nn.BatchNorm2d(256),
                                    nn.ELU(True))
        #256x12x12
        self.conv51 = nn.Sequential(nn.Conv2d(256, 256, kernel_size = 3, stride = 2, padding = 1),
                                    nn.BatchNorm2d(256),
                                    nn.ELU(True))
        #256x6x6
        self.conv52 = nn.Sequential(nn.Conv2d(256, 160, kernel_size = 3, stride = 1, padding = 1),
                                    nn.BatchNorm2d(160),
                                    nn.ELU(True))
        #160x6x6
        self.conv53 = nn.Sequential(nn.Conv2d(160, 320, kernel_size = 3, stride = 1, padding = 1),
                                    nn.BatchNorm2d(320),
                                    nn.ELU(True))
        #320x6x6
        #2.conv-filter to 1x1
        self.conv_final = nn.Sequential(nn.Conv2d(320, 1, kernel_size = 6, stride = 1, padding = 0),
                                        nn.BatchNorm2d(1),
                                        nn.ELU(True))
        #1x1x1
        self.sig = nn.Sigmoid()
    def forward(self,x):
        out = self.conv11(x)
        out = self.conv12(out)

        out = self.conv21(out)
        out = self.conv22(out)
        out = self.conv23(out)

        out = self.conv31(out)
        out = self.conv32(out)
        out = self.conv33(out)

        out = self.conv41(out)
        out = self.conv42(out)
        out = self.conv43(out)

        out = self.conv51(out)
        out = self.conv52(out)
        out = self.conv53(out)

        out = self.conv_final(out)
        out = out.contiguous()
        out = out.view(out.size(0), -1)
        # 1
        out = self.sig(out)
        return out


class SuperResolutionDiscriminator(nn.Module):
    def __init__(self):
        super(SuperResolutionDiscriminator,self).__init__()
        #3x96x96
        self.conv11 = nn.Sequential(nn.Conv2d(3, 64, kernel_size = 3, stride = 1,padding = 1),
                                    nn.LeakyReLU(0.2))
        #64x96x96
        self.conv12 = nn.Sequential(nn.Conv2d(64, 64, kernel_size = 3, stride = 2,padding = 1),
                                    nn.BatchNorm2d(64),
                                    nn.LeakyReLU(0.2))
        #64x48x48

        self.conv21 = nn.Sequential(nn.Conv2d(64, 128, kernel_size =3, stride = 1, padding = 1),
                                    nn.BatchNorm2d(128),
                                    nn.LeakyReLU(0.2))
        #128x48x48
        self.conv22 = nn.Sequential(nn.Conv2d(128, 128, kernel_size = 3, stride = 2, padding = 1),
                                    nn.BatchNorm2d(128),
                                    nn.LeakyReLU(0.2))
        #128x24x24

        self.conv31 = nn.Sequential(nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1),
                                    nn.BatchNorm2d(256),
                                    nn.LeakyReLU(0.2))
        #256x24x24
        self.conv32 = nn.Sequential(nn.Conv2d(256, 256, kernel_size = 3, stride = 2, padding = 1),
                                    nn.BatchNorm2d(256),
                                    nn.LeakyReLU(0.2))
        #256x12x12

        self.conv41 = nn.Sequential(nn.Conv2d(256, 512, kernel_size = 3, stride = 1, padding = 1),
                                    nn.BatchNorm2d(512),
                                    nn.LeakyReLU(0.2))
        #512x12x12
        self.conv42 = nn.Sequential(nn.Conv2d(512, 512, kernel_size = 3, stride = 2, padding = 1),
                                    nn.BatchNorm2d(512),
                                    nn.LeakyReLU(0.2))
        #512x6x6
        self.conv_final = nn.Sequential(nn.Conv2d(512, 512, kernel_size = 6, stride = 1, padding = 0),
                                        nn.BatchNorm2d(512),
                                        nn.LeakyReLU(0.2))
        #512x1x1
        self.fc = nn.Sequential(nn.Linear(512, 1024),
                                nn.LeakyReLU(0.2),
                                nn.Linear(1024, 1))
                                # nn.Sigmoid())
    def forward(self, x):
        out = self.conv11(x)
        out = self.conv12(out)

        out = self.conv21(out)
        out = self.conv22(out)

        out = self.conv31(out)
        out = self.conv32(out)

        out = self.conv41(out)
        out = self.conv42(out)

        out = self.conv_final(out)
        #512x1x1
        out = out.contiguous()
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

