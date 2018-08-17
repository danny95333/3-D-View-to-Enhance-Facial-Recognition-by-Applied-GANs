import torch.nn as nn
import torch

class localDiscriminator(nn.Module):
    def __init__(self,nc,ndf,nf,nd,np,ni,ms):
        super(localDiscriminator,self).__init__()
        # 3 x 96 x 96
        self.conv11 = nn.Sequential(nn.Conv2d(nc,ndf,kernel_size=3,stride=1,padding=1),
                                   nn.BatchNorm2d(ndf),
                                   nn.ELU(True))
        # ndf x 96 x 96
        self.conv12 = nn.Sequential(nn.Conv2d(ndf,ndf*2,kernel_size=3,stride=1,padding=1),
                                   nn.BatchNorm2d(ndf*2),
                                   nn.ELU(True))

        # (ndfx2) x 96 x 96
        self.conv21 = nn.Sequential(nn.Conv2d(ndf*2,ndf*2,kernel_size=3,stride=2,padding=1),
                                   nn.BatchNorm2d(ndf*2),
                                   nn.ELU(True))
        # (ndfx2) x 48 x 48
        self.conv22 = nn.Sequential(nn.Conv2d(ndf*2,ndf*2,kernel_size=3,stride=1,padding=1),
                                   nn.BatchNorm2d(ndf*2),
                                   nn.ELU(True))
        # (ndfx2) x 48 x 48
        self.conv23 = nn.Sequential(nn.Conv2d(ndf*2,ndf*4,kernel_size=3,stride=1,padding=1),
                                   nn.BatchNorm2d(ndf*4),
                                   nn.ELU(True))

        # (ndfx4) x 48 x 48
        self.conv31 = nn.Sequential(nn.Conv2d(ndf*4,ndf*4,kernel_size=3,stride=2,padding=1),
                                   nn.BatchNorm2d(ndf*4),
                                   nn.ELU(True))
        # (ndfx4) x 24 x 24
        self.conv32 = nn.Sequential(nn.Conv2d(ndf*4,ndf*3,kernel_size=3,stride=1,padding=1),
                                   nn.BatchNorm2d(ndf*3),
                                   nn.ELU(True))
        # (ndfx3) x 24 x 24
        self.conv33 = nn.Sequential(nn.Conv2d(ndf*3,ndf*6,kernel_size=3,stride=1,padding=1),
                                   nn.BatchNorm2d(ndf*6),
                                   nn.ELU(True))

        # (ndfx6) x 24 x 24
        self.conv41 = nn.Sequential(nn.Conv2d(ndf*6,ndf*6,kernel_size=3,stride=2,padding=1),
                                   nn.BatchNorm2d(ndf*6),
                                   nn.ELU(True))
        # (ndfx6) x 12 x 12
        self.conv42 = nn.Sequential(nn.Conv2d(ndf*6,ndf*4,kernel_size=3,stride=1,padding=1),
                                   nn.BatchNorm2d(ndf*4),
                                   nn.ELU(True))
        # (ndfx4) x 12 x 12
        self.conv43 = nn.Sequential(nn.Conv2d(ndf*4,ndf*8,kernel_size=3,stride=1,padding=1),
                                   nn.BatchNorm2d(ndf*8),
                                   nn.ELU(True))
        # (ndfx8) x 12 x 12
        self.conv51 = nn.Sequential(nn.Conv2d(ndf*8,ndf*8,kernel_size=3,stride=2,padding=1),
                                   nn.BatchNorm2d(ndf*8),
                                   nn.ELU(True))
        # (ndfx8) x 6 x 6
        self.conv52 = nn.Sequential(nn.Conv2d(ndf*8,ndf*5,kernel_size=3,stride=1,padding=1),
                                   nn.BatchNorm2d(ndf*5),
                                   nn.ELU(True))
        # (ndfx5) x 12 x 12
        self.conv53 = nn.Sequential(nn.Conv2d(ndf*5,nf+1,kernel_size=3,stride=1,padding=1),
                                   nn.BatchNorm2d(nf+1),
                                   nn.ELU(True))
        # (nf+1) x 6 x 6
        if(ms==32):
          self.avgpool = nn.AvgPool2d(2,1)
        elif(ms==48):
          self.avgpool = nn.AvgPool2d(3,1)
        elif(ms==64):
          self.avgpool = nn.AvgPool2d(4,1)
        else:
          self.avgpool = nn.AvgPool2d(6,1)

        # (nf+1) x 1 x 1
        self.fc_gan = nn.Sequential(nn.Linear(nf+1, 1), # real/fake classification
                                    nn.Sigmoid())

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
        #print(out)
        out = self.avgpool(out)
        # 16 x 321 x 1 x 1
        out = out.view(out.size(0),-1)

        gan_branch = self.fc_gan(out)
        return gan_branch


class Discriminator(nn.Module):
    def __init__(self,nc,ndf,nf,nd,np,ni):
        super(Discriminator,self).__init__()
        # 3 x 96 x 96
        self.conv11 = nn.Sequential(nn.Conv2d(nc,ndf,kernel_size=3,stride=1,padding=1),
                                   nn.BatchNorm2d(ndf),
                                   nn.ELU(True))
        # ndf x 96 x 96
        self.conv12 = nn.Sequential(nn.Conv2d(ndf,ndf*2,kernel_size=3,stride=1,padding=1),
                                   nn.BatchNorm2d(ndf*2),
                                   nn.ELU(True))

        # (ndfx2) x 96 x 96
        self.conv21 = nn.Sequential(nn.Conv2d(ndf*2,ndf*2,kernel_size=3,stride=2,padding=1),
                                   nn.BatchNorm2d(ndf*2),
                                   nn.ELU(True))
        # (ndfx2) x 48 x 48
        self.conv22 = nn.Sequential(nn.Conv2d(ndf*2,ndf*2,kernel_size=3,stride=1,padding=1),
                                   nn.BatchNorm2d(ndf*2),
                                   nn.ELU(True))
        # (ndfx2) x 48 x 48
        self.conv23 = nn.Sequential(nn.Conv2d(ndf*2,ndf*4,kernel_size=3,stride=1,padding=1),
                                   nn.BatchNorm2d(ndf*4),
                                   nn.ELU(True))

        # (ndfx4) x 48 x 48
        self.conv31 = nn.Sequential(nn.Conv2d(ndf*4,ndf*4,kernel_size=3,stride=2,padding=1),
                                   nn.BatchNorm2d(ndf*4),
                                   nn.ELU(True))
        # (ndfx4) x 24 x 24
        self.conv32 = nn.Sequential(nn.Conv2d(ndf*4,ndf*3,kernel_size=3,stride=1,padding=1),
                                   nn.BatchNorm2d(ndf*3),
                                   nn.ELU(True))
        # (ndfx3) x 24 x 24
        self.conv33 = nn.Sequential(nn.Conv2d(ndf*3,ndf*6,kernel_size=3,stride=1,padding=1),
                                   nn.BatchNorm2d(ndf*6),
                                   nn.ELU(True))

        # (ndfx6) x 24 x 24
        self.conv41 = nn.Sequential(nn.Conv2d(ndf*6,ndf*6,kernel_size=3,stride=2,padding=1),
                                   nn.BatchNorm2d(ndf*6),
                                   nn.ELU(True))
        # (ndfx6) x 12 x 12
        self.conv42 = nn.Sequential(nn.Conv2d(ndf*6,ndf*4,kernel_size=3,stride=1,padding=1),
                                   nn.BatchNorm2d(ndf*4),
                                   nn.ELU(True))
        # (ndfx4) x 12 x 12
        self.conv43 = nn.Sequential(nn.Conv2d(ndf*4,ndf*8,kernel_size=3,stride=1,padding=1),
                                   nn.BatchNorm2d(ndf*8),
                                   nn.ELU(True))
        # (ndfx8) x 12 x 12
        self.conv51 = nn.Sequential(nn.Conv2d(ndf*8,ndf*8,kernel_size=3,stride=2,padding=1),
                                   nn.BatchNorm2d(ndf*8),
                                   nn.ELU(True))
        # (ndfx8) x 6 x 6
        self.conv52 = nn.Sequential(nn.Conv2d(ndf*8,ndf*5,kernel_size=3,stride=1,padding=1),
                                   nn.BatchNorm2d(ndf*5),
                                   nn.ELU(True))
        # (ndfx5) x 12 x 12
        self.conv53 = nn.Sequential(nn.Conv2d(ndf*5,nf+1,kernel_size=3,stride=1,padding=1),
                                   nn.BatchNorm2d(nf+1),
                                   nn.ELU(True))
        # (nf+1) x 6 x 6 nf = 320 hidden size
        self.avgpool = nn.AvgPool2d(6,1)
        # (nf+1) x 1 x 1
        self.fc_id = nn.Linear(nf+1, nd) # identity classification
        self.fc_pose = nn.Linear(nf+1, np) # pose classification
        self.fc_light = nn.Linear(nf+1, ni) # lightclassification
        self.fc_gan = nn.Sequential(nn.Linear(nf+1, 1), # real/fake classification
                                    nn.Sigmoid())

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
        # 16 x 321 x 1 x 1
        out = out.view(out.size(0),-1)

        id_branch = self.fc_id(out)
        pose_branch = self.fc_pose(out)
        light_branch = self.fc_light(out)
        gan_branch = self.fc_gan(out)
        return id_branch, pose_branch, gan_branch, light_branch

class FaceReco(nn.Module):
    def __init__(self,nc,ndf,nf,nd):
        super(FaceReco,self).__init__()
        # 3 x 96 x 96
        self.conv11 = nn.Sequential(nn.Conv2d(nc,ndf,kernel_size=3,stride=1,padding=1),
                                   nn.BatchNorm2d(ndf),
                                   nn.ELU(True))
        # ndf x 96 x 96
        self.conv12 = nn.Sequential(nn.Conv2d(ndf,ndf*2,kernel_size=3,stride=1,padding=1),
                                   nn.BatchNorm2d(ndf*2),
                                   nn.ELU(True))

        # (ndfx2) x 96 x 96
        self.conv21 = nn.Sequential(nn.Conv2d(ndf*2,ndf*2,kernel_size=3,stride=2,padding=1),
                                   nn.BatchNorm2d(ndf*2),
                                   nn.ELU(True))
        # (ndfx2) x 48 x 48
        self.conv22 = nn.Sequential(nn.Conv2d(ndf*2,ndf*2,kernel_size=3,stride=1,padding=1),
                                   nn.BatchNorm2d(ndf*2),
                                   nn.ELU(True))
        # (ndfx2) x 48 x 48
        self.conv23 = nn.Sequential(nn.Conv2d(ndf*2,ndf*4,kernel_size=3,stride=1,padding=1),
                                   nn.BatchNorm2d(ndf*4),
                                   nn.ELU(True))

        # (ndfx4) x 48 x 48
        self.conv31 = nn.Sequential(nn.Conv2d(ndf*4,ndf*4,kernel_size=3,stride=2,padding=1),
                                   nn.BatchNorm2d(ndf*4),
                                   nn.ELU(True))
        # (ndfx4) x 24 x 24
        self.conv32 = nn.Sequential(nn.Conv2d(ndf*4,ndf*3,kernel_size=3,stride=1,padding=1),
                                   nn.BatchNorm2d(ndf*3),
                                   nn.ELU(True))
        # (ndfx3) x 24 x 24
        self.conv33 = nn.Sequential(nn.Conv2d(ndf*3,ndf*6,kernel_size=3,stride=1,padding=1),
                                   nn.BatchNorm2d(ndf*6),
                                   nn.ELU(True))

        # (ndfx6) x 24 x 24
        self.conv41 = nn.Sequential(nn.Conv2d(ndf*6,ndf*6,kernel_size=3,stride=2,padding=1),
                                   nn.BatchNorm2d(ndf*6),
                                   nn.ELU(True))
        # (ndfx6) x 12 x 12
        self.conv42 = nn.Sequential(nn.Conv2d(ndf*6,ndf*4,kernel_size=3,stride=1,padding=1),
                                   nn.BatchNorm2d(ndf*4),
                                   nn.ELU(True))
        # (ndfx4) x 12 x 12
        self.conv43 = nn.Sequential(nn.Conv2d(ndf*4,ndf*8,kernel_size=3,stride=1,padding=1),
                                   nn.BatchNorm2d(ndf*8),
                                   nn.ELU(True))
        # (ndfx8) x 12 x 12
        self.conv51 = nn.Sequential(nn.Conv2d(ndf*8,ndf*8,kernel_size=3,stride=2,padding=1),
                                   nn.BatchNorm2d(ndf*8),
                                   nn.ELU(True))
        # (ndfx8) x 6 x 6
        self.conv52 = nn.Sequential(nn.Conv2d(ndf*8,ndf*5,kernel_size=3,stride=1,padding=1),
                                   nn.BatchNorm2d(ndf*5),
                                   nn.ELU(True))
        # (ndfx5) x 12 x 12
        self.conv53 = nn.Sequential(nn.Conv2d(ndf*5,nf+1,kernel_size=3,stride=1,padding=1),
                                   nn.BatchNorm2d(nf+1),
                                   nn.ELU(True))
        # (nf+1) x 6 x 6 nf = 320 hidden size
        self.avgpool = nn.AvgPool2d(6,1)
        # (nf+1) x 1 x 1
        self.fc_id = nn.Linear(nf+1, nd) # identity classification

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
        # 16 x 321 x 1 x 1
        out = out.view(out.size(0),-1)

        id_branch = self.fc_id(out)
        return id_branch

class Generator(nn.Module):
    def __init__(self,nc,nf,ngf,nz,np,ni):
        super(Generator,self).__init__()
        # 3 x 96 x 96
        self.conv11 = nn.Sequential(nn.Conv2d(nc,ngf,kernel_size=3,stride=1,padding=1),
                                   nn.BatchNorm2d(ngf),
                                   nn.ELU(True))
        # ngf x 96 x 96
        self.conv12 = nn.Sequential(nn.Conv2d(ngf,ngf*2,kernel_size=3,stride=1,padding=1),
                                   nn.BatchNorm2d(ngf*2),
                                   nn.ELU(True))

        # (ngfx2) x 96 x 96
        self.conv21 = nn.Sequential(nn.Conv2d(ngf*2,ngf*2,kernel_size=3,stride=2,padding=1),
                                   nn.BatchNorm2d(ngf*2),
                                   nn.ELU(True))
        # (ngfx2) x 48 x 48
        self.conv22 = nn.Sequential(nn.Conv2d(ngf*2,ngf*2,kernel_size=3,stride=1,padding=1),
                                   nn.BatchNorm2d(ngf*2),
                                   nn.ELU(True))
        # (ngfx2) x 48 x 48
        self.conv23 = nn.Sequential(nn.Conv2d(ngf*2,ngf*4,kernel_size=3,stride=1,padding=1),
                                   nn.BatchNorm2d(ngf*4),
                                   nn.ELU(True))

        # (ngfx4) x 48 x 48
        self.conv31 = nn.Sequential(nn.Conv2d(ngf*4,ngf*4,kernel_size=3,stride=2,padding=1),
                                   nn.BatchNorm2d(ngf*4),
                                   nn.ELU(True))
        # (ngfx4) x 24 x 24
        self.conv32 = nn.Sequential(nn.Conv2d(ngf*4,ngf*3,kernel_size=3,stride=1,padding=1),
                                   nn.BatchNorm2d(ngf*3),
                                   nn.ELU(True))
        # (ngfx3) x 24 x 24
        self.conv33 = nn.Sequential(nn.Conv2d(ngf*3,ngf*6,kernel_size=3,stride=1,padding=1),
                                   nn.BatchNorm2d(ngf*6),
                                   nn.ELU(True))

        # (ngfx6) x 24 x 24
        self.conv41 = nn.Sequential(nn.Conv2d(ngf*6,ngf*6,kernel_size=3,stride=2,padding=1),
                                   nn.BatchNorm2d(ngf*6),
                                   nn.ELU(True))
        # (ngfx6) x 12 x 12
        self.conv42 = nn.Sequential(nn.Conv2d(ngf*6,ngf*4,kernel_size=3,stride=1,padding=1),
                                   nn.BatchNorm2d(ngf*4),
                                   nn.ELU(True))
        # (ngfx4) x 12 x 12
        self.conv43 = nn.Sequential(nn.Conv2d(ngf*4,ngf*8,kernel_size=3,stride=1,padding=1),
                                   nn.BatchNorm2d(ngf*8),
                                   nn.ELU(True))
        # (ngfx8) x 12 x 12
        self.conv51 = nn.Sequential(nn.Conv2d(ngf*8,ngf*8,kernel_size=3,stride=2,padding=1),
                                   nn.BatchNorm2d(ngf*8),
                                   nn.ELU(True))
        # (ngfx8) x 6 x 6
        self.conv52 = nn.Sequential(nn.Conv2d(ngf*8,ngf*5,kernel_size=3,stride=1,padding=1),
                                   nn.BatchNorm2d(ngf*5),
                                   nn.ELU(True))
        # (ngfx5) x 12 x 12
        self.conv53 = nn.Sequential(nn.Conv2d(ngf*5,nf+1,kernel_size=3,stride=1,padding=1),
                                   nn.BatchNorm2d(nf+1),
                                   nn.ELU(True))
        # (nf+1) x 6 x 6
        self.avgpool = nn.AvgPool2d(6,1)
        # (nf+1) x 6 x 6

        self.fc = nn.Linear((nf+1+np+nz+ni),320*6*6)
        # 320 x 6 x 6

        self.fconv52 = nn.Sequential(nn.ConvTranspose2d(320, ngf*5, 3, stride=1, padding=1),
                                     nn.BatchNorm2d(ngf*5),
                                     nn.ELU(True))
        # (ngfx5) x 6 x 6
        self.fconv51 = nn.Sequential(nn.ConvTranspose2d(ngf*5, ngf*8, 3, stride=1, padding=1),
                                     nn.BatchNorm2d(ngf*8),
                                     nn.ELU(True))

        # (ngfx8) x 6 x 6
        self.fconv43 = nn.Sequential(nn.ConvTranspose2d(ngf*8, ngf*8, 3, stride=2, padding=1, output_padding=1),
                                     nn.BatchNorm2d(ngf*8),
                                     nn.ELU(True))
        # (ngfx8) x 12 x 12
        self.fconv42 = nn.Sequential(nn.ConvTranspose2d(ngf*8, ngf*4, 3, stride=1, padding=1),
                                     nn.BatchNorm2d(ngf*4),
                                     nn.ELU(True))
        # (ngfx4) x 12 x 12
        self.fconv41 = nn.Sequential(nn.ConvTranspose2d(ngf*4, ngf*6, 3, stride=1, padding=1),
                                     nn.BatchNorm2d(ngf*6),
                                     nn.ELU(True))
        # (ngfx6) x 12 x 12
        self.fconv33 = nn.Sequential(nn.ConvTranspose2d(ngf*6, ngf*6, 3, stride=2, padding=1, output_padding=1),
                                     nn.BatchNorm2d(ngf*6),
                                     nn.ELU(True))
        # (ngfx6) x 24 x 24
        self.fconv32 = nn.Sequential(nn.ConvTranspose2d(ngf*6, ngf*3, 3, stride=1, padding=1),
                                     nn.BatchNorm2d(ngf*3),
                                     nn.ELU(True))
        # (ngfx3) x 24 x 24
        self.fconv31 = nn.Sequential(nn.ConvTranspose2d(ngf*3, ngf*4, 3, stride=1, padding=1),
                                     nn.BatchNorm2d(ngf*4),
                                     nn.ELU(True))
        # (ngfx4) x 24 x 24
        self.fconv23 = nn.Sequential(nn.ConvTranspose2d(ngf*4, ngf*4, 3, stride=2, padding=1, output_padding=1),
                                     nn.BatchNorm2d(ngf*4),
                                     nn.ELU(True))
        # (ngfx4) x 48 x 48
        self.fconv22 = nn.Sequential(nn.ConvTranspose2d(ngf*4, ngf*2, 3, stride=1, padding=1),
                                     nn.BatchNorm2d(ngf*2),
                                     nn.ELU(True))
        # (ngfx2) x 48 x 48
        self.fconv21 = nn.Sequential(nn.ConvTranspose2d(ngf*2, ngf*2, 3, stride=1, padding=1),
                                     nn.BatchNorm2d(ngf*2),
                                     nn.ELU(True))
        # (ngfx2) x 48 x 48
        self.fconv13 = nn.Sequential(nn.ConvTranspose2d(ngf*2, ngf*2, 3, stride=2, padding=1, output_padding=1),
                                     nn.BatchNorm2d(ngf*2),
                                     nn.ELU(True))
        # (ngfx2) x 96 x 96
        self.fconv12 = nn.Sequential(nn.ConvTranspose2d(ngf*2, ngf, 3, stride=1, padding=1),
                                     nn.BatchNorm2d(ngf),
                                     nn.ELU(True))
        # (ngf) x 96 x 96
        self.fconv11 = nn.Sequential(nn.ConvTranspose2d(ngf, nc, 3, stride=1, padding=1),
                                     nn.Tanh())

    def forward(self,x,z,p,i):
        # z is for noise and p is for pose code
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
        out = out.view(out.size(0),-1)

        # concatenate noise, identity feature and pose code
        out = torch.cat((out,z,p,i),1)


        out = self.fc(out)
        out = out.view(out.size(0),320,6,6)
        out = self.fconv52(out)
        out = self.fconv51(out)

        out = self.fconv43(out)
        out = self.fconv42(out)
        out = self.fconv41(out)

        out = self.fconv33(out)
        out = self.fconv32(out)
        out = self.fconv31(out)

        out = self.fconv23(out)
        out = self.fconv22(out)
        out = self.fconv21(out)

        out = self.fconv13(out)
        out = self.fconv12(out)
        out = self.fconv11(out)

        return out
