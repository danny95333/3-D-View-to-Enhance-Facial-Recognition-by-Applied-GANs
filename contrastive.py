import torch
import torch.nn as nn
import torch.nn.functional as F




class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss Function
    Based on Ting's paper

    """
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, img_1, img_2, label_1):
        euclidean_distance = F.pairwise_distance(img_1, img_2)
        # eu_dis no problem with two variable

        # no problem with computation seprately
        # problem is part_1 is a tensor can't  multiply by the variable

        loss_contrastive = torch.mean((label_1)*torch.pow(euclidean_distance, 2) +(1-label_1) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        # use torch.mean to make 64Lx64L tensor to be 1L x 1L
        # print(loss_contrastive.data.size())
        return loss_contrastive