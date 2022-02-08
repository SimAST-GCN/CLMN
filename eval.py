import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def simcse_loss(y_pred):
    """用于SimCSE训练的loss
    """
    # 构造标签
    idxs = torch.arange(0, y_pred.shape[0])
    idxs_1 = idxs[None, :]
    idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None]
    y_true = np.equal(idxs_1, idxs_2)
    y_true = y_true.float()
    # 计算相似度
    y_pred = F.normalize(y_pred, p=2, dim=1)
    similarities = torch.mm(y_pred,torch.transpose(y_pred,0,1))
    similarities = similarities - torch.eye(y_pred.shape[0])*1e12
    similarities = similarities * 20
    cl = torch.nn.CrossEntropyLoss()
    loss = cl(similarities,y_true)
    return loss

class CLloss(nn.Module):
    def __init__(self,device):
        super(CLloss, self).__init__()
        self.cl = torch.nn.CrossEntropyLoss()
        self.device = device
    
    def forward(self,y_pred):
        # 构造标签
        idxs = torch.arange(0, y_pred.shape[0]).to(self.device)
        idxs_1 = idxs[None, :]
        idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None]
        y_true = torch.eq(idxs_1, idxs_2)
        y_true = y_true.float()
        # 计算相似度
        y_pred = F.normalize(y_pred, p=2, dim=1)
        similarities = torch.mm(y_pred,torch.transpose(y_pred,0,1))
        similarities = similarities - torch.eye(y_pred.shape[0]).to(self.device)*1e12
        similarities = similarities * 20
        loss = self.cl(similarities,y_true)
        return loss