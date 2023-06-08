import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from DataLoader import BatchSize, device
import math

# 计算一维的高斯分布向量
def gaussian(window_size, sigma):
    gauss = torch.Tensor([torch.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in torch.tensor(range(window_size))])
    return gauss/gauss.sum()
 
 
# 创建高斯核，通过两个一维高斯分布向量进行矩阵乘法得到
# 可以设定channel参数拓展为3通道
def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window
 
# 计算SSIM
# 直接使用SSIM的公式，但是在计算均值时，不是直接求像素平均值，而是采用归一化的高斯核卷积来代替。
# 在计算方差和协方差时用到了公式Var(X)=E[X^2]-E[X]^2, cov(X,Y)=E[XY]-E[X]E[Y].
# 正如前面提到的，上面求期望的操作采用高斯核卷积代替。
def loss_ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    '''
    L is the dynamic range of the pixel values.
    '''
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1
 
        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range
 
    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)
 
    img1 = img1.to(device) 
    img2 = img2.to(device) 
    window = window.to(device) 
    
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)
 
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
 
    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2
 
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
 
    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity
 
    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)
 
    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)
 
    if full:
        return ret, cs
    return ret
 
 
# Classes to re-use window
class LOSS_SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(LOSS_SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range
 
        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)
 
    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
 
        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel
 
        return 1-loss_ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)

    



def loss_psnr(Pred_nmlzd, GT_nmlzd):  #预测值与真值（Predict and ground truth)
    '''
    The Pred and GT should be in the range of [0, 1] which lets
    the calculation of PSNR make sense in image processing field.
    '''
    assert Pred_nmlzd.shape == torch.Size([BatchSize, 1, 100, 100]), "The shape of Pred_nmlzd is not (BatchSize, 1, 100, 100)"
    assert GT_nmlzd.shape == torch.Size([BatchSize, 1, 100, 100]), "The shape of GT_nmlzd is not (BatchSize, 1, 100, 100)"        #验证数据集尺寸
    assert (Pred_nmlzd>=0.0).all() and (Pred_nmlzd<=1.0).all(), "The Pred_nmlzd amplitude value is out of range [0, 1]"
    assert (GT_nmlzd>=0.0).all() and (GT_nmlzd<=1.0).all(), "The GT_nmlzd amplitude value is out of range [0, 1]"                #验证数据集大小

    mse = torch.mean((Pred_nmlzd - GT_nmlzd)**2, dim=(2,3))   # [BatchSize, 1]     求均方误差 dim=(2,3)即对2，3维作为特征计算均值
    assert mse.shape == torch.Size([BatchSize, 1]), "mse is not for each image in a batch"
    assert (mse>=0.0).all() and (mse<=1.0).all(), "The calculated mse is out of range [0, 1]"
    PIXEL_MAX = 1.0              # [BatchSize, 1]
    
    psnr_batch = torch.where(mse != 0, 20*torch.log10(PIXEL_MAX/torch.sqrt(mse)), torch.ones_like(mse)*100)  # [BatchSize, 1]
    psnr = torch.mean(psnr_batch)
    # Check whether it is right for the calculation of psnr for each image in a batch
    psnr_mean = (torch.sum(((mse[mse == 0] + 1)*100)) + torch.sum(20 * torch.log10(PIXEL_MAX / torch.sqrt(mse[mse != 0])))) / BatchSize  #?
    # print(psnr_mean, psnr)
    # print(mse.shape)
    # print((torch.sqrt(mse[mse != 0])).shape)
    # print((20 * torch.log10(PIXEL_MAX / torch.sqrt(mse[mse != 0]))).shape)
    # print(BatchSize)
    # exit()
    assert torch.abs(psnr_mean - psnr) <= 1e-5, "There is something wrong for calculation of psnr"
    return psnr

class LOSS_PSNR(nn.Module):
    def __init__(self):   #这是一个特殊的函数，它的作用主要是事先把一些重要的属性填写进来，它的特点是第一个参数永远是self，表示创建的实例本身
        super(LOSS_PSNR, self).__init__()
    def forward(self, Pred_nmlzd, GT_nmlzd):
        return loss_psnr(Pred_nmlzd, GT_nmlzd)       

# def loss_BCE(Pred,GT):
#      '''
#     This function is used to fit the binary output
#     '''
#     loss_BCE = torch.nn.BCEWithLogitsLoss




def loss_cos_mse(Pred, GT):
    '''
    This function is used to penalize neural network
    Considering the net output phase is imprinted with periodic nature
    The cosine function can eliminate the influence of periodic nature
    '''
    Diff = torch.abs(GT - Pred)
    Diff_cos = torch.cos(Diff)
    Imatrix = torch.ones_like(Diff_cos).to(device)
    loss_cos_mse = F.mse_loss(Imatrix, Diff_cos)
    return loss_cos_mse

class LOSS_COS_MSE(nn.Module):
    def __init__(self):
        super(LOSS_COS_MSE, self).__init__()
    def forward(self, Pred, GT):
        return loss_cos_mse(Pred, GT)


def loss_cos_mae(Pred, GT):
    Diff = torch.abs(GT - Pred)
    Diff_cos = torch.cos(Diff)
    Imatrix = torch.ones_like(Diff_cos).to(device)
    loss_cos_mae = F.l1_loss(Imatrix, Diff_cos)
    return loss_cos_mae

class LOSS_COS_MAE(nn.Module):
    def __init__(self):
        super(LOSS_COS_MAE, self).__init__()
    def forward(self, Pred, GT):
        return loss_cos_mae(Pred, GT)


def piecewise_phs_diff(Pred, GT):
    # Pred: torch.size([BatchSize, 1, 100, 100]), in the range of [0, 2pi]
    # GT: torch.size([BatchSize, 1, 100, 100]), in the range of [0, 2pi]
    assert Pred.shape == torch.Size([BatchSize, 1, 100, 100]) and GT.shape == torch.Size([BatchSize, 1, 100, 100]), "Shapes of Pred and/or GT are wrong"
    abs_diff = torch.abs(Pred - GT)  # torch.size([BatchSize, 1, 100, 100]), , in the range of [0, 2pi]
    assert abs_diff.max() <= 2*torch.pi and abs_diff.min() >= 0.0, "abs_diff of Pred and GT is out of range [0, 2pi]"
    x = abs_diff/torch.pi*2
    y = (2*torch.pi-abs_diff)/torch.pi*2
    piecewise_loss = torch.where((abs_diff<=torch.pi), x, y)
    return piecewise_loss.mean()

class PIECEWISE_MAE(nn.Module):
    def __init__(self):
        super(PIECEWISE_MAE, self).__init__()
    def forward(self, Pred, GT):
        return piecewise_phs_diff(Pred, GT)


def loss_fg_bg_mse_mae(Pred, GT_scaleup, GT_nmlzd, error_name=None):
    assert (GT_nmlzd >= 0.0).all() and (GT_nmlzd <= 1.0).all(), "The GT_nmlzd is out of range [0, 1]"
    fg_mask = torch.where(GT_nmlzd > 0.5, 1.0, 0.0)
    fg_pixel_num = torch.sum(fg_mask, dim=(2,3))     # (BatchSize, 1)
    bg_pixel_num = ((torch.ones_like(fg_pixel_num)).to(device)) * 10000 - fg_pixel_num
    # print(fg_pixel_num)
    # print(bg_pixel_num)
    # exit()
    assert (fg_pixel_num > 0).all() and (bg_pixel_num > 0).all() and ((fg_pixel_num + bg_pixel_num) == 10000).all(), "The pixel amount of the foreground and the background is wrong calculated"
    if error_name == 'mse':
        diff = (GT_scaleup - Pred)**2                       # (BatchSize, 1, 100, 100)
    elif error_name == 'mae':
        diff = torch.abs(GT_scaleup - Pred)                 # (BatchSize, 1, 100, 100)
    fg_diff = diff * fg_mask
    bg_diff = diff - diff * fg_mask
    # fg_diff_mean = torch.sum(fg_diff, dim=(2,3)) / fg_pixel_num   # Mean square error
    # bg_diff_mean = torch.sum(bg_diff, dim=(2,3)) / bg_pixel_num   # Size of (BatchSize, 1)
    fg_diff_mean = torch.sqrt(torch.sum(fg_diff, dim=(2,3)) / fg_pixel_num)   # Root Mean square error
    bg_diff_mean = torch.sqrt(torch.sum(bg_diff, dim=(2,3)) / bg_pixel_num)   # Size of (BatchSize, 1)
    # print("diff.shape is \n {}".format(diff.shape))
    # print("fg_diff.shape is \n {}".format(fg_diff.shape))
    # print("bg_diff.shape is \n {}".format(bg_diff.shape))
    # print("fg_diff_mean is \n {}".format(fg_diff_mean))
    # print("bg_diff_mean is \n {}".format(bg_diff_mean))
    # print("fg_diff_mean of a batch is \n {}".format(fg_diff_mean.mean()))
    # print("bg_diff_mean of a batch is \n {}".format(bg_diff_mean.mean()))
    # exit()
    return fg_diff_mean.mean(), bg_diff_mean.mean()

class LOSS_FG_BG_MSE_MAE(nn.Module):
    def __init__(self, error_name=None):
        super(LOSS_FG_BG_MSE_MAE, self).__init__()
        self.error_name = error_name
    def forward(self, Pred, GT_scaleup, GT_nmlzd):
        return loss_fg_bg_mse_mae(Pred, GT_scaleup, GT_nmlzd, error_name = self.error_name)


def Ac_intensity_proportion(Pred, GT_nmlzd):
    assert (GT_nmlzd >= 0.0).all() and (GT_nmlzd <= 1.0).all(), "The GT_nmlzd is out of range [0, 1]"
    FG_mask = torch.where(GT_nmlzd > 0.5, 1.0, 0.0)
    BG_mask = torch.where(GT_nmlzd < 0.5, 1.0, 0.0)
    assert (FG_mask + BG_mask == 1.0).all(), "The FG_mask and/or BG_mask is wrong"
    FG_sum = torch.sum(Pred**2 * FG_mask, dim=(2,3))
    BG_sum = torch.sum(Pred**2 * BG_mask, dim=(2,3))
    Total = torch.sum(Pred**2, dim=(2,3))
    assert (torch.abs((FG_sum + BG_sum) - Total) < 1e-2).all(), "The FG_sum and/or BG_mask is wrong"
    assert ((FG_sum / Total) >= 0.0).all() and ((FG_sum / Total) <= 1.0).all(), "The intensity proportion of the Ac foreground is out of range [0, 1]"
    assert ((BG_sum / Total) >= 0.0).all() and ((BG_sum / Total) <= 1.0).all(), "The intensity proportion of the Ac background is out of range [0, 1]"
    return (FG_sum / Total).mean(), (BG_sum / Total).mean()

class LOSS_INTENSITY(nn.Module):
    def __init__(self):
        super(LOSS_INTENSITY, self).__init__()
    def forward(self, Pred, GT_nmlzd):
        return Ac_intensity_proportion(Pred, GT_nmlzd)


def get_max_dist(Pred, GT):
    Pred_max = torch.max(torch.max(Pred, -1)[0], -1)[0] # [BatchSize, 1]
    GT_max = torch.max(torch.max(GT, -1)[0], -1)[0]     # [BatchSize, 1]
    dist = torch.abs(Pred_max - GT_max)                 # [BatchSize, 1]
    # print("Pred_max.mean() is {}".format(Pred_max.mean()))
    # print("GT_max.mean() is {}".format(GT_max.mean()))
    # print("dist.mean() is {}".format(dist.mean()))
    # print("dist.max() is {}".format(dist.max()))
    # return dist.mean()
    return dist.max()

class MAX_DIST(nn.Module):
    def __init__(self):
        super(MAX_DIST, self).__init__()
    def forward(self, Pred, GT):
        return get_max_dist(Pred, GT)


# ##########################
def loss_mae_fg(Pred, GT, normalized= 'Max'):
    assert (GT >= 0.0).all() and (GT <= 1.0).all(), "The GT is out of range [0, 1]"
    if normalized == 'Max':
        Max = torch.max(torch.max(Pred, -1)[0], -1)[0] # [BatchSize, 1]
        assert Max.shape == torch.Size([BatchSize, 1]), "The shape of Max is not [BatchSize, 1]"
        Max = Max.unsqueeze(-1).unsqueeze(-1)           # [BatchSize, 1, 1, 1]
        assert Max.size() == torch.Size([BatchSize, 1, 1, 1]), "Max.size() is wrong"
        Pred = Pred / (Max + 1e-4)
    elif normalized == 'Mean':
        GT_fg_mask = torch.where(GT> 0.5, 1.0, 0.0).to(device)
        num_fg_pixel = torch.sum(GT_fg_mask, dim=(-1, -2))
        Mean = (torch.sum(Pred * GT_fg_mask, dim=(2, 3))/num_fg_pixel)   # The mean value of reconstructed target amplitude hologram's foreground, [BatchSize, 1]
        Mean = Mean.unsqueeze(-1).unsqueeze(-1)
        Pred = torch.clamp(Pred/Mean, 0.0, 1.0)
    elif normalized == 'None':
        pass
    # assert ((GT == 1) + (GT == 0)).all(), "GT is not binary image"
    assert (Pred >= 0.0).all() and (Pred <= 1.0).all(), "The Pred is out of range [0, 1]"
    fg_loss = torch.sum(torch.where(GT>=0.5, torch.abs(GT-Pred), torch.zeros_like(GT)), dim=(2, 3))
    num_fg_pixel = torch.sum((GT >= 0.5), dim=(2,3))
    return (fg_loss/num_fg_pixel).mean()

class LOSS_MAE_FG(nn.Module):
    def __init__(self):
        super(LOSS_MAE_FG, self).__init__()
    def forward(self, Pred, GT, normalized= 'Max'):
        return loss_mae_fg(Pred, GT, normalized)


def loss_mse_fg(Pred, GT, normalized= 'Max'):
    assert (GT >= 0.0).all() and (GT <= 1.0).all(), "The GT is out of range [0, 1]"
    if normalized == 'Max':
        Max = torch.max(torch.max(Pred, -1)[0], -1)[0] # [BatchSize, 1]
        assert Max.shape == torch.Size([BatchSize, 1]), "The shape of Max is not [BatchSize, 1]"
        Max = Max.unsqueeze(-1).unsqueeze(-1)           # [BatchSize, 1, 1, 1]
        assert Max.size() == torch.Size([BatchSize, 1, 1, 1]), "Max.size() is wrong"
        Pred = Pred / (Max + 1e-4)
    elif normalized == 'Mean':
        GT_binarized = torch.where(GT> 0.5, 1.0, 0.0).to(device)
        num_fg_pixel = torch.sum(GT_binarized, dim=(-1, -2))
        Mean = (torch.sum(Pred * GT_binarized, dim=(2, 3))/num_fg_pixel)   # The mean value of reconstructed target amplitude hologram's foreground, [BatchSize, 1]
        Mean = Mean.unsqueeze(-1).unsqueeze(-1)
        Pred = torch.clamp(Pred/Mean, 0.0, 1.0)
    elif normalized == 'None':
        pass
    # assert ((GT == 1) + (GT == 0)).all(), "GT is not binary image"
    assert (Pred >= 0.0).all() and (Pred <= 1.0).all(), "The Pred is out of range [0, 1]"
    fg_loss = torch.sum(torch.where(GT>=0.5, (GT-Pred)**2, torch.zeros_like(GT)), dim=(2, 3))
    num_fg_pixel = torch.sum((GT >= 0.5), dim=(2,3))
    # return (fg_loss/num_fg_pixel).mean()
    return torch.sqrt((fg_loss/num_fg_pixel)).mean()

class LOSS_MSE_FG(nn.Module):
    def __init__(self):
        super(LOSS_MSE_FG, self).__init__()
    def forward(self, Pred, GT, normalized= 'Max'):
        return loss_mse_fg(Pred, GT, normalized)


def loss_mse_bg(Pred, GT, normalized= 'Max'):
    assert (GT >= 0.0).all() and (GT <= 1.0).all(), "The GT is out of range [0, 1]"
    if normalized == 'Max':
        Max = torch.max(torch.max(Pred, -1)[0], -1)[0] # [BatchSize, 1]
        assert Max.shape == torch.Size([BatchSize, 1]), "The shape of Max is not [BatchSize, 1]"
        Max = Max.unsqueeze(-1).unsqueeze(-1)           # [BatchSize, 1, 1, 1]
        assert Max.size() == torch.Size([BatchSize, 1, 1, 1]), "Max.size() is wrong"
        Pred = Pred / (Max + 1e-4)
    elif normalized == 'Mean':
        GT_binarized = torch.where(GT> 0.5, 1.0, 0.0).to(device)
        num_fg_pixel = torch.sum(GT_binarized, dim=(-1, -2))
        Mean = (torch.sum(Pred * GT_binarized, dim=(2, 3))/num_fg_pixel)   # The mean value of reconstructed target amplitude hologram's foreground, [BatchSize, 1]
        Mean = Mean.unsqueeze(-1).unsqueeze(-1)
        Pred = torch.clamp(Pred/Mean, 0.0, 1.0)
    elif normalized == 'None':
        pass
    assert (Pred >= 0.0).all() and (Pred <= 1.0).all(), "The Pred is out of range [0, 1]"
    bg_loss = torch.sum(torch.where(GT<0.5, (GT-Pred)**2, torch.zeros_like(GT)), dim=(2, 3))
    num_bg_pixel = torch.sum((GT < 0.5), dim=(2,3))
    # return (fg_loss/num_fg_pixel).mean()
    return torch.sqrt((bg_loss/num_bg_pixel)).mean()

class LOSS_MSE_BG(nn.Module):
    def __init__(self):
        super(LOSS_MSE_BG, self).__init__()
    def forward(self, Pred, GT, normalized= 'Max'):
        return loss_mse_bg(Pred, GT, normalized)

def loss_fg_bg_afterEnegyAssignment(Pred, GT, fg_mask, loss_kind):
    # Pred is the actual reconstructed amplitude hologram
    # GT is the scale upped expected amplitude hologram based on 2500 total energy
    # fg_mask shows that the foreground is 1 and background is 0
    # loss_kind is 'mse' or 'mae'
    fg_pixel_num = torch.sum(fg_mask, dim=(2,3)) # (BatchSize, 1)
    bg_pixel_num = ((torch.ones((BatchSize, 1))).to(device)) * 2500 - fg_pixel_num
    assert (fg_pixel_num > 0).all() and (bg_pixel_num > 0).all() and ((fg_pixel_num + bg_pixel_num) == 2500).all(), "The pixel acount of the foreground and the background is wrong calculated"
    if loss_kind == 'mse':
        diff = (GT-Pred)**2                     # (BatchSize, 1, 100, 100)
    elif loss_kind == 'mae':
        diff = torch.abs(GT - Pred)             # (BatchSize, 1, 100, 100)
    fg_diff = diff * fg_mask
    bg_diff = diff - diff * fg_mask
    fg_diff_mean = torch.sum(fg_diff, dim=(2,3)) / fg_pixel_num   # Mean square error or Mean absolute error
    bg_diff_mean = torch.sum(bg_diff, dim=(2,3)) / bg_pixel_num   # Size of (BatchSize, 1)
    # print("fg_diff_mean is \n {}".format(fg_diff_mean))
    # print("bg_diff_mean is \n {}".format(bg_diff_mean))
    # print("fg_diff_mean of a batch is \n {}".format(fg_diff_mean.mean()))
    # print("bg_diff_mean of a batch is \n {}".format(bg_diff_mean.mean()))    
    # return fg_diff_mean.mean() + bg_diff_mean.mean()
    return fg_diff_mean.mean(), bg_diff_mean.mean()

class LOSS_FG_BG_AFTERENERGYASSIGN(nn.Module):
    def __init__(self):
        super(LOSS_FG_BG_AFTERENERGYASSIGN, self).__init__()
    def forward(self, Pred, GT, fg_mask, loss_kind):
        return loss_fg_bg_afterEnegyAssignment(Pred, GT, fg_mask, loss_kind=loss_kind)


def loss_mse_fg_AfterEnergyAssignment(Pred, GT):
    assert (GT >= 0.0).all() and (GT <= 1.0).all(), "The GT is out of range [0, 1]"
    assert (Pred >= 0.0).all() and (Pred <= 1.0).all(), "The Pred is out of range [0, 1]"
    # assert Pred.max() <= 1.0 and Pred.min() >= 0.0, "The Pred is out of range [0,1]"
    fg_loss = torch.sum(torch.where(GT>=0.5, (GT-Pred)**2, torch.zeros_like(GT)), dim=(2, 3))
    num_fg_pixel = torch.sum((GT >= 0.5), dim=(2,3))
    # return (fg_loss/num_fg_pixel).mean()
    return torch.sqrt((fg_loss/num_fg_pixel)).mean()

class LOSS_MSE_FG_AfterEnergyAssignment(nn.Module):
    def __init__(self):
        super(LOSS_MSE_FG_AfterEnergyAssignment, self).__init__()
    def forward(self, Pred, GT):
        return loss_mse_fg_AfterEnergyAssignment(Pred, GT)


def loss_mse_bg_AfterEnergyAssignment(Pred, GT):
    # assert Pred.max() <= 1.0 and Pred.min() >= 0.0, "The Pred is out of range [0,1]"
    bg_loss = torch.sum(torch.where(GT<0.5, (GT-Pred)**2, torch.zeros_like(GT)), dim=(2, 3))
    num_bg_pixel = torch.sum((GT < 0.5), dim=(2,3))
    # return (bg_loss/num_bg_pixel).mean()
    return torch.sqrt((bg_loss/num_bg_pixel)).mean()

class LOSS_MSE_BG_AfterEnergyAssignment(nn.Module):
    def __init__(self):
        super(LOSS_MSE_BG_AfterEnergyAssignment, self).__init__()
    def forward(self, Pred, GT):
        return loss_mse_bg_AfterEnergyAssignment(Pred, GT)


def loss_mae_fg_AfterEnergyAssignment(Pred, GT):
    fg_loss = torch.sum(torch.where(GT>=0.5, torch.abs(GT-Pred), torch.zeros_like(GT)), dim=(2, 3))
    num_fg_pixel = torch.sum((GT >= 0.5), dim=(2,3))
    return (fg_loss/num_fg_pixel).mean()

class LOSS_MAE_FG_AfterEnergyAssignment(nn.Module):
    def __init__(self):
        super(LOSS_MAE_FG_AfterEnergyAssignment, self).__init__()
    def forward(self, Pred, GT):
        return loss_mae_fg_AfterEnergyAssignment(Pred, GT)


def loss_mae_bg_AfterEnergyAssignment(Pred, GT):
    bg_loss = torch.sum(torch.where(GT<0.5, torch.abs(GT-Pred), torch.zeros_like(GT)), dim=(2, 3))
    num_bg_pixel = torch.sum((GT < 0.5), dim=(2,3))
    return (bg_loss/num_bg_pixel).mean()

class LOSS_MAE_BG_AfterEnergyAssignment(nn.Module):
    def __init__(self):
        super(LOSS_MAE_BG_AfterEnergyAssignment, self).__init__()
    def forward(self, Pred, GT):
        return loss_mae_bg_AfterEnergyAssignment(Pred, GT)

def Max_value_penalty(Pred, GT):
    # The predicted amplitude hologram's maximum pixel value 
    Pred_Max = torch.max(torch.max(Pred, -1)[0], -1)[0]                    # [BatchSize, 1]
    # Scale up expected amplitude hologram by energy
    Expected_Total_Energy = (torch.ones((BatchSize, 1)) * 2500).to(device) # [BatchSize, 1]
    GT_Batch_Energy = torch.sum(GT ** 2, dim=(2,3))                        # [BatchSize, 1]
    Expected_GT_Energy_Ratio = Expected_Total_Energy / GT_Batch_Energy     # [BatchSize, 1]
    Expected_GT_Amp_Ratio = torch.sqrt(Expected_GT_Energy_Ratio).unsqueeze(-1).unsqueeze(-1) # [BatchSize, 1, 1, 1]
    GT_scaledup = GT * Expected_GT_Amp_Ratio                               # [BatchSize, 1, 1, 1]
    assert ((torch.sum(GT_scaledup**2, dim=(2,3)) - Expected_Total_Energy) <= 1e-3).all(), "The scaledup GT energy is not 2500"
    # The scaled up expected amplitude hologram's maximum pixel value
    GT_scaledup_Max = torch.max(torch.max(GT_scaledup, -1)[0], -1)[0]      # [BatchSize, 1]
    diff = torch.abs(Pred_Max - GT_scaledup_Max)
    return diff.mean()

class MAX_VALUE_PENALTY(nn.Module):
    def __init__(self):
        super(MAX_VALUE_PENALTY, self).__init__()
    def forward(self, Pred, GT):
        return Max_value_penalty(Pred, GT)

def loss_MSEorMAE_fg_bg(Pred, GT, loss = 'None', normalized = 'None'):
    assert (GT >= 0.0).all() and (GT <= 1.0).all(), "The GT is out of range [0, 1]"
    if normalized == 'Max_total':
        Max = torch.max(torch.max(Pred, -1)[0], -1)[0] # [BatchSize, 1]
        assert Max.shape == torch.Size([BatchSize, 1]), "The shape of Max is not [BatchSize, 1]"
        Max = Max.unsqueeze(-1).unsqueeze(-1)           # [BatchSize, 1, 1, 1]
        assert Max.size() == torch.Size([BatchSize, 1, 1, 1]), "Max.size() is wrong"
        Pred = Pred / (Max + 1e-4)
        if loss == 'MSE':
            fg_loss = torch.sum(torch.where(GT>=0.5, (GT-Pred)**2, torch.zeros_like(GT)), dim=(2, 3))
            bg_loss = torch.sum(torch.where(GT<0.5, (GT-Pred)**2, torch.zeros_like(GT)), dim=(2,3))
        elif loss == 'MAE':
            fg_loss = torch.sum(torch.where(GT>=0.5, torch.abs(GT-Pred), torch.zeros_like(GT)), dim=(2, 3))
            bg_loss = torch.sum(torch.where(GT<0.5, torch.abs(GT-Pred), torch.zeros_like(GT)), dim=(2,3))
        num_fg_pixel = torch.sum((GT >= 0.5), dim=(2,3))
        num_bg_pixel = torch.sum((GT < 0.5), dim=(2,3))
        return torch.sqrt((fg_loss/num_fg_pixel)).mean(), torch.sqrt((bg_loss/num_bg_pixel)).mean()
        # return (fg_loss/num_fg_pixel).mean(), (bg_loss/num_bg_pixel).mean()
        
    elif normalized == 'Mean_total':
        GT_binarized = torch.where(GT> 0.5, 1.0, 0.0).to(device)
        num_fg_pixel = torch.sum(GT_binarized, dim=(-1, -2))
        Mean = (torch.sum(Pred * GT_binarized, dim=(2, 3))/num_fg_pixel)   # The mean value of reconstructed target amplitude hologram's foreground, [BatchSize, 1]
        Mean = Mean.unsqueeze(-1).unsqueeze(-1)
        Pred = torch.clamp(Pred/Mean, 0.0, 1.0)
        if loss == 'MSE':
            fg_loss = torch.sum(torch.where(GT>=0.5, (GT-Pred)**2, torch.zeros_like(GT)), dim=(2, 3))
            bg_loss = torch.sum(torch.where(GT<0.5, (GT-Pred)**2, torch.zeros_like(GT)), dim=(2,3))
        elif loss == 'MAE':
            fg_loss = torch.sum(torch.where(GT>=0.5, torch.abs(GT-Pred), torch.zeros_like(GT)), dim=(2, 3))
            bg_loss = torch.sum(torch.where(GT<0.5, torch.abs(GT-Pred), torch.zeros_like(GT)), dim=(2,3))
        num_fg_pixel = torch.sum((GT >= 0.5), dim=(2,3))
        num_bg_pixel = torch.sum((GT < 0.5), dim=(2,3))
        return torch.sqrt((fg_loss/num_fg_pixel)).mean(), torch.sqrt((bg_loss/num_bg_pixel)).mean()
        # return (fg_loss/num_fg_pixel).mean(), (bg_loss/num_bg_pixel).mean()

    elif normalized == 'Max_respectiveFGBG':
        GT_binarized = torch.where(GT> 0.5, 1.0, 0.0).to(device)
        GT_fg_mask = torch.where(GT_binarized == 1.0, 1.0, 0.0).to(device)
        GT_bg_mask = torch.where(GT_binarized == 0.0, 1.0, 0.0).to(device)
        GT_fg_pixel_num = torch.sum(GT_fg_mask, dim=(-1, -2))
        GT_bg_pixel_num = torch.sum(GT_bg_mask, dim=(-1, -2))
        assert torch.sum(GT_fg_mask+GT_bg_mask) == BatchSize * 50 * 50, "The sum of foreground and background is wrong"
        assert torch.sum(GT_fg_pixel_num+GT_bg_pixel_num) == BatchSize * 50 * 50, "The sum of foreground and background is wrong"
        GT_fg = GT * GT_fg_mask
        GT_bg = GT * GT_bg_mask
        Pred_fg = Pred * GT_fg_mask
        Pred_bg = Pred * GT_bg_mask
        Pred_fg_max = torch.max(torch.max(Pred_fg, -1)[0], -1)[0] # [BatchSize, 1]
        Pred_fg_max = Pred_fg_max.unsqueeze(-1).unsqueeze(-1)     # [BatchSize, 1, 1, 1]
        Pred_bg_max = torch.max(torch.max(Pred_bg, -1)[0], -1)[0] # [BatchSize, 1]
        Pred_bg_max = Pred_bg_max.unsqueeze(-1).unsqueeze(-1)     # [BatchSize, 1, 1, 1]
        Pred_fg_normlzd = Pred_fg / Pred_fg_max
        Pred_bg_normlzd = Pred_bg / Pred_bg_max
        if loss == 'MSE':
            Pred_fg_loss = torch.sum((GT_fg - Pred_fg_normlzd)**2, dim=(-1,-2)) / GT_fg_pixel_num
            Pred_bg_loss = torch.sum((GT_bg - Pred_bg_normlzd)**2, dim=(-1,-2)) / GT_bg_pixel_num
        elif loss == 'MAE':
            Pred_fg_loss = torch.sum(torch.abs(GT_fg - Pred_fg_normlzd), dim=(-1,-2)) / GT_fg_pixel_num
            Pred_bg_loss = torch.sum(torch.abs(GT_bg - Pred_bg_normlzd), dim=(-1,-2)) / GT_bg_pixel_num
        return Pred_fg_loss.mean(), Pred_bg_loss.mean()

    elif normalized == 'Mean_repectiveFGBG':
        GT_binarized = torch.where(GT> 0.5, 1.0, 0.0).to(device)
        GT_fg_mask = torch.where(GT_binarized == 1.0, 1.0, 0.0).to(device)
        GT_bg_mask = torch.where(GT_binarized == 0.0, 1.0, 0.0).to(device)
        GT_fg_pixel_num = torch.sum(GT_fg_mask, dim=(-1, -2))
        GT_bg_pixel_num = torch.sum(GT_bg_mask, dim=(-1, -2))
        assert torch.sum(GT_fg_mask+GT_bg_mask) == BatchSize * 50 * 50, "The sum of foreground and background is wrong"
        assert torch.sum(GT_fg_pixel_num+GT_bg_pixel_num) == BatchSize * 50 * 50, "The sum of foreground and background is wrong"
        GT_fg = GT * GT_fg_mask
        GT_bg = GT * GT_bg_mask
        Pred_fg = Pred * GT_fg_mask
        Pred_bg = Pred * GT_bg_mask
        Pred_fg_mean = torch.sum(Pred_fg, dim=(2,3)) / GT_fg_pixel_num # [BatchSize, 1]
        Pred_fg_mean = Pred_fg_mean.unsqueeze(-1).unsqueeze(-1)        # [BatchSize, 1, 1, 1]
        Pred_bg_mean = torch.sum(Pred_bg, dim=(2,3)) / GT_bg_pixel_num # [BatchSize, 1]
        Pred_bg_mean = Pred_bg_mean.unsqueeze(-1).unsqueeze(-1)        # [BatchSize, 1, 1, 1]
        Pred_fg_normlzd = torch.clamp(Pred_fg / Pred_bg_mean, 0.0, 1.0)
        Pred_bg_normlzd = torch.clamp(Pred_bg / Pred_bg_mean, 0.0, 1.0)
        if loss == 'MSE':
            Pred_fg_loss = torch.sum((GT_fg - Pred_fg_normlzd)**2, dim=(-1,-2)) / GT_fg_pixel_num
            Pred_bg_loss = torch.sum((GT_bg - Pred_bg_normlzd)**2, dim=(-1,-2)) / GT_bg_pixel_num
        elif loss == 'MAE':
            Pred_fg_loss = torch.sum(torch.abs(GT_fg - Pred_fg_normlzd), dim=(-1,-2)) / GT_fg_pixel_num
            Pred_bg_loss = torch.sum(torch.abs(GT_bg - Pred_bg_normlzd), dim=(-1,-2)) / GT_bg_pixel_num
        return Pred_fg_loss.mean(), Pred_bg_loss.mean()

    elif normalized == 'None':
        GT_binarized = torch.where(GT> 0.5, 1.0, 0.0).to(device)
        GT_fg_mask = torch.where(GT_binarized == 1.0, 1.0, 0.0).to(device)
        GT_bg_mask = torch.where(GT_binarized == 0.0, 1.0, 0.0).to(device)
        GT_fg_pixel_num = torch.sum(GT_fg_mask, dim=(-1, -2))
        GT_bg_pixel_num = torch.sum(GT_bg_mask, dim=(-1, -2))
        GT_fg = GT * GT_fg_mask
        GT_bg = GT * GT_bg_mask
        if loss == 'MSE':
            Pred_fg_loss = torch.sum((GT_fg - Pred)**2, dim=(-1,-2)) / GT_fg_pixel_num
            Pred_bg_loss = torch.sum((GT_bg - Pred)**2, dim=(-1,-2)) / GT_bg_pixel_num
        elif loss == 'MAE':
            Pred_fg_loss = torch.sum(torch.abs(GT_fg - Pred), dim=(-1,-2)) / GT_fg_pixel_num
            Pred_bg_loss = torch.sum(torch.abs(GT_bg - Pred), dim=(-1,-2)) / GT_bg_pixel_num
        return Pred_fg_loss.mean(), Pred_bg_loss.mean()


class LOSS_MSEorMAE_fg_bg(nn.Module):
    def __init__(self):
        super(LOSS_MSEorMAE_fg_bg, self).__init__()
    def forward(self, Pred, GT, loss = 'None', normalized = 'None'):
        return loss_MSEorMAE_fg_bg(Pred, GT, loss = loss, normalized = normalized)
# ##########################

def Var(Pred):
    assert torch.var(Pred, dim=(2, 3)).shape == torch.Size([BatchSize, 1]), "variance calculation of Pred is wrong"
    return torch.var(Pred, dim=(2, 3)).mean()

class LOSS_VAR(nn.Module):
    def __init__(self):
        super(LOSS_VAR, self).__init__()
    def forward(self, Pred):
        return Var(Pred)

def FG_BG(Pred, GT_nmlzd, MetricName='', FG_BG_label=0.5):
    assert (GT_nmlzd >= 0.0).all() and (GT_nmlzd <= 1.0).all(), "The GT_nmlzd is out of range [0, 1]"
    fg_mask = torch.where(GT_nmlzd>FG_BG_label, 1.0, 0.0).to(device)
    bg_mask = torch.where(GT_nmlzd<FG_BG_label, 1.0, 0.0).to(device)
    num_fg_pixel = torch.sum(fg_mask, dim=(-1, -2))              # (BatchSize, 1)
    num_bg_pixel = torch.sum(bg_mask, dim=(-1, -2))              # (BatchSize, 1)
    assert (num_fg_pixel>0).all() and (num_bg_pixel>0).all(), "num_fg_pixel and/or num_bg_pixel < 0"
    assert (num_fg_pixel + num_bg_pixel == 100*100).all(), "num_fg_pixel, num_bg_pixel are Wrong"
    assert num_fg_pixel.shape == torch.Size([BatchSize, 1]) and num_bg_pixel.shape == torch.Size([BatchSize, 1]), "num_fg_pixel or num_bg_pixel is not for individual image"
    Pred_fg = Pred * fg_mask   # (BatchSize, 1, 100, 100)
    Pred_bg =  Pred * bg_mask  # (BatchSize, 1, 100, 100)
    if MetricName == 'AVG':
        Pred_fg_avg = (torch.sum(Pred_fg, dim=(2, 3))/num_fg_pixel).mean()
        Pred_bg_avg = (torch.sum(Pred_bg, dim=(2, 3))/num_bg_pixel).mean()
        return Pred_fg_avg, Pred_bg_avg
    elif MetricName == 'MAX':
        Pred_fg_max = torch.mean(torch.max(torch.max(Pred_fg,2)[0], 2)[0])
        Pred_bg_max = torch.mean(torch.max(torch.max(Pred_bg,2)[0], 2)[0])
        return Pred_fg_max, Pred_bg_max
    elif MetricName == 'MIN':
        Pred_fg_min = torch.mean(torch.min(torch.min(Pred_fg,2)[0], 2)[0])
        Pred_bg_min = torch.mean(torch.min(torch.min(Pred_bg,2)[0], 2)[0])
        return Pred_fg_min, Pred_bg_min
    elif MetricName == 'VAR':
        Pred_fg_var, Pred_bg_var = 0, 0
        for index in range(BatchSize):
            # For foreground
            Pred_fg_index = Pred_fg[index]
            Pred_fg_index = Pred_fg_index[Pred_fg_index!=0]
            Pred_fg_index_var = torch.var(Pred_fg_index)
            Pred_fg_var += Pred_fg_index_var
            # For background
            Pred_bg_index = Pred_bg[index]
            Pred_bg_index = Pred_bg_index[Pred_bg_index!=0]
            Pred_bg_index_var = torch.var(Pred_bg_index)
            Pred_bg_var += Pred_bg_index_var
        Pred_fg_var = Pred_fg_var / BatchSize
        Pred_bg_var = Pred_bg_var / BatchSize
        return Pred_fg_var, Pred_bg_var
    elif MetricName == 'TV':
        h_tv, v_tv = torch.zeros(BatchSize, 1, 100, 100).to(device), torch.zeros(BatchSize, 1, 100, 100).to(device)
        h_tv[:,:,:,1:] = torch.abs(Pred[:,:,:,1:]-Pred[:,:,:,:-1]) # (Batchsize, 1, 50, 49)
        v_tv[:,:,1:,:] = torch.abs(Pred[:,:,1:,:]-Pred[:,:,:-1,:]) # (Batchsize, 1, 49, 50)
        fg_mask_squeezed = fg_mask.squeeze(1)
        bg_mask_squeezed = bg_mask.squeeze(1)
        assert fg_mask_squeezed.shape == torch.Size([BatchSize, 100, 100]) and bg_mask_squeezed.shape == torch.Size([BatchSize, 100, 100]), "fg_mask_squeezed and bg_mask_squeezed shape wrong"
        fg_mask_numpy = fg_mask_squeezed.cpu().numpy()
        bg_mask_numpy = bg_mask_squeezed.cpu().numpy()
        # GT = GT.squeeze(1).cpu().numpy()
        eroded_fg = np.zeros((BatchSize, 100, 100))
        eroded_bg = np.zeros((BatchSize, 100, 100))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2, 2))
        for i in range(BatchSize):
            eroded_fg[i] = cv2.erode(fg_mask_numpy[i], kernel, iterations=1)     # (100, 100)
            eroded_bg[i] = cv2.erode(bg_mask_numpy[i], kernel, iterations=1)     # (100, 100)
            # Binary_condition_fg = ((eroded_fg[i] * GT[i]) == 1) + ((eroded_fg[i] * GT[i]) == 0)
            # while not Binary_condition_fg.all():
            #     eroded_fg[i] = cv2.erode(eroded_fg[i], kernel, iterations=1)
            # eroded_fg[i] = cv2.erode(eroded_fg[i], kernel, iterations=1)
            # Binary_condition_bg = ((eroded_bg[i] * GT[i]) == 1) + ((eroded_bg[i] * GT[i]) == 0)
            # while not Binary_condition_bg.all():
            #     eroded_bg[i] = cv2.erode(eroded_bg[i], kernel, iterations=1)
            # eroded_bg[i] = cv2.erode(eroded_bg[i], kernel, iterations=1)
        eroded_torch_fg = torch.from_numpy(eroded_fg).unsqueeze(1).to(device)  # torch, (BatchSize, 1, 100, 100)
        eroded_torch_bg = torch.from_numpy(eroded_bg).unsqueeze(1).to(device)  # torch, (BatchSize, 1, 100, 100)
        assert (torch.sum(eroded_torch_fg, dim=(-1, -2)) > 0.0).all() and (torch.sum(eroded_torch_bg, dim=(-1, -2)) > 0.0).all(), "after erode the number of foreground pixels degragates to 0"
        h_tv_fg = h_tv * eroded_torch_fg
        v_tv_fg = v_tv * eroded_torch_fg
        h_tv_bg = h_tv * eroded_torch_bg
        v_tv_bg = v_tv * eroded_torch_bg
        tv_fg = (torch.sum(h_tv_fg, dim=(2,3)) + torch.sum(v_tv_fg, dim=(2,3))) / torch.sum(eroded_torch_fg, dim=(2,3))
        tv_bg = (torch.sum(h_tv_bg, dim=(2,3)) + torch.sum(v_tv_bg, dim=(2,3))) / torch.sum(eroded_torch_bg, dim=(2,3))
        return tv_fg.mean(), tv_bg.mean()
    elif MetricName == 'Energy':
        pred_fg_energy = torch.sum(Pred_fg**2, dim=(2,3))  # (BatchSize, 1)
        pred_bg_energy = torch.sum(Pred_bg**2, dim=(2,3))  # (BatchSize, 1)
        # diff_fg_energy = (torch.ones_like(pred_fg_energy)*50*50 - pred_fg_energy) / num_fg_pixel
        # diff_bg_energy = pred_bg_energy / num_bg_pixel
        diff_fg_energy = (torch.ones_like(pred_fg_energy)*50*50 - pred_fg_energy) / (torch.ones_like(pred_fg_energy)*50*50)   # (BatchSize, 1), 1-ratio of foreground energy
        diff_bg_energy = pred_bg_energy / (torch.ones_like(pred_fg_energy)*50*50)   # (BatchSize, 1), ratio of background energy
        return diff_fg_energy.mean(), diff_bg_energy.mean()
    else:
        raise ValueError(
                "The MetricName is not in [AVG, MAX, MIN, VAR, TV, Energy] ")
        

class LOSS_FG_BG(nn.Module):
    def __init__(self, MetricName):
        super(LOSS_FG_BG, self).__init__()
        self.MetricName = MetricName
    def forward(self, Pred, GT_nmlzd):
        return FG_BG(Pred, GT_nmlzd, MetricName=self.MetricName, FG_BG_label=0.5)

class LOSS_FG_BG_v1(nn.Module):
    def __init__(self):
        super(LOSS_FG_BG_v1, self).__init__()
    def forward(self, Pred, GT_nmlzd, MetricName):
        return FG_BG(Pred, GT_nmlzd, MetricName=MetricName, FG_BG_label=0.5)

# def FG_BG(Pred, GT, MetricName='', FG_BG_label=0.5):
#     # assert ((GT==1) + (GT==0)).all(), "GT isn't binary image"
#     fg_mask, bg_mask = torch.where(GT>=FG_BG_label, 1.0, 0.0).to(device), torch.where(GT<=FG_BG_label, 1.0, 0.0).to(device)
#     # fg_mask, bg_mask = torch.where(GT==FG_label, 1.0, 0.0).to(device), torch.where(GT==BG_label, 1.0, 0.0).to(device)        # (BatchSize, 1, 100, 100)
#     num_fg_pixel, num_bg_pixel = torch.sum(fg_mask, dim=(-1, -2)), torch.sum(bg_mask, dim=(-1, -2))                        # (BatchSize, 1)
#     assert torch.all(num_fg_pixel > torch.zeros_like(num_fg_pixel)) and torch.all(num_bg_pixel > torch.zeros_like(num_bg_pixel)), "num_fg_pixel and / or num_bg_pixel < 0"
#     assert torch.sum(num_fg_pixel) + torch.sum(num_bg_pixel) == BatchSize * 50 * 50, "num_fg_pixel, num_bg_pixel are Wrong"
#     assert num_fg_pixel.shape == torch.Size([BatchSize, 1]) and num_bg_pixel.shape == torch.Size([BatchSize, 1]), "num_fg_pixel or num_bg_pixel is not for individual image"
#     Pred_fg = Pred * fg_mask   # (BatchSize, 1, 100, 100)
#     Pred_bg =  Pred * bg_mask  # (BatchSize, 1, 100, 100)
#     # Pred_fg_maxnormalized = (Pred/(torch.max(torch.max(Pred, -1)[0], -1)[0]).unsqueeze(-1).unsqueeze(-1)) * fg
#     # Pred_bg_maxnormalized =  (Pred/(torch.max(torch.max(Pred, -1)[0], -1)[0]).unsqueeze(-1).unsqueeze(-1)) * bg
#     if MetricName == 'AVG':
#         Pred_fg_avg = (torch.sum(Pred_fg, dim=(2, 3))/num_fg_pixel).mean()
#         Pred_bg_avg = (torch.sum(Pred_bg, dim=(2, 3))/num_bg_pixel).mean()
#         return Pred_fg_avg, Pred_bg_avg
#     elif MetricName == 'MAX':
#         Pred_fg_max = torch.mean(torch.max(torch.max(Pred_fg,2)[0], 2)[0])
#         Pred_bg_max = torch.mean(torch.max(torch.max(Pred_bg,2)[0], 2)[0])
#         return Pred_fg_max, Pred_bg_max
#     elif MetricName == 'MIN':
#         Pred_fg_min = torch.mean(torch.min(torch.min(Pred_fg,2)[0], 2)[0])
#         Pred_bg_min = torch.mean(torch.min(torch.min(Pred_bg,2)[0], 2)[0])
#         return Pred_fg_min, Pred_bg_min
#     elif MetricName == 'VAR':
#         Pred_fg_var, Pred_bg_var = 0, 0
#         for index in range(BatchSize):
#             # For foreground
#             Pred_fg_index = Pred_fg[index]
#             Pred_fg_index = Pred_fg_index[Pred_fg_index!=0]
#             Pred_fg_index_var = torch.var(Pred_fg_index)
#             Pred_fg_var += Pred_fg_index_var
#             # For background
#             Pred_bg_index = Pred_bg[index]
#             Pred_bg_index = Pred_bg_index[Pred_bg_index!=0]
#             Pred_bg_index_var = torch.var(Pred_bg_index)
#             Pred_bg_var += Pred_bg_index_var
#         Pred_fg_var = Pred_fg_var / BatchSize
#         Pred_bg_var = Pred_bg_var / BatchSize
#         return Pred_fg_var, Pred_bg_var
#     elif MetricName == 'TV':
#         h_tv, v_tv = torch.zeros(BatchSize, 1, 100, 100).to(device), torch.zeros(BatchSize, 1, 100, 100).to(device)
#         h_tv[:,:,:,1:] = torch.abs(Pred[:,:,:,1:]-Pred[:,:,:,:-1]) # (Batchsize, 1, 50, 49)
#         v_tv[:,:,1:,:] = torch.abs(Pred[:,:,1:,:]-Pred[:,:,:-1,:]) # (Batchsize, 1, 49, 50)
#         fg_mask_squeezed = fg_mask.squeeze(1)
#         bg_mask_squeezed = bg_mask.squeeze(1)
#         assert fg_mask_squeezed.shape == torch.Size([BatchSize, 100, 100]) and bg_mask_squeezed.shape == torch.Size([BatchSize, 100, 100]), "fg_mask_squeezed and bg_mask_squeezed shape wrong"
#         fg_mask_numpy = fg_mask_squeezed.cpu().numpy()
#         bg_mask_numpy = bg_mask_squeezed.cpu().numpy()
#         # GT = GT.squeeze(1).cpu().numpy()
#         eroded_fg = np.zeros((BatchSize, 100, 100))
#         eroded_bg = np.zeros((BatchSize, 100, 100))
#         kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2, 2))
#         for i in range(BatchSize):
#             eroded_fg[i] = cv2.erode(fg_mask_numpy[i], kernel, iterations=1)     # (100, 100)
#             eroded_bg[i] = cv2.erode(bg_mask_numpy[i], kernel, iterations=1)     # (100, 100)
#             # Binary_condition_fg = ((eroded_fg[i] * GT[i]) == 1) + ((eroded_fg[i] * GT[i]) == 0)
#             # while not Binary_condition_fg.all():
#             #     eroded_fg[i] = cv2.erode(eroded_fg[i], kernel, iterations=1)
#             # eroded_fg[i] = cv2.erode(eroded_fg[i], kernel, iterations=1)
#             # Binary_condition_bg = ((eroded_bg[i] * GT[i]) == 1) + ((eroded_bg[i] * GT[i]) == 0)
#             # while not Binary_condition_bg.all():
#             #     eroded_bg[i] = cv2.erode(eroded_bg[i], kernel, iterations=1)
#             # eroded_bg[i] = cv2.erode(eroded_bg[i], kernel, iterations=1)
#         eroded_torch_fg = torch.from_numpy(eroded_fg).unsqueeze(1).to(device)  # torch, (BatchSize, 1, 100, 100)
#         eroded_torch_bg = torch.from_numpy(eroded_bg).unsqueeze(1).to(device)  # torch, (BatchSize, 1, 100, 100)
#         assert (torch.sum(eroded_torch_fg, dim=(-1, -2)) > 0.0).all() and (torch.sum(eroded_torch_bg, dim=(-1, -2)) > 0.0).all(), "after erode the number of foreground pixels degragates to 0"
#         h_tv_fg = h_tv * eroded_torch_fg
#         v_tv_fg = v_tv * eroded_torch_fg
#         h_tv_bg = h_tv * eroded_torch_bg
#         v_tv_bg = v_tv * eroded_torch_bg
#         tv_fg = (torch.sum(h_tv_fg, dim=(2,3)) + torch.sum(v_tv_fg, dim=(2,3))) / torch.sum(eroded_torch_fg, dim=(-1, -2))
#         tv_bg = (torch.sum(h_tv_bg, dim=(2,3)) + torch.sum(v_tv_bg, dim=(2,3))) / torch.sum(eroded_torch_bg, dim=(-1, -2))
#         return tv_fg.mean(), tv_bg.mean()
#     elif MetricName == 'Energy':
#         pred_fg_energy = torch.sum(Pred_fg**2, dim=(2,3))  # (BatchSize, 1)
#         pred_bg_energy = torch.sum(Pred_bg**2, dim=(2,3))  # (BatchSize, 1)
#         # diff_fg_energy = (torch.ones_like(pred_fg_energy)*50*50 - pred_fg_energy) / num_fg_pixel
#         # diff_bg_energy = pred_bg_energy / num_bg_pixel
#         diff_fg_energy = (torch.ones_like(pred_fg_energy)*50*50 - pred_fg_energy) / (torch.ones_like(pred_fg_energy)*50*50)   # (BatchSize, 1), 1-ratio of foreground energy
#         diff_bg_energy = pred_bg_energy / (torch.ones_like(pred_fg_energy)*50*50)   # (BatchSize, 1), ratio of background energy
#         return diff_fg_energy.mean(), diff_bg_energy.mean()




def TV(Pred):
    count_h =  (Pred.size()[2]-1) * Pred.size()[3]
    count_w = Pred.size()[2] * (Pred.size()[3] - 1)
    assert count_h == 99*100 and count_w == 99*100, "count_h and count_w are wrong"
    h_tv = torch.sum(torch.abs(Pred[:,:,1:,:]-Pred[:,:,:-1,:]), dim=(2,3))/count_h
    w_tv = torch.sum(torch.abs(Pred[:,:,:,1:]-Pred[:,:,:,:-1]), dim=(2,3))/count_w
    assert h_tv.shape == torch.Size([BatchSize, 1]) and w_tv.shape == torch.Size([BatchSize, 1]), "When calculate Total variance, the shapes of h_tv and w_tv are wrong"
    return (h_tv+w_tv).mean()

class Loss_TV(nn.Module):
    def __init__(self):
        super(Loss_TV, self).__init__()
    def forward(self, Pred):
        return TV(Pred)



class Loss_TV_diff(nn.Module):
    def __init__(self):
        super(Loss_TV_diff, self).__init__()
    def forward(self, Pred, GT):
        return F.l1_loss(TV(Pred),TV(GT))


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, Pred, GT):
        # Prepare target image
        GT_fg_mask, GT_bg_mask = torch.where(GT>0.9, 1.0, 0.0), torch.where(GT<0.1, 1.0, 0.0)
        GT_fg_num, GT_bg_num = torch.sum(GT_fg_mask, dim=(2,3)), torch.sum(GT_bg_mask, dim=(2,3))
        Pred = F.sigmoid(Pred)
        GT = torch.where(GT>0.9, 1.0, 0.0)
        # logpt_fg = torch.where(GT_fg_mask > 0.5, -self.alpha * (1-Pred)**self.gamma * torch.log(Pred), torch.zeros_like(Pred))
        # logpt_bg = torch.where(GT_bg_mask > 0.5, -(1-self.alpha) * Pred**self.gamma * torch.log(1-Pred), torch.zeros_like(Pred))
        x = -self.alpha * (1-Pred)**self.gamma * torch.log(Pred)
        y = torch.zeros_like(x)
        logpt_fg = torch.where(GT_fg_mask > 0.5, x, y)
        x = -(1-self.alpha) * Pred**self.gamma * torch.log(1-Pred)
        y = torch.zeros_like(x)
        logpt_bg = torch.where(GT_bg_mask > 0.5, x, y)
        fg_FL = torch.sum(logpt_fg, dim=(2,3)) / GT_fg_num
        bg_FL = torch.sum(logpt_bg, dim=(2,3)) / GT_bg_num
        return fg_FL.mean() + bg_FL.mean()

class FocalLoss_Like(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super(FocalLoss_Like, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, Pred, GT, GT_nmlzd):
        GTFg_mask = torch.where(GT_nmlzd > 0.5, 1.0, 0.0)                             # (BatchSize, 1, 100, 100)
        GTBg_mask = torch.where(GT_nmlzd < 0.5, 1.0, 0.0)                             # (BatchSize, 1, 100, 100)
        PredFg_proportion = torch.sqrt(torch.sum(Pred**2 * GTFg_mask, dim=(2,3)) / 2500) # (BatchSize, 1)
        PredBg_proportion = torch.sqrt(torch.sum(Pred**2 * GTBg_mask, dim=(2,3)) / 2500) # (BatchSize, 1)
        assert (PredFg_proportion >= 0.0).all() and (PredFg_proportion <= 1.0).all(), "PredFg_proportion is not in [0, 1]"
        assert (PredBg_proportion >= 0.0).all() and (PredBg_proportion <= 1.0).all(), "PredBg_proportion is not in [0, 1]"
        # print("PredFg_proportion {} \nPredBg_proportion {}".format(PredFg_proportion, PredBg_proportion))
        PredFg_proportion = PredFg_proportion.unsqueeze(-1).unsqueeze(-1)               # (BatchSize, 1, 1, 1)
        PredBg_proportion = PredBg_proportion.unsqueeze(-1).unsqueeze(-1)               # (BatchSize, 1, 100, 100)
        Diff = torch.abs(Pred - GT)                                                   # (BatchSize, 1, 100, 100)
        PredFg_diff = Diff * GTFg_mask                                                # (BatchSize, 1, 100, 100)
        PredBg_diff = Diff * GTBg_mask                                                # (BatchSize, 1, 100, 100)
        FL_like = self.alpha * (1-PredFg_proportion)**self.gamma * PredFg_diff + (1 - self.alpha) * PredBg_proportion**self.gamma * PredBg_diff
        # print("FL_like mean {}".format(torch.mean(FL_like, dim=(2,3))))
        # print(FL_like.mean())
        return FL_like.mean()


class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm, self).__init__()
    def forward(self, pred, truth):
        '''
        pred & truth: the amplitude value return from ASM function, its shape is (BatchSize, 1, 100, 100)
        formula: pixel-wise distance, then do **2, then sum, then we get l2norm's square shape of (BatchSize,1)
                 then, get mean value among minibatch
        return: value that in the range of [0, 1], the mean of return value is 0.25
        '''
        assert (torch.sum(torch.sum((pred - truth)**2, dim = -1), dim = -1) / 2500).shape[0] == BatchSize, 'L2norm term wrong!'
        return (torch.sum(torch.sum((pred - truth)**2, dim = -1), dim = -1) / 2500).mean()
        # return (torch.sum(torch.sum((pred - truth)**2, dim = -1), dim = -1) / 500).mean()

class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self,x):
        '''
        x: (pred - truth), its shape is (BatchSize, 1, 100, 100) i.e., torch.Size([BatchSize, 1, 100, 100])
        '''
        x = x.squeeze(1)
        assert x.shape == torch.Size([BatchSize, 100, 100])
        batch_size = x.size()[0]
        count_h =  (x.size()[1]-1) * x.size()[2]
        count_w = x.size()[1] * (x.size()[2] - 1)
        # h_tv = torch.pow((x[:,1:,:]-x[:,:-1,:]),2).sum()
        # w_tv = torch.pow((x[:,:,1:]-x[:,:,:-1]),2).sum()
        h_tv = torch.abs(x[:,1:,:]-x[:,:-1,:]).sum()
        w_tv = torch.abs(x[:,:,1:]-x[:,:,:-1]).sum()
        # print("count_h:{}, count_w:{}".format(count_h, count_w))
        # print("h_tv:{}, w_tv:{}".format(h_tv, w_tv))
        # print("tvloss:{}, tvloss.shape:{}".format((h_tv/count_h + w_tv/count_w) / batch_size, ((h_tv/count_h + w_tv/count_w) / batch_size).shape))
        # exit()
        return (h_tv/count_h + w_tv/count_w) / batch_size
        # return (h_tv/count_h + w_tv/count_w) / batch_size * 5

class Gradient_Net(nn.Module):
  def __init__(self):
    super(Gradient_Net, self).__init__()
    kernel_x = [[-1., 0., 1.], [-1., 0., 1.], [-1., 0., 1.]]
    kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0).to(device)

    kernel_y = [[-1., -1., -1.], [0., 0., 0.], [1., 1., 1.]]
    kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0).to(device)

    self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
    self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

  def forward(self, x):
    grad_x = F.conv2d(x, self.weight_x, padding=1)
    grad_y = F.conv2d(x, self.weight_y, padding=1)
    gradient_x = torch.abs(grad_x)
    gradient_y = torch.abs(grad_y)
    return gradient_x, gradient_y

def gradient(x):
    '''
    the input of gradient function is torch.Size([16, 1, 100, 100])
    the output of gradient function is torch.Size([16, 1, 100, 100])
    '''
    gradient_model = Gradient_Net().to(device)
    g_x, g_y = gradient_model(x)
    return g_x, g_y


class NORM(nn.Module):
    def __init__(self):
        super(NORM, self).__init__()

    def forward(self, pred, truth):
        '''
         pred, truth: torch.Size([16, 1, 100, 100])
        '''
        dx_pred, dy_pred = gradient(pred)    # torch.Size([16, 1, 100, 100])
        dx_truth, dy_truth = gradient(truth) # torch.Size([16, 1, 100, 100])
        Identity = torch.ones_like(dx_pred)  # torch.Size([16, 1, 100, 100])
        n_pred = torch.stack([-dx_pred.squeeze(1), -dy_pred.squeeze(1), Identity.squeeze(1)], dim=1)
        n_truth = torch.stack([-dx_truth.squeeze(1), -dy_truth.squeeze(1), Identity.squeeze(1)], dim=1)
        cos_similarity = F.cosine_similarity(n_pred, n_truth)
        return (1-cos_similarity).sum() / (cos_similarity.shape[0] * cos_similarity.shape[-2] * cos_similarity.shape[-1])
        # return (1-cos_similarity).sum() / (cos_similarity.shape[0] * cos_similarity.shape[-2] * cos_similarity.shape[-1]) * 5 # cos_similarity torch.Size([16, 100, 100])
