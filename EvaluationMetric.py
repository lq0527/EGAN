import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity, peak_signal_noise_ratio, mean_squared_error
from DataLoader import BatchSize, device
from AuxiliaryFunction import *
from LossFunction import *




def PSNR(Pred_nmlzd, GT_nmlzd,BatchSize=BatchSize):
    Max = torch.max(torch.max(Pred_nmlzd, -1)[0], -1)[0] # [BatchSize, 1]
    assert Max.shape == torch.Size([BatchSize, 1]), "The shape of Max is not [BatchSize, 1]"
    # Max_test = Max
    # print('max is',Max_test)
    Max = Max.unsqueeze(-1).unsqueeze(-1)          # [BatchSize, 1, 1, 1]
    assert Max.size() == torch.Size([BatchSize, 1, 1, 1]), "Max.size() is wrong"
    Pred_nmlzd = Pred_nmlzd / (Max+0.00001)                  # [BatchSize, 1, 100, 100]
    torch.set_printoptions(threshold=np.inf)

    assert Pred_nmlzd.shape == torch.Size([BatchSize, 1, 100, 100]), "The shape of Pred_nmlzd is not (BatchSize, 1, 100, 100)"
    assert GT_nmlzd.shape == torch.Size([BatchSize, 1, 100, 100]), "The shape of GT_nmlzd is not (BatchSize, 1, 100, 100)"
    assert (Pred_nmlzd>=0.0).all() and (Pred_nmlzd<=1.0001).all(), "The Pred_nmlzd amplitude value is out of range [0, 1]"
    assert (GT_nmlzd>=0.0).all() and (GT_nmlzd<=1.0).all(), "The GT_nmlzd amplitude value is out of range [0, 1]"
    # print('Pred_nmlzd')
    # print(Pred_nmlzd)
    mse = torch.mean((Pred_nmlzd - GT_nmlzd)**2, dim=(2,3))   # [BatchSize, 1]
    assert mse.shape == torch.Size([BatchSize, 1]), "mse is not for each image in a batch"
    assert (mse>=0.0).all() and (mse<=1.0).all(), "The calculated mse is out of range [0, 1]"
    # print('mse')
    # print(mse)
    assert (mse>=0.0).all() and (mse<=1.0).all(), "The calculated mse is out of range [0, 1]"
    
    PIXEL_MAX = torch.ones_like(mse)              # [BatchSize, 1]
    
    psnr_batch = torch.where(mse != 0, 20*torch.log10(PIXEL_MAX/torch.sqrt(mse)), torch.ones_like(mse)*100)  # [BatchSize, 1]
    psnr = torch.mean(psnr_batch)
    # Check whether it is right for the calculation of psnr for each image in a batch
    # psnr_mean = (torch.sum(((mse[mse == 0] + 1)*100)) + torch.sum(20 * torch.log10(PIXEL_MAX / torch.sqrt(mse[mse != 0])))) / BatchSize
    # assert torch.abs(psnr_mean - psnr) <= 1e-2, "There is something wrong for calculation of psnr"
    return psnr


def gaussian_window(self, size, channels, sigma):
    # This is function to model a Gaussian Function
    gaussian = np.arange(-(size / 2), size / 2)
    gaussian = np.exp(-1. * (gaussian**2) / (2 * sigma**2))
    gaussian = np.outer(gaussian, gaussian.reshape((size, 1))) # Compute the outer product, extend to 2D
    gaussian = gaussian / np.sum(gaussian)
    gaussian = np.reshape(gaussian, (1, size, size, 1)) # Reshape to 4D
    gaussian = np.tile(gaussian, (1, 1, 1, channels)) # Construct an array by repeating gaussian the number of times given by (1, 1, 1, channels)
    return gaussian

def gaussian(window_size, sigma):
    gauss = torch.Tensor([torch.exp(-(x - window_size//2)**2 / float(2 * sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1) # Expand dimension
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0) # mm does not broadcast, torch.matmul can broadcast
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size = 11, window = None, size_average = True, full = False, val_range = None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    '''
    L is the dynamic range of the pixel values.
    '''
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
        (_, channel, height, width) = img1.size() # _ is minibatch
        if window is None:
            real_size = min(window_size, height, width)
            window = create_window(real_size, channel = channel).to(img1.device)

        mu1 = F.conv2d(img1, window, padding = padd, groups=channel) #mu1代表图像1的平均值
        mu2 = F.conv2d(img2, window, padding = padd, groups=channel) #mu2代表图像2的平均值
        mu1_sq = mu1.pow(2)#pow()求乘方函数
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding = padd, groups=channel) - mu1_sq  #sigma1代表Img1的标准差
        sigma2_sq = F.conv2d(img2 * img2, window, padding = padd, groups=channel) - mu2_sq  #sigma2代表Img2的标准差
        sigma12 = F.conv2d(img1 * img2, window, padding = padd, groups=channel) - mu1_mu2  #sigma12代表Img1Img2的协方差

        C1 = (0.01 * L) ** 2
        C2 = (0.03 * L) ** 2

        v1 = 2.0 * mu1_mu2 + C1
        v2 = mu1_sq + mu2_sq + C1
        ls = v1 / v2 # Luminance sensitivity

        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs = v1 / v2 # Contrast & Structure sensitivity

        ssim_map = ls * cs # it is the general form of Structural SIMilarity (SSIM) index between signal img1 and img2
        assert ssim_map <= 1, "does not satisfy boundedness condition!"

        if size_average:
            cs = cs.mean()
            ret = ssim_map.mean()
        else:
            cs = cs.mean(1).mean(1).mean(1)
            ret = ssim_map.mean(1).mean(1).mean(1)
        if full:
            return ret, cs
        return ret



def msssim(img1, img2, window_size = 11, size_average = True, val_range = None, normalize = None):
    # This function takes the distorted img and reference img as the input,
    # and the system iteratively applies a low=pass filter  and downsamples the filtered images by a factor of 2.
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0] # level is 5
    ssims = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)

        # Relu normalize (not compliant with original definition)
        if normalize == "relu":
            ssims.append(torch.relu(sim))
            mcs.append(torch.relu(cs))
        else:
            ssims.append(sim)
            mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    ssims = torch.stack(ssims)
    mcs = torch.stack(mcs)

    # Simple normalize (not compliant with original definition)
    # TODO: remove support for normalize == True (kept for backward support)
    if normalize == 'simple' or normalize == True:
        ssims = (ssims + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = ssims ** weights

    output = torch.prod(pow1[:-1]) * pow2[-1]
    return output

def Validation(NOTE, output_ch, test_loader, net, clamp = False):
    # Loader well trained net pth file
    net = net.to(device)
    # net = nn.DataParallel(net)
    net_path = "./model_pth/" + NOTE + ".pth"  #存放.pth的路径
    
    metric_file = './Results/Metrics/' + NOTE + '.txt'
    with open(metric_file,'a') as file0:
        print("Used net path is: {}".format(net_path), file=file0)  #在txt文件写入使用的网络的路径
    assert os.path.exists(net_path), "file: '{}' does not exist.".format(net_path)
    # state_dict =torch.load(net_path, map_location = device)['state_dict']
    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     if 'module' not in k:
    #         k = 'module.'+k
    #     else:
    #         k = k.replace('features.module.', 'module.features.')
    #     new_state_dict[k]=v
    # net.load_state_dict(new_state_dict)
    net.load_state_dict(torch.load(net_path, map_location = device)) # , strict=False; torch.load_state_dict()函数就是用于将预训练的参数权重加载到新的模型之中
    # Prepare folders to save results
    validation_save_path = "./Results/Visulaization/" + NOTE + "/Validation/"
    if not os.path.exists(validation_save_path):
        os.makedirs(validation_save_path)
    MetricComparison_save_path = "./Results/Visulaization/" + NOTE + "/MetricComparison/"
    if not os.path.exists(MetricComparison_save_path):
        os.makedirs(MetricComparison_save_path)
    ATA_amp_save_path = "./Results/ATA_Amplitude/" + NOTE+ "/"
    if not os.path.exists(ATA_amp_save_path):
        os.makedirs(ATA_amp_save_path)

    num = len(test_loader)
    avg_mae_Ac, avg_mae_Aretri = 0.0, 0.0
    avg_mse_Ac, avg_mse_Aretri = 0.0, 0.0
    # avg_fg_Ac_withAe, avg_fg_Ar_withAe = 0.0, 0.0
    avg_psnr_Ac_from_Asoutput_Pu = 0.0   #通过AsPu重建的Ac
    avg_psnr_Ac_from_Asretrieved_Pu = 0.0 #通过A1Pu重建的Ac
    # avg_psnr_Ac_from_Asretrieved_Puscaled = 0.0
    avg_psnr_Ac_from_Asoutput_Psretrived = 0.0  #AsPs
    avg_psnr_Ac_from_Psretrived_A1retrived = 0.0  #A1Ps
    # avg_psnr_Ac_from_Asoutput_A1threshold = 0.0
    # avg_psnr_Ac_from_Psretrieved_A1threshold = 0.0
    # avg_psnr_Ac_from_Asoutput_Pu_mean_normlzd = 0.0
    # avg_psnr_Ac_from_Asretrieved_Pu_mean_normlzd = 0.0
    # avg_psnr_Ac_from_Asoutput_A1threshold_mean_normlzd = 0.0
    # avg_psnr_Ac_from_Psretrieved_A1threshold_mean_normlzd = 0.0
    avg_mae_Ps_withPu, avg_mae_PS_withPSmean = 0.0, 0.0
    avg_var_A1, avg_tv_A1 = 0.0, 0.0
    # avg_cosmae_Ps_and_Psretrieved = 0.0

    total_row = 7
    total_column = 5
    test_times = 0.0

    val_avg_psnr = 0
    val_avg_ssim = 0
    val_avg_mse = 0
    val_avg_rmse = 0
    val_avg_acc = 0
    val_avg_eff = 0

    PSNR_for_Boxplot, SSIM_for_Boxplot, ACC_for_Boxplot, EFF_for_Boxplot = [], [], [], []

    net.eval()
    with torch.no_grad():
        
        for _, test_data in enumerate(test_loader):
            test_times += 1
            # test_input, test_output_phs = test_data
            # test_input, test_output_phs = test_input.to(device), test_output_phs.to(device)
            
            test_input = test_data     # torch.Size([16, 1, 100, 100])
            # test_input_energy_scalup = Amplitude_Scaleup(Ae_Batch = test_input.to(device), scaleup = 'Energy_Proportionated')
            # normalizer = Amplitude_Normalization_Scaleup(Ae_Batch = test_input.to(device), normalized = 'Energy_Proportionated')
            if output_ch == 1:
                test_output_Amp = net(test_input.to(device))
            elif output_ch == 2:
                test_output = net(test_input.to(device))
                test_output_Amp = test_output[:,0].unsqueeze(1)          # torch.Size([16, 1, 100, 100])

            # test_output_phs = test_output_phs - torch.floor(test_output_phs/(2*torch.pi)) * (2*torch.pi)

            # IASA(NOTE, ATA_Amp=test_output_Amp[0].unsqueeze(1), Expected_Amp=test_input[0].unsqueeze(1).to(device), Z_distance=20e-3, Lam=6.4e-4, fs=1/(50e-3/50), iterations=10)###调用IASA 绘图
            
            # Reconstruction based on Asoutput and Pu
            uniform_Pu = torch.zeros_like(test_output_Amp)     # torch.Size([16, 1, 100, 100])
            test_Ac, test_Pc = ASM(d=20e-3, PhsHolo=uniform_Pu, AmpHolo=test_output_Amp.to(device))
            # Retrieved Psretrieved and A1retrieved
            test_A1, test_Ps = ASM(d=-20e-3, PhsHolo=test_Pc, AmpHolo=test_input.to(device))
            # Reconstruction based on Asretrieved and Pu
            Ac_from_Asretrieved_Pu,  _ = ASM(d=20e-3, PhsHolo=uniform_Pu, AmpHolo=test_A1.to(device))  #Ac_from_Psretrieved_Au
            # Reconstruction based on Asoutput (or Psretrieved) and Auscaled
            #uniform_Au_scaled = uniform_Au * torch.mean(test_A1, (2,3)).unsqueeze(1).unsqueeze(1)  #?
            # uniform_Pu_scaled = uniform_Pu * torch.mean(test_Ps, (2,3)).unsqueeze(1).unsqueeze(1)
            Ac_from_Asoutput_Pu, _ = ASM(d=20e-3, PhsHolo=uniform_Pu , AmpHolo=test_output_Amp.to(device)) #Ac_from_Asoutput_Pu
            #Ac_from_Asretrieved_Au, _ = ASM(d=20e-3, PhsHolo=uniform_Pu, AmpHolo=uniform_Au_scaled)#Ac_from_Psretrieved_Auscaled
            # Reconstruction based on Asoutput (or Psretrieved) and A1retrieved
            Ac_from_Asoutput_Psretrived, _ = ASM(d=20e-3, PhsHolo=test_Ps, AmpHolo=test_output_Amp.to(device))  #Ac_from_Asoutput_A1retrived
            check_Ae, _ = ASM(d=20e-3, PhsHolo=test_Ps, AmpHolo=test_A1.to(device))
            # # Reconstruction based on Asoutput (or Psretrieved) and A1threshold
            # Psthreshold = torch.zeros_like(test_Ps).to(device)
            # Threshold = test_Ps.max()/5 #???
            # Psthreshold[test_Ps>Threshold] = 2*torch.pi
            # Psthreshold[test_Ps<=Threshold] = 0.0
            # Ac_from_Asoutput_Psthreshold, _ = ASM(d=20e-3, PhsHolo= Psthreshold , AmpHolo=test_output_Amp)
            # Ac_from_Asretrieved_Psthreshold, _ = ASM(d=20e-3, PhsHolo= Psthreshold , AmpHolo=test_A1)  #Ac_from_Psretrieved_A1threshold
            
            
            Max = torch.max(torch.max(test_Ac, -1)[0], -1)[0]      # [BatchSize, 1]
            assert Max.shape == torch.Size([BatchSize, 1]), "The shape of Max is not [BatchSize, 1]"
            Max = Max.unsqueeze(-1).unsqueeze(-1)                  # [BatchSize, 1, 1, 1]
            assert Max.size() == torch.Size([BatchSize, 1, 1, 1]), "Max.size() is wrong"
            assert ((test_input==1) + (test_input==0)).all(), "test_input is not binary image" 
            # test_input_fg_mask = torch.where(test_input>0.5, 1.0, 0.0).to(device)
            # num_fg_pixel = torch.sum(test_input_fg_mask.to(device), dim=(-1, -2))
            # Mean = (torch.sum(test_Ac*test_input_fg_mask, dim=(2, 3))/num_fg_pixel)   # [BatchSize, 1]
            # Mean = Mean.unsqueeze(-1).unsqueeze(-1)                # [BatchSize, 1, 1, 1]
            # test_A1, test_Ps = ASM(d=-20e-3, PhsHolo=test_Pc, AmpHolo=test_input.to(device)*Mean)
            # Ac_from_Asoutput_Pu_mean_normlzd = torch.clamp(test_Ac/Mean, 0.0, 1.0)
            # Mean_Psretri_Pu = (torch.sum(Ac_from_Asretrieved_Pu*test_input_fg_mask, dim=(2, 3))/num_fg_pixel)   # [BatchSize, 1]
            # Mean_Psretri_Pu = Mean_Psretri_Pu.unsqueeze(-1).unsqueeze(-1)                # [BatchSize, 1, 1, 1]
            # Ac_from_Asretrieved_Pu_mean_normlzd = torch.clamp(Ac_from_Asretrieved_Pu/Mean_Psretri_Pu, 0.0, 1.0)
            # Mean_Asoutput_A1threshold = (torch.sum(Ac_from_Asoutput_A1threshold*test_input_fg_mask, dim=(2, 3))/num_fg_pixel)   # [BatchSize, 1]
            # Mean_Asoutput_A1threshold = Mean_Asoutput_A1threshold.unsqueeze(-1).unsqueeze(-1)                # [BatchSize, 1, 1, 1]
            # Ac_from_Asoutput_A1threshold_mean_normlzd = torch.clamp(Ac_from_Asoutput_A1threshold/Mean_Asoutput_A1threshold, 0.0, 1.0)
            # Mean_Psretri_A1threshold = (torch.sum(Ac_from_Psretrieved_A1threshold*test_input_fg_mask, dim=(2, 3))/num_fg_pixel)   # [BatchSize, 1]
            # Mean_Psretri_A1threshold = Mean_Psretri_A1threshold.unsqueeze(-1).unsqueeze(-1)                # [BatchSize, 1, 1, 1]
            # Ac_from_Psretrieved_A1threshold_mean_normlzd = torch.clamp(Ac_from_Psretrieved_A1threshold/Mean_Psretri_A1threshold, 0.0, 1.0)

            mae_Ac_withAe = F.l1_loss(test_Ac/(torch.max(torch.max(test_Ac, -1)[0], -1)[0]).unsqueeze(-1).unsqueeze(-1), test_input.to(device))
            mae_Ar_withAe = F.l1_loss(Ac_from_Asretrieved_Pu/(torch.max(torch.max(Ac_from_Asretrieved_Pu, -1)[0], -1)[0]).unsqueeze(-1).unsqueeze(-1), test_input.to(device))
            mse_Ac_withAe = F.mse_loss(test_Ac/(torch.max(torch.max(test_Ac, -1)[0], -1)[0]).unsqueeze(-1).unsqueeze(-1), test_input.to(device))
            mse_Ar_withAe = F.mse_loss(Ac_from_Asretrieved_Pu/(torch.max(torch.max(Ac_from_Asretrieved_Pu, -1)[0], -1)[0]).unsqueeze(-1).unsqueeze(-1), test_input.to(device))
         
            psnr_Amp_c = PSNR(test_Ac, test_input.to(device))
            psnr_Amp_r = PSNR(Ac_from_Asretrieved_Pu, test_input.to(device))
         
            psnr_Amp_from_Asoutput_A1retrived = PSNR(Ac_from_Asoutput_Psretrived, test_input.to(device))##x
            psnr_Amp_from_Psretrieved_A1retrived = PSNR(check_Ae, test_input.to(device))

            # mae_Phs_1 = F.l1_loss(test_Ps, uniform_Pu)   #MAE
            # mae_Phs_1mean = F.l1_loss(test_Ps, torch.mean(test_Ps))
            # assert Var(uniform_Pu) == 0, "Calculation of variance of uniform P is wrong"
            var_Amp_1 = Var(test_A1)
            tv_Amp_1 = TV(test_A1)
            # cosmae_Phs = loss_cos_mse(test_output_Amp, test_Ps)
           
            avg_mae_Ac += mae_Ac_withAe
            avg_mae_Aretri += mae_Ar_withAe
            avg_mse_Ac += mse_Ac_withAe
            avg_mse_Aretri += mse_Ar_withAe
            avg_psnr_Ac_from_Asoutput_Pu += psnr_Amp_c
            avg_psnr_Ac_from_Asretrieved_Pu += psnr_Amp_r
            avg_psnr_Ac_from_Asoutput_Psretrived += psnr_Amp_from_Asoutput_A1retrived
            avg_psnr_Ac_from_Psretrived_A1retrived += psnr_Amp_from_Psretrieved_A1retrived

            avg_var_A1 += var_Amp_1
            avg_tv_A1 += tv_Amp_1


            test_input = test_input.squeeze(1).cpu().numpy()       # (16, 100, 100)
            # test_input_energy_scalup = test_input_energy_scalup.squeeze(1).cpu().numpy()
            uniform_Pu = uniform_Pu.squeeze(1).cpu().numpy()
            test_output_Amp = test_output_Amp.squeeze(1).detach().cpu().numpy()
            test_Ac = test_Ac.squeeze(1).detach().cpu().numpy()    # (16, 100, 100)
            test_Pc = test_Pc.squeeze(1).detach().cpu().numpy()
            test_A1 = test_A1.squeeze(1).detach().cpu().numpy()
            test_Ps = test_Ps.squeeze(1).detach().cpu().numpy()
            Ac_from_Asoutput_Psretrived = Ac_from_Asoutput_Psretrived.squeeze(1).detach().cpu().numpy()  #AsPs
            Ac_from_Asoutput_Pu = Ac_from_Asoutput_Pu.squeeze(1).detach().cpu().numpy()  #AsPu
            Ac_from_Asretrieved_Pu = Ac_from_Asretrieved_Pu.squeeze(1).detach().cpu().numpy()  #A1Pu
            check_Ae = check_Ae.squeeze(1).detach().cpu().numpy()

            test_Ac_max = np.max(test_Ac, axis=(1,2))  # (16,)
            # test_Ac_mean = (np.sum(test_Ac*test_input_fg_mask, axis=(1,2))/num_fg_pixel)   # (16,)
            test_Ac_max_normalized = test_Ac / (test_Ac_max[:, np.newaxis, np.newaxis]+1e-4)  # (16, 100, 100) / (16, 1, 1) -> (16, 100, 100)
            # test_Ac_mean_normalized = np.clip(test_Ac / (test_Ac_mean[:, np.newaxis, np.newaxis]+1e-4), 0.0, 1.0)
            # test_Ac_energy_normalized = test_Ac / normalizer  # (16, 100, 100)
            # test_Ac_energy_normalized = np.where(test_Ac_energy_normalized>1.0, 2-test_Ac_energy_normalized, test_Ac_energy_normalized)


            test_A1_max = np.max(test_A1, axis=(1,2))
            test_A1_max_normalized = test_A1 / (test_A1_max[:, np.newaxis, np.newaxis]+1e-4)

            Ac_from_Asoutput_Psretrived_max = np.max(Ac_from_Asoutput_Psretrived, axis=(1,2))
            Ac_from_Asoutput_Psretrived_max_normalized = Ac_from_Asoutput_Psretrived / (Ac_from_Asoutput_Psretrived_max[:, np.newaxis, np.newaxis]+1e-4)
            Ac_from_Asoutput_Pu_max = np.max(Ac_from_Asoutput_Pu, axis=(1,2))
            Ac_from_Asoutput_Pu_max_normalized = Ac_from_Asoutput_Pu / (Ac_from_Asoutput_Pu_max[:, np.newaxis, np.newaxis]+1e-4)

            Ac_from_Asretrieved_Pu_max = np.max(Ac_from_Asretrieved_Pu, axis=(1,2))
            # Ac_from_Asretrieved_Pu_mean = (np.sum(Ac_from_Asretrieved_Pu*test_input_fg_mask, axis=(1,2))/num_fg_pixel)
            Ac_from_Asretrieved_Pu_max_normalized = Ac_from_Asretrieved_Pu / (Ac_from_Asretrieved_Pu_max[:, np.newaxis, np.newaxis]+1e-4)
            # Ac_from_Asretrieved_Pu_mean_normalized = np.clip(Ac_from_Asretrieved_Pu / (Ac_from_Asretrieved_Pu_mean[:, np.newaxis, np.newaxis]+1e-4), 0.0 , 1.0)
            # Ac_from_Asretrieved_Pu_energy_normalized = Ac_from_Asretrieved_Pu / normalizer
            # Ac_from_Asretrieved_Pu_energy_normalized = np.where(Ac_from_Asretrieved_Pu_energy_normalized>1.0, 2-Ac_from_Asretrieved_Pu_energy_normalized, Ac_from_Asretrieved_Pu_energy_normalized)

            check_Ae_max = np.max(check_Ae, axis=(1,2))
            # check_Ae_mean = (np.sum(check_Ae*test_input_fg_mask, axis=(1,2))/num_fg_pixel)
            check_Ae_max_normalized = check_Ae / (check_Ae_max[:, np.newaxis, np.newaxis]+1e-4)
            # check_Ae_mean_normalized = np.clip(check_Ae / (check_Ae_mean[:, np.newaxis, np.newaxis]+1e-4), 0.0 , 1.0)

            # uniform_Au_scaled = np.ones_like(test_A1) * np.mean(test_A1, (1,2)).reshape(BatchSize, 1, 1)
            
            with open(metric_file,'a') as file0:  #写入txt文档
                print('  index      reconstructed from                                                         PSNR       SSIM     MSE    Accuracy    Efficacy', file=file0)

            batch_sum_psnr = 0
            batch_sum_ssim = 0
            batch_sum_mse = 0
            
            batch_sum_rmse = 0
            batch_sum_acc = 0
            batch_sum_eff = 0

            for index in range(BatchSize):  #BatchSize=16
                
                assert (test_input[index]).max() <= 1.0, "(test_input[index]).max() > 1.0"
                assert (test_output_Amp[index]).max() < (1+0.01), "(test_output_amp[index]).max() > 1"   #x
                assert (test_Ac[index]).max() <= 1.0, "(test_Ac[index]).max() > 1.0"

                psnr = peak_signal_noise_ratio(test_Ac_max_normalized[index], test_input[index])
                psnr = np.round(psnr, decimals=2) #保留两位小数
                myssim = structural_similarity(test_Ac_max_normalized[index], test_input[index])
                myssim = np.round(myssim, decimals=2)
                mse = mean_squared_error(test_Ac_max_normalized[index], test_input[index])
                mse = np.round(mse, decimals=2)
                acc = accuracy(test_Ac_max_normalized[index], test_input[index])
                acc = np.round(acc, decimals=2)
                eff = efficacy(test_Ac_max_normalized[index], test_input[index])
                eff = np.round(eff, decimals=2)
              
                #def save_AS_xlsx(save_path, save_OutAmp_np, save_Retriphase_np, save_Amp_np, save_PPOH_A1retrievedThreshold_np, save_img_index):
                save_AS_xlsx(ATA_amp_save_path, test_output_Amp[index], test_Ps[index], test_input[index], index)
                # save_PS_xlsx(PTA_phase_save_path, test_output_phs[index], test_Ps[index], test_input[index], A1threshold[index], index)
                Reconstructed_Dict = {
                'From Asout and Pu                              ': test_Ac_max_normalized[index],
                'From AsRetrieved and Pu                        ': Ac_from_Asretrieved_Pu_max_normalized[index],
                'From Asout and Pu                              ': Ac_from_Asoutput_Pu_max_normalized[index], 
                # 'From PsRetrieved and A1Scaled                  ': Ac_from_Asretrieved_Puscaled_max_normalized[index], 
                'From Asout and Psretrievd                      ': Ac_from_Asoutput_Psretrived_max_normalized[index], 
                'From PSRetrieved and A1retrievd                ': check_Ae_max_normalized[index], 
                # 'From Asout and A1threshold                     ': Ac_from_Asoutput_A1threshold_max_normalized[index],
                # 'From PSRetrieved and A1threshold               ': Ac_from_Psretrieved_A1threshold_max_normalized[index], 
                # 'From Asout and Au (mean normlzd)               ': test_Ac_mean_normalized[index],
                # 'From PsRetrieved and Au (mean normlzd)         ': Ac_from_Asretrieved_Pu_mean_normalized[index],
                # 'From Asout and A1threshold (mean normlzd)      ': Ac_from_Asoutput_A1threshold_mean_normalized[index],
                # 'From PSRetrieved and A1threshold (mean normlzd)': Ac_from_Psretrieved_A1threshold_mean_normalized[index]
                }
                for key_i, value_i in Reconstructed_Dict.items():
                    # Calculate PSNR
                    psnr = peak_signal_noise_ratio(value_i, test_input[index])
                    # Calculate SSIM
                    myssim = structural_similarity(value_i, test_input[index],data_range=1.0)
                    # Calculate MSE
                    mse = mean_squared_error(value_i, test_input[index])

                    acc = accuracy(value_i, test_input[index])
                    eff = efficacy(value_i, test_input[index])

                    with open(metric_file,'a') as file0:
                        print('  {:.3f}      {}                           {:.3f}     {:.3f}    {:.3f}     {:.3f}'.format(index, key_i, psnr, myssim, mse, acc, eff), file=file0)

                    if key_i == 'From Asout and Pu                              ':
                        batch_sum_psnr += psnr
                        batch_sum_ssim += myssim
                        batch_sum_mse += mse
                        batch_sum_rmse += np.sqrt(mse)
                        batch_sum_acc += acc
                        batch_sum_eff += eff
                ###############################################################绘制输入、输出、以及多个重建图############################
                if test_times == 1:
                    MetricsEvaluation_Dict = {
                        'AE_': test_input[index], #输入，即期待幅值图 Expected Hologram
                        'AS_': test_output_Amp[index],#输出幅值图
                        'OU_': test_Ac_max_normalized[index], #经过均一化的重建图
                        'RU_': Ac_from_Asretrieved_Pu_max_normalized[index],#
                        # 'RO_': Ac_from_Asoutput_Pu_max_normalized[index],#
                        'OR_': Ac_from_Asoutput_Psretrived_max_normalized[index], 
                        'RR_': check_Ae_max_normalized[index], 
                        }
                    subplot_index = 0
                    psnr_list, ssim_list, mse_list, acc_list, eff_list= [], [], [], [], []
                    for key_i, value_i in MetricsEvaluation_Dict.items():
                        subplot_index += 1
                        # if subplot_index == 2:
                        #     plot_subimage(total_row=4, total_column=4, sub_index=int(subplot_index), img=value_i, title=key_i, bar_min=0, bar_max=1, title_size=6)
                        # else:
                        psnr = peak_signal_noise_ratio(value_i, test_input[index])
                        psnr = np.round(psnr, decimals=2)
                        myssim = structural_similarity(value_i, test_input[index])
                        myssim = np.round(myssim, decimals=2)
                        mse = mean_squared_error(value_i, test_input[index])
                        mse = np.round(mse, decimals=2)
                        acc = accuracy(value_i, test_input[index])
                        acc = np.round(acc, decimals=2)
                        eff = efficacy(value_i, test_input[index])
                        eff = np.round(eff, decimals=2)
                        psnr_list.append(psnr)
                        ssim_list.append(myssim)
                        mse_list.append(mse)
                        acc_list.append(acc)
                        eff_list.append(eff)
                        plot_subimage(total_row=2, total_column=3, sub_index=int(subplot_index), img=value_i, title=key_i+str(psnr)+', '+str(myssim)+', '+str(mse)+', '+str(acc), bar_min=0, bar_max=1, title_size=6)
                    plt.savefig(MetricComparison_save_path + str(index)+'.png')  #保存AE AS OU RU OR RR的图像
                    plt.clf() # plt.clf () # 清除当前 figure 的所有axes，但是不关闭这个 window，所以能继续复用于其他的 plot
                    #################################################################几个不同重建图的指标曲线
                    plt.figure(figsize=(8, 6))  #尺寸
                    length_of_our_list = len(psnr_list)
                    plt.axis([1, length_of_our_list+1, 0, 100])  #左端的y轴值，代表PSNR，范围从0到100
                    plt.axis('on')
                    plt.grid(True)
                    plt.plot(range(1,length_of_our_list+1,1), psnr_list, 'r*-', ms=4, label='psnr')   #红色——————————
                    plt.xlabel('Case')  #x轴名称
                    plt.ylabel('PSNR', color='r')  #y轴名称
                    plt.tick_params(axis = 'y', labelcolor = 'r')
                    plt.legend(loc = 'upper left') #图例位置 放在左边
                    
                    ax2 = plt.twinx() ## 将plt1的x轴也分配给ax2使用
                    ax2.plot(range(1,length_of_our_list+1,1), ssim_list, 'go--', ms=4, label='ssim')  #绿色------------
                    ax2.plot(range(1,length_of_our_list+1,1), mse_list, 'bs-.', ms=4, label='mse')    #蓝色-·-·-·-·-·-·-·-·-
                    ax2.plot(range(1,length_of_our_list+1,1), acc_list, 'k<:', ms=4, label='acc')     #黑色······
                    # ax2.plot(range(1,length_of_our_list+1,1), eff_list, 'y>--', ms=4, label='eff')
                    ax2.set_ylabel('ssim & mse & acc & eff', color = 'k')
                    plt.tick_params(axis = 'y', labelcolor = 'k')
                    plt.legend(loc = 'upper right') #图例位置 放在右边
                    plt.savefig(MetricComparison_save_path + 'MetricCurves_' + str(test_times)+'.png')#绘制PSNR ssim mse acc等指标的图像
                    plt.clf()

                    plt.imshow(test_input[index],vmax=test_input[index].max(),vmin=test_input[index].min())
                    plt.axis('off')
                    plt.savefig(MetricComparison_save_path + str(index) + '_'+ 'AE.png', dpi=300,bbox_inches='tight')
                    plt.clf()
                    plt.imshow(test_output_Amp[index],vmax=test_output_Amp[index].max(),vmin=test_output_Amp[index].min())
                    plt.axis('off')
                    plt.savefig(MetricComparison_save_path + str(index) + '_'+ 'AS.png', dpi=300,bbox_inches='tight')
                    plt.clf()
                    plt.imshow(test_Ac_max_normalized[index],vmax=test_Ac_max_normalized[index].max(),vmin=test_Ac_max_normalized[index].min())
                    plt.axis('off')
                    plt.savefig(MetricComparison_save_path + str(index) + '_'+ 'Reconstructed.png', dpi=300,bbox_inches='tight')
                    plt.clf()



                ###############################################################################################
            batch_avg_psnr = batch_sum_psnr / BatchSize
            batch_avg_ssim = batch_sum_ssim / BatchSize
            batch_avg_mse = batch_sum_mse / BatchSize
            batch_avg_rmse = batch_sum_rmse / BatchSize
            batch_avg_acc = batch_sum_acc / BatchSize
            batch_avg_eff = batch_sum_eff / BatchSize

            val_avg_psnr += batch_avg_psnr
            val_avg_ssim += batch_avg_ssim
            val_avg_mse += batch_avg_mse
            val_avg_rmse += batch_avg_rmse
            val_avg_acc += batch_avg_acc
            val_avg_eff += batch_avg_eff

        assert test_times == num, "test_times is not same as num of test_loader"

        
        with open(metric_file,'a') as file0:
            print("\n", file=file0)
            print("=============================", file=file0)
            print("========= avg batch ==========", file=file0)
            avg_mae_Ac /= num
            # avg_mae_Aretri /= num
            avg_mse_Ac /= num
            # avg_mse_Aretri /= num
            avg_psnr_Ac_from_Asoutput_Pu /= num
      
            avg_mae_Ps_withPu /= num
            avg_mae_PS_withPSmean /= num
            avg_var_A1 /= num
            # avg_cosmae_Ps_and_Psretrieved /= num

            val_avg_psnr /= num
            val_avg_ssim /= num
            val_avg_mse /= num
            val_avg_rmse /= num
            val_avg_acc /= num
            val_avg_eff /= num
            
            print("batch_avg_psnr =     {} \n".format(batch_avg_psnr), file=file0)
            print("batch_avg_ssim =     {} \n".format(batch_avg_ssim), file=file0)
            print("batch_avg_rmse =     {} \n".format(batch_avg_rmse), file=file0)
            print("batch_avg_acc =     {} \n".format(batch_avg_acc), file=file0)
            print("batch_avg_eff =     {} \n".format(batch_avg_eff), file=file0)
            print("=============================", file=file0)
            print("========= avg diff ==========", file=file0)
            
            

            

            print("avg_mae_Ac =     {} \n".format(avg_mae_Ac), file=file0)
            # print("avg_mae_Aretri = {} \n".format(avg_mae_Aretri), file=file0)
            print("avg_mse_Ac =     {} \n".format(avg_mse_Ac), file=file0)
            # print("avg_mse_Aretri = {} \n".format(avg_mse_Aretri), file=file0)
            print("avg_psnr_Ac_from_Asoutput_Pu =             {} \n".format(avg_psnr_Ac_from_Asoutput_Pu), file=file0)
          
            print("avg_var_A1 =            {} \n".format(avg_var_A1), file=file0)
            # print("avg_cosmae_Ps_and_Psretrieved = {} \n".format(avg_cosmae_Ps_and_Psretrieved), file=file0)

            print("val_avg_psnr =     {} \n".format(val_avg_psnr), file=file0)
            print("val_avg_ssim =     {} \n".format(val_avg_ssim), file=file0)
            print("val_avg_mse =     {} \n".format(val_avg_mse), file=file0)
            print("val_avg_rmse =     {} \n".format(val_avg_rmse), file=file0)
            print("val_avg_acc =      {} \n".format(val_avg_acc), file=file0)
            print("val_avg_eff =      {} \n".format(val_avg_eff), file=file0)

            print("=============================")
            print("=============================")




def Validation1(NOTE, output_ch, test_loader, net, clamp = False, metric_file='', net_path=''):
    # Loader well trained net pth file
    net = net.to(device)
    # net = nn.DataParallel(net)
    # net_path = "./model_pth/" + NOTE + ".pth"  #存放.pth的路径
    # metric_file = './Results/Metrics/' + NOTE + '.txt'
    with open(metric_file,'a') as file0:
        print("Used net path is: {}".format(net_path), file=file0)  #在txt文件写入使用的网络的路径
    assert os.path.exists(net_path), "file: '{}' does not exist.".format(net_path)
    # state_dict =torch.load(net_path, map_location = device)['state_dict']
    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     if 'module' not in k:
    #         k = 'module.'+k
    #     else:
    #         k = k.replace('features.module.', 'module.features.')
    #     new_state_dict[k]=v
    # net.load_state_dict(new_state_dict)
    net.load_state_dict(torch.load(net_path, map_location = device)) # , strict=False; torch.load_state_dict()函数就是用于将预训练的参数权重加载到新的模型之中
    # Prepare folders to save results
    validation_save_path = "./Results/Visulaization/" + NOTE + "/Validation_test/"
    if not os.path.exists(validation_save_path):
        os.makedirs(validation_save_path)
    MetricComparison_save_path = "./Results/Visulaization/" + NOTE + "/MetricComparison_test/"
    if not os.path.exists(MetricComparison_save_path):
        os.makedirs(MetricComparison_save_path)
    ATA_amp_save_path = "./Results/ATA_Amplitude/" + NOTE+ "_test/"
    if not os.path.exists(ATA_amp_save_path):
        os.makedirs(ATA_amp_save_path)

    num = len(test_loader)
    avg_mae_Ac, avg_mae_Aretri = 0.0, 0.0
    avg_mse_Ac, avg_mse_Aretri = 0.0, 0.0
    # avg_fg_Ac_withAe, avg_fg_Ar_withAe = 0.0, 0.0
    avg_psnr_Ac_from_Asoutput_Pu = 0.0   #通过AsPu重建的Ac
    avg_psnr_Ac_from_Asretrieved_Pu = 0.0 #通过A1Pu重建的Ac
    # avg_psnr_Ac_from_Asretrieved_Puscaled = 0.0
    avg_psnr_Ac_from_Asoutput_Psretrived = 0.0  #AsPs
    avg_psnr_Ac_from_Psretrived_A1retrived = 0.0  #A1Ps
    # avg_psnr_Ac_from_Asoutput_A1threshold = 0.0
    # avg_psnr_Ac_from_Psretrieved_A1threshold = 0.0
    # avg_psnr_Ac_from_Asoutput_Pu_mean_normlzd = 0.0
    # avg_psnr_Ac_from_Asretrieved_Pu_mean_normlzd = 0.0
    # avg_psnr_Ac_from_Asoutput_A1threshold_mean_normlzd = 0.0
    # avg_psnr_Ac_from_Psretrieved_A1threshold_mean_normlzd = 0.0
    avg_mae_Ps_withPu, avg_mae_PS_withPSmean = 0.0, 0.0
    avg_var_A1, avg_tv_A1 = 0.0, 0.0
    # avg_cosmae_Ps_and_Psretrieved = 0.0

    total_row = 7
    total_column = 5
    test_times = 0.0

    val_avg_psnr = 0
    val_avg_ssim = 0
    val_avg_mse = 0
    val_avg_rmse = 0
    val_avg_acc = 0
    val_avg_eff = 0

    PSNR_for_Boxplot, SSIM_for_Boxplot, ACC_for_Boxplot, EFF_for_Boxplot = [], [], [], []

    net.eval()
    with torch.no_grad():
        
        for _, test_data in enumerate(test_loader):
            test_times += 1
            # test_input, test_output_phs = test_data
            # test_input, test_output_phs = test_input.to(device), test_output_phs.to(device)
            
            test_input = test_data     # torch.Size([16, 1, 100, 100])
            # test_input_energy_scalup = Amplitude_Scaleup(Ae_Batch = test_input.to(device), scaleup = 'Energy_Proportionated')
            # normalizer = Amplitude_Normalization_Scaleup(Ae_Batch = test_input.to(device), normalized = 'Energy_Proportionated')
            if output_ch == 1:
                test_output_Amp = net(test_input.to(device))
            elif output_ch == 2:
                test_output = net(test_input.to(device))
                test_output_Amp = test_output[:,0].unsqueeze(1)          # torch.Size([16, 1, 100, 100])

            # test_output_phs = test_output_phs - torch.floor(test_output_phs/(2*torch.pi)) * (2*torch.pi)

            # IASA(NOTE, ATA_Amp=test_output_Amp[0].unsqueeze(1), Expected_Amp=test_input[0].unsqueeze(1).to(device), Z_distance=20e-3, Lam=6.4e-4, fs=1/(50e-3/50), iterations=10)###调用IASA 绘图
            
            # Reconstruction based on Asoutput and Pu
            uniform_Pu = torch.zeros_like(test_output_Amp)     # torch.Size([16, 1, 100, 100])
            test_Ac, test_Pc = ASM(d=20e-3, PhsHolo=uniform_Pu, AmpHolo=test_output_Amp.to(device))
            # Retrieved Psretrieved and A1retrieved
            test_A1, test_Ps = ASM(d=-20e-3, PhsHolo=test_Pc, AmpHolo=test_input.to(device))
            # Reconstruction based on Asretrieved and Pu
            Ac_from_Asretrieved_Pu,  _ = ASM(d=20e-3, PhsHolo=uniform_Pu, AmpHolo=test_A1.to(device))  #Ac_from_Psretrieved_Au
            # Reconstruction based on Asoutput (or Psretrieved) and Auscaled
            #uniform_Au_scaled = uniform_Au * torch.mean(test_A1, (2,3)).unsqueeze(1).unsqueeze(1)  #?
            # uniform_Pu_scaled = uniform_Pu * torch.mean(test_Ps, (2,3)).unsqueeze(1).unsqueeze(1)
            Ac_from_Asoutput_Pu, _ = ASM(d=20e-3, PhsHolo=uniform_Pu , AmpHolo=test_output_Amp.to(device)) #Ac_from_Asoutput_Pu
            #Ac_from_Asretrieved_Au, _ = ASM(d=20e-3, PhsHolo=uniform_Pu, AmpHolo=uniform_Au_scaled)#Ac_from_Psretrieved_Auscaled
            # Reconstruction based on Asoutput (or Psretrieved) and A1retrieved
            Ac_from_Asoutput_Psretrived, _ = ASM(d=20e-3, PhsHolo=test_Ps, AmpHolo=test_output_Amp.to(device))  #Ac_from_Asoutput_A1retrived
            check_Ae, _ = ASM(d=20e-3, PhsHolo=test_Ps, AmpHolo=test_A1.to(device))
            # # Reconstruction based on Asoutput (or Psretrieved) and A1threshold
            # Psthreshold = torch.zeros_like(test_Ps).to(device)
            # Threshold = test_Ps.max()/5 #???
            # Psthreshold[test_Ps>Threshold] = 2*torch.pi
            # Psthreshold[test_Ps<=Threshold] = 0.0
            # Ac_from_Asoutput_Psthreshold, _ = ASM(d=20e-3, PhsHolo= Psthreshold , AmpHolo=test_output_Amp)
            # Ac_from_Asretrieved_Psthreshold, _ = ASM(d=20e-3, PhsHolo= Psthreshold , AmpHolo=test_A1)  #Ac_from_Psretrieved_A1threshold
            
            
            Max = torch.max(torch.max(test_Ac, -1)[0], -1)[0]      # [BatchSize, 1]
            assert Max.shape == torch.Size([BatchSize, 1]), "The shape of Max is not [BatchSize, 1]"
            Max = Max.unsqueeze(-1).unsqueeze(-1)                  # [BatchSize, 1, 1, 1]
            assert Max.size() == torch.Size([BatchSize, 1, 1, 1]), "Max.size() is wrong"
            assert ((test_input==1) + (test_input==0)).all(), "test_input is not binary image" 
            # test_input_fg_mask = torch.where(test_input>0.5, 1.0, 0.0).to(device)
            # num_fg_pixel = torch.sum(test_input_fg_mask.to(device), dim=(-1, -2))
            # Mean = (torch.sum(test_Ac*test_input_fg_mask, dim=(2, 3))/num_fg_pixel)   # [BatchSize, 1]
            # Mean = Mean.unsqueeze(-1).unsqueeze(-1)                # [BatchSize, 1, 1, 1]
            # test_A1, test_Ps = ASM(d=-20e-3, PhsHolo=test_Pc, AmpHolo=test_input.to(device)*Mean)
            # Ac_from_Asoutput_Pu_mean_normlzd = torch.clamp(test_Ac/Mean, 0.0, 1.0)
            # Mean_Psretri_Pu = (torch.sum(Ac_from_Asretrieved_Pu*test_input_fg_mask, dim=(2, 3))/num_fg_pixel)   # [BatchSize, 1]
            # Mean_Psretri_Pu = Mean_Psretri_Pu.unsqueeze(-1).unsqueeze(-1)                # [BatchSize, 1, 1, 1]
            # Ac_from_Asretrieved_Pu_mean_normlzd = torch.clamp(Ac_from_Asretrieved_Pu/Mean_Psretri_Pu, 0.0, 1.0)
            # Mean_Asoutput_A1threshold = (torch.sum(Ac_from_Asoutput_A1threshold*test_input_fg_mask, dim=(2, 3))/num_fg_pixel)   # [BatchSize, 1]
            # Mean_Asoutput_A1threshold = Mean_Asoutput_A1threshold.unsqueeze(-1).unsqueeze(-1)                # [BatchSize, 1, 1, 1]
            # Ac_from_Asoutput_A1threshold_mean_normlzd = torch.clamp(Ac_from_Asoutput_A1threshold/Mean_Asoutput_A1threshold, 0.0, 1.0)
            # Mean_Psretri_A1threshold = (torch.sum(Ac_from_Psretrieved_A1threshold*test_input_fg_mask, dim=(2, 3))/num_fg_pixel)   # [BatchSize, 1]
            # Mean_Psretri_A1threshold = Mean_Psretri_A1threshold.unsqueeze(-1).unsqueeze(-1)                # [BatchSize, 1, 1, 1]
            # Ac_from_Psretrieved_A1threshold_mean_normlzd = torch.clamp(Ac_from_Psretrieved_A1threshold/Mean_Psretri_A1threshold, 0.0, 1.0)

            mae_Ac_withAe = F.l1_loss(test_Ac/(torch.max(torch.max(test_Ac, -1)[0], -1)[0]).unsqueeze(-1).unsqueeze(-1), test_input.to(device))
            mae_Ar_withAe = F.l1_loss(Ac_from_Asretrieved_Pu/(torch.max(torch.max(Ac_from_Asretrieved_Pu, -1)[0], -1)[0]).unsqueeze(-1).unsqueeze(-1), test_input.to(device))
            mse_Ac_withAe = F.mse_loss(test_Ac/(torch.max(torch.max(test_Ac, -1)[0], -1)[0]).unsqueeze(-1).unsqueeze(-1), test_input.to(device))
            mse_Ar_withAe = F.mse_loss(Ac_from_Asretrieved_Pu/(torch.max(torch.max(Ac_from_Asretrieved_Pu, -1)[0], -1)[0]).unsqueeze(-1).unsqueeze(-1), test_input.to(device))
         
            psnr_Amp_c = PSNR(test_Ac, test_input.to(device))
            psnr_Amp_r = PSNR(Ac_from_Asretrieved_Pu, test_input.to(device))
         
            psnr_Amp_from_Asoutput_A1retrived = PSNR(Ac_from_Asoutput_Psretrived, test_input.to(device))##x
            psnr_Amp_from_Psretrieved_A1retrived = PSNR(check_Ae, test_input.to(device))

            # mae_Phs_1 = F.l1_loss(test_Ps, uniform_Pu)   #MAE
            # mae_Phs_1mean = F.l1_loss(test_Ps, torch.mean(test_Ps))
            # assert Var(uniform_Pu) == 0, "Calculation of variance of uniform P is wrong"
            var_Amp_1 = Var(test_A1)
            tv_Amp_1 = TV(test_A1)
            # cosmae_Phs = loss_cos_mse(test_output_Amp, test_Ps)
           
            avg_mae_Ac += mae_Ac_withAe
            avg_mae_Aretri += mae_Ar_withAe
            avg_mse_Ac += mse_Ac_withAe
            avg_mse_Aretri += mse_Ar_withAe
            avg_psnr_Ac_from_Asoutput_Pu += psnr_Amp_c
            avg_psnr_Ac_from_Asretrieved_Pu += psnr_Amp_r
            avg_psnr_Ac_from_Asoutput_Psretrived += psnr_Amp_from_Asoutput_A1retrived
            avg_psnr_Ac_from_Psretrived_A1retrived += psnr_Amp_from_Psretrieved_A1retrived

            avg_var_A1 += var_Amp_1
            avg_tv_A1 += tv_Amp_1


            test_input = test_input.squeeze(1).cpu().numpy()       # (16, 100, 100)
            # test_input_energy_scalup = test_input_energy_scalup.squeeze(1).cpu().numpy()
            uniform_Pu = uniform_Pu.squeeze(1).cpu().numpy()
            test_output_Amp = test_output_Amp.squeeze(1).detach().cpu().numpy()
            test_Ac = test_Ac.squeeze(1).detach().cpu().numpy()    # (16, 100, 100)
            test_Pc = test_Pc.squeeze(1).detach().cpu().numpy()
            test_A1 = test_A1.squeeze(1).detach().cpu().numpy()
            test_Ps = test_Ps.squeeze(1).detach().cpu().numpy()
            Ac_from_Asoutput_Psretrived = Ac_from_Asoutput_Psretrived.squeeze(1).detach().cpu().numpy()  #AsPs
            Ac_from_Asoutput_Pu = Ac_from_Asoutput_Pu.squeeze(1).detach().cpu().numpy()  #AsPu
            Ac_from_Asretrieved_Pu = Ac_from_Asretrieved_Pu.squeeze(1).detach().cpu().numpy()  #A1Pu
            check_Ae = check_Ae.squeeze(1).detach().cpu().numpy()

            test_Ac_max = np.max(test_Ac, axis=(1,2))  # (16,)
            # test_Ac_mean = (np.sum(test_Ac*test_input_fg_mask, axis=(1,2))/num_fg_pixel)   # (16,)
            test_Ac_max_normalized = test_Ac / (test_Ac_max[:, np.newaxis, np.newaxis]+1e-4)  # (16, 100, 100) / (16, 1, 1) -> (16, 100, 100)
            # test_Ac_mean_normalized = np.clip(test_Ac / (test_Ac_mean[:, np.newaxis, np.newaxis]+1e-4), 0.0, 1.0)
            # test_Ac_energy_normalized = test_Ac / normalizer  # (16, 100, 100)
            # test_Ac_energy_normalized = np.where(test_Ac_energy_normalized>1.0, 2-test_Ac_energy_normalized, test_Ac_energy_normalized)


            test_A1_max = np.max(test_A1, axis=(1,2))
            test_A1_max_normalized = test_A1 / (test_A1_max[:, np.newaxis, np.newaxis]+1e-4)

            Ac_from_Asoutput_Psretrived_max = np.max(Ac_from_Asoutput_Psretrived, axis=(1,2))
            Ac_from_Asoutput_Psretrived_max_normalized = Ac_from_Asoutput_Psretrived / (Ac_from_Asoutput_Psretrived_max[:, np.newaxis, np.newaxis]+1e-4)
            Ac_from_Asoutput_Pu_max = np.max(Ac_from_Asoutput_Pu, axis=(1,2))
            Ac_from_Asoutput_Pu_max_normalized = Ac_from_Asoutput_Pu / (Ac_from_Asoutput_Pu_max[:, np.newaxis, np.newaxis]+1e-4)

            Ac_from_Asretrieved_Pu_max = np.max(Ac_from_Asretrieved_Pu, axis=(1,2))
            # Ac_from_Asretrieved_Pu_mean = (np.sum(Ac_from_Asretrieved_Pu*test_input_fg_mask, axis=(1,2))/num_fg_pixel)
            Ac_from_Asretrieved_Pu_max_normalized = Ac_from_Asretrieved_Pu / (Ac_from_Asretrieved_Pu_max[:, np.newaxis, np.newaxis]+1e-4)
            # Ac_from_Asretrieved_Pu_mean_normalized = np.clip(Ac_from_Asretrieved_Pu / (Ac_from_Asretrieved_Pu_mean[:, np.newaxis, np.newaxis]+1e-4), 0.0 , 1.0)
            # Ac_from_Asretrieved_Pu_energy_normalized = Ac_from_Asretrieved_Pu / normalizer
            # Ac_from_Asretrieved_Pu_energy_normalized = np.where(Ac_from_Asretrieved_Pu_energy_normalized>1.0, 2-Ac_from_Asretrieved_Pu_energy_normalized, Ac_from_Asretrieved_Pu_energy_normalized)

            check_Ae_max = np.max(check_Ae, axis=(1,2))
            # check_Ae_mean = (np.sum(check_Ae*test_input_fg_mask, axis=(1,2))/num_fg_pixel)
            check_Ae_max_normalized = check_Ae / (check_Ae_max[:, np.newaxis, np.newaxis]+1e-4)
            # check_Ae_mean_normalized = np.clip(check_Ae / (check_Ae_mean[:, np.newaxis, np.newaxis]+1e-4), 0.0 , 1.0)

            # uniform_Au_scaled = np.ones_like(test_A1) * np.mean(test_A1, (1,2)).reshape(BatchSize, 1, 1)
            
            with open(metric_file,'a') as file0:  #写入txt文档
                print('  index      reconstructed from                                                         PSNR       SSIM     MSE    Accuracy    Efficacy', file=file0)

            batch_sum_psnr = 0
            batch_sum_ssim = 0
            batch_sum_mse = 0
            
            batch_sum_rmse = 0
            batch_sum_acc = 0
            batch_sum_eff = 0

            for index in range(BatchSize):  #BatchSize=16
                
                assert (test_input[index]).max() <= 1.0, "(test_input[index]).max() > 1.0"
                assert (test_output_Amp[index]).max() < (1+0.01), "(test_output_amp[index]).max() > 1"   #x
                assert (test_Ac[index]).max() <= 1.0, "(test_Ac[index]).max() > 1.0"

                psnr = peak_signal_noise_ratio(test_Ac_max_normalized[index], test_input[index])
                psnr = np.round(psnr, decimals=2) #保留两位小数
                myssim = structural_similarity(test_Ac_max_normalized[index], test_input[index])
                myssim = np.round(myssim, decimals=2)
                mse = mean_squared_error(test_Ac_max_normalized[index], test_input[index])
                mse = np.round(mse, decimals=2)
                acc = accuracy(test_Ac_max_normalized[index], test_input[index])
                acc = np.round(acc, decimals=2)
                eff = efficacy(test_Ac_max_normalized[index], test_input[index])
                eff = np.round(eff, decimals=2)
              
                #def save_AS_xlsx(save_path, save_OutAmp_np, save_Retriphase_np, save_Amp_np, save_PPOH_A1retrievedThreshold_np, save_img_index):
                save_AS_xlsx(ATA_amp_save_path, test_output_Amp[index], test_Ps[index], test_input[index], index)
                # save_PS_xlsx(PTA_phase_save_path, test_output_phs[index], test_Ps[index], test_input[index], A1threshold[index], index)
                Reconstructed_Dict = {
                'From Asout and Pu                              ': test_Ac_max_normalized[index],
                'From AsRetrieved and Pu                        ': Ac_from_Asretrieved_Pu_max_normalized[index],
                'From Asout and Pu                              ': Ac_from_Asoutput_Pu_max_normalized[index], 
                # 'From PsRetrieved and A1Scaled                  ': Ac_from_Asretrieved_Puscaled_max_normalized[index], 
                'From Asout and Psretrievd                      ': Ac_from_Asoutput_Psretrived_max_normalized[index], 
                'From PSRetrieved and A1retrievd                ': check_Ae_max_normalized[index], 
                }
                for key_i, value_i in Reconstructed_Dict.items():
                    # Calculate PSNR
                    psnr = peak_signal_noise_ratio(value_i, test_input[index])
                    # Calculate SSIM
                    myssim = structural_similarity(value_i, test_input[index],data_range=1.0)
                    # Calculate MSE
                    mse = mean_squared_error(value_i, test_input[index])
                    acc = accuracy(value_i, test_input[index])
                    eff = efficacy(value_i, test_input[index])

                    with open(metric_file,'a') as file0:
                        print('  {:.3f}      {}                           {:.3f}     {:.3f}    {:.3f}     {:.3f}'.format(index, key_i, psnr, myssim, mse, acc, eff), file=file0)

                    if key_i == 'From Asout and Pu                              ':
                        batch_sum_psnr += psnr
                        batch_sum_ssim += myssim
                        batch_sum_mse += mse
                        batch_sum_rmse += np.sqrt(mse)
                        batch_sum_acc += acc
                        batch_sum_eff += eff
                ###############################################################绘制输入、输出、以及多个重建图############################
                if test_times == 1:
                    MetricsEvaluation_Dict = {
                        'AE_': test_input[index], #输入，即期待幅值图 Expected Hologram
                        'AS_': test_output_Amp[index],#输出幅值图
                        'OU_': test_Ac_max_normalized[index], #经过均一化的重建图
                        'RU_': Ac_from_Asretrieved_Pu_max_normalized[index],#
                        'OR_': Ac_from_Asoutput_Psretrived_max_normalized[index], 
                        'RR_': check_Ae_max_normalized[index], 
                        }
                    subplot_index = 0
                    psnr_list, ssim_list, mse_list, acc_list, eff_list= [], [], [], [], []
                    for key_i, value_i in MetricsEvaluation_Dict.items():
                        subplot_index += 1
                        # if subplot_index == 2:
                        #     plot_subimage(total_row=4, total_column=4, sub_index=int(subplot_index), img=value_i, title=key_i, bar_min=0, bar_max=1, title_size=6)
                        # else:
                        psnr = peak_signal_noise_ratio(value_i, test_input[index])
                        psnr = np.round(psnr, decimals=2)
                        myssim = structural_similarity(value_i, test_input[index])
                        myssim = np.round(myssim, decimals=2)
                        mse = mean_squared_error(value_i, test_input[index])
                        mse = np.round(mse, decimals=2)
                        acc = accuracy(value_i, test_input[index])
                        acc = np.round(acc, decimals=2)
                        eff = efficacy(value_i, test_input[index])
                        eff = np.round(eff, decimals=2)
                        psnr_list.append(psnr)
                        ssim_list.append(myssim)
                        mse_list.append(mse)
                        acc_list.append(acc)
                        eff_list.append(eff)
                        plot_subimage(total_row=2, total_column=3, sub_index=int(subplot_index), img=value_i, title=key_i+str(psnr)+', '+str(myssim)+', '+str(mse)+', '+str(acc), bar_min=0, bar_max=1, title_size=6)
                    plt.savefig(MetricComparison_save_path + str(index)+'.png')  #保存AE AS OU RU OR RR的图像
                    plt.clf() # plt.clf () # 清除当前 figure 的所有axes，但是不关闭这个 window，所以能继续复用于其他的 plot
                    #################################################################几个不同重建图的指标曲线
                    plt.figure(figsize=(8, 6))  #尺寸
                    length_of_our_list = len(psnr_list)
                    plt.axis([1, length_of_our_list+1, 0, 100])  #左端的y轴值，代表PSNR，范围从0到100
                    plt.axis('on')
                    plt.grid(True)
                    plt.plot(range(1,length_of_our_list+1,1), psnr_list, 'r*-', ms=4, label='psnr')   #红色——————————
                    plt.xlabel('Case')  #x轴名称
                    plt.ylabel('PSNR', color='r')  #y轴名称
                    plt.tick_params(axis = 'y', labelcolor = 'r')
                    plt.legend(loc = 'upper left') #图例位置 放在左边
                    
                    ax2 = plt.twinx() ## 将plt1的x轴也分配给ax2使用
                    ax2.plot(range(1,length_of_our_list+1,1), ssim_list, 'go--', ms=4, label='ssim')  #绿色------------
                    ax2.plot(range(1,length_of_our_list+1,1), mse_list, 'bs-.', ms=4, label='mse')    #蓝色-·-·-·-·-·-·-·-·-
                    ax2.plot(range(1,length_of_our_list+1,1), acc_list, 'k<:', ms=4, label='acc')     #黑色······
                    # ax2.plot(range(1,length_of_our_list+1,1), eff_list, 'y>--', ms=4, label='eff')
                    ax2.set_ylabel('ssim & mse & acc & eff', color = 'k')
                    plt.tick_params(axis = 'y', labelcolor = 'k')
                    plt.legend(loc = 'upper right') #图例位置 放在右边
                    plt.savefig(MetricComparison_save_path + 'MetricCurves_' + str(test_times)+'.png')#绘制PSNR ssim mse acc等指标的图像
                    plt.clf()

                    plt.imshow(test_input[index],vmax=test_input[index].max(),vmin=test_input[index].min())
                    plt.axis('off')
                    plt.savefig(MetricComparison_save_path + str(index) + '_'+ 'AE.png', dpi=300,bbox_inches='tight')
                    plt.clf()
                    plt.imshow(test_output_Amp[index],vmax=test_output_Amp[index].max(),vmin=test_output_Amp[index].min())
                    plt.axis('off')
                    plt.savefig(MetricComparison_save_path + str(index) + '_'+ 'AS.png', dpi=300,bbox_inches='tight')
                    plt.clf()
                    plt.imshow(test_Ac_max_normalized[index],vmax=test_Ac_max_normalized[index].max(),vmin=test_Ac_max_normalized[index].min())
                    plt.axis('off')
                    plt.savefig(MetricComparison_save_path + str(index) + '_'+ 'Reconstructed.png', dpi=300,bbox_inches='tight')
                    plt.clf()



                ###############################################################################################
            batch_avg_psnr = batch_sum_psnr / BatchSize
            batch_avg_ssim = batch_sum_ssim / BatchSize
            batch_avg_mse = batch_sum_mse / BatchSize
            batch_avg_rmse = batch_sum_rmse / BatchSize
            batch_avg_acc = batch_sum_acc / BatchSize
            batch_avg_eff = batch_sum_eff / BatchSize

            val_avg_psnr += batch_avg_psnr
            val_avg_ssim += batch_avg_ssim
            val_avg_mse += batch_avg_mse
            val_avg_rmse += batch_avg_rmse
            val_avg_acc += batch_avg_acc
            val_avg_eff += batch_avg_eff

        assert test_times == num, "test_times is not same as num of test_loader"

        
        with open(metric_file,'a') as file0:
            print("\n", file=file0)
            print("=============================", file=file0)
            print("========= avg batch ==========", file=file0)
            avg_mae_Ac /= num
            # avg_mae_Aretri /= num
            avg_mse_Ac /= num
            # avg_mse_Aretri /= num
            avg_psnr_Ac_from_Asoutput_Pu /= num
      
            avg_mae_Ps_withPu /= num
            avg_mae_PS_withPSmean /= num
            avg_var_A1 /= num
            # avg_cosmae_Ps_and_Psretrieved /= num

            val_avg_psnr /= num
            val_avg_ssim /= num
            val_avg_mse /= num
            val_avg_rmse /= num
            val_avg_acc /= num
            val_avg_eff /= num
            
            print("batch_avg_psnr =     {} \n".format(batch_avg_psnr), file=file0)
            print("batch_avg_ssim =     {} \n".format(batch_avg_ssim), file=file0)
            print("batch_avg_rmse =     {} \n".format(batch_avg_rmse), file=file0)
            print("batch_avg_acc =     {} \n".format(batch_avg_acc), file=file0)
            print("batch_avg_eff =     {} \n".format(batch_avg_eff), file=file0)
            print("=============================", file=file0)
            print("========= avg diff ==========", file=file0)
            
            

            

            print("avg_mae_Ac =     {} \n".format(avg_mae_Ac), file=file0)
            # print("avg_mae_Aretri = {} \n".format(avg_mae_Aretri), file=file0)
            print("avg_mse_Ac =     {} \n".format(avg_mse_Ac), file=file0)
            # print("avg_mse_Aretri = {} \n".format(avg_mse_Aretri), file=file0)
            print("avg_psnr_Ac_from_Asoutput_Pu =             {} \n".format(avg_psnr_Ac_from_Asoutput_Pu), file=file0)
          
            print("avg_var_A1 =            {} \n".format(avg_var_A1), file=file0)
            # print("avg_cosmae_Ps_and_Psretrieved = {} \n".format(avg_cosmae_Ps_and_Psretrieved), file=file0)

            print("val_avg_psnr =     {} \n".format(val_avg_psnr), file=file0)
            print("val_avg_ssim =     {} \n".format(val_avg_ssim), file=file0)
            print("val_avg_mse =     {} \n".format(val_avg_mse), file=file0)
            print("val_avg_rmse =     {} \n".format(val_avg_rmse), file=file0)
            print("val_avg_acc =      {} \n".format(val_avg_acc), file=file0)
            print("val_avg_eff =      {} \n".format(val_avg_eff), file=file0)

            print("=============================")
            print("=============================")


