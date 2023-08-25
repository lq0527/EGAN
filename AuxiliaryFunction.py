import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import os
from tqdm import tqdm
import math
import tifffile as tiff
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from skimage.metrics import structural_similarity, peak_signal_noise_ratio, mean_squared_error
import cv2
from torch.autograd import Variable

from DataLoader import MyDataset, BatchSize, device, indices_train


def ASM(d=20e-3, PhsHolo=torch.zeros((1,1,100,100)), AmpHolo=torch.zeros((1,1,100,100)), Lam=6.4e-4, fs=1 / (320e-6), BatchSize=BatchSize ):  #wave propagation
    '''
    ASM (Angular Spectrum Method) is a model fomulating wave propagation between two holpgram plane
    d is positive means propagate from source hologram to target hologram
    d is negative means propagate from target hologram to source hologram
    Args:
        d: its signal determines whether the ASM or Inverse ASM is applied
           its absolute value determines the propagation distance
        PhsHolo: phase hologram
        AmpHolo: amplitude hologram
        Lam is wavelength and fs is sample frequency
        m and n are the number of meta-cells on  x and y axis of PZT
    '''
    assert Lam > 0, "Wavelength < 0"
    assert abs(d) >= 10e-3 and abs(d) <= 40e-3, "the d is out of range"
    
    Holo = torch.cat([PhsHolo, AmpHolo], dim=1)
    assert Holo.shape == torch.Size([BatchSize, 2, 100, 100]), "Holo.shape != (BS, 2, 100, 100)"
    m, n = Holo.shape[-2], Holo.shape[-1]
    assert m == 100 and n == 100, "The width and/or height of Holo is wrong"

    Phs = PhsHolo.squeeze(1)  # [BatchSize, 100, 100]
    Amp = AmpHolo.squeeze(1)  # [BatchSize, 100, 100]
    assert Phs.shape == torch.Size([BatchSize, 100, 100]), "phs.shape != torch.Size([BatchSize, 1, 100, 100])"
    assert Amp.shape == torch.Size([BatchSize, 100, 100]), "phs.shape != torch.Size([BatchSize, 1, 100, 100])"

    Re = Amp * torch.cos(Phs) # [BatchSize, 100, 100]
    Im = Amp * torch.sin(Phs) # [BatchSize, 100, 100]
    Complex = Re + 1j * Im

    # FFT
    Complex_freqdomian = torch.fft.fftshift(torch.fft.fftn(Complex))
    # Propagator
    [Freq_x, Freq_y] = torch.meshgrid((torch.arange(m)-m/2) * (fs/m), (torch.arange(n)-n/2) * (fs/n))
    #assert torch.all(1 / (Lam**2) - Freq_x**2 - Freq_y**2 >= 0) == True, "[1 / (Lam**2) - Freq_x**2 - Freq_y**2 < 0] in ASM"

    w_of_Freqx_Freqy = torch.sqrt((torch.abs(1 / (Lam**2) - Freq_x**2 - Freq_y**2)))
    Propagator = torch.zeros((m, n), dtype=torch.complex128)
    if torch.all((1 / (Lam**2) - Freq_x**2 - Freq_y**2)>= 0) == True:
        Propagator = torch.exp(1j * 2 * np.pi * w_of_Freqx_Freqy * d).to(device)
    else:
        Propagator = torch.where((1 / (Lam**2) - Freq_x**2 - Freq_y**2)<0 , 0 , np.exp(1j * 2 * np.pi * w_of_Freqx_Freqy * d )).to(device)
    # Propagator = torch.exp(1j * 2 * np.pi * w_of_Freqx_Freqy * d).to(device)
    # Transform to another hologram plane
    Complex_freqdomian2 = Complex_freqdomian * Propagator
    # IFFT
    Complex2 = torch.fft.ifftn(torch.fft.ifftshift(Complex_freqdomian2))

    Amp2 = torch.abs(Complex2)
    # Normalization Option1
    # Amp2 = Amp2 / (Amp2.max()+0.001)

    # Amp2 = Amp2 / (torch.max(torch.max(Amp2, -1)[0], -1)[0]).unsqueeze(-1).unsqueeze(-1)

    #Normalization Option2

    Max = torch.max(torch.max(Amp2, -1)[0], -1)[0] # [BatchSize]
    Min = torch.min(torch.min(Amp2, -1)[0], -1)[0] # [BatchSize]
    
    assert Max.size() == torch.Size([BatchSize]), "Max1.size() is wrong"
    Max = Max.unsqueeze(-1).unsqueeze(-1)
    Min = Min.unsqueeze(-1).unsqueeze(-1)
    
    assert Max.size() == torch.Size([BatchSize, 1, 1]), "Max.size() is wrong"
    Amp2 = ( Amp2 )/ (Max+0.000001)
    
    # Amp2 = ( Amp2-Min )/ (Max-Min)
    # assert torch.abs(Amp2.max() - 1.0) <= 0.00001, "Amp2.max() != 1.0"
    assert torch.all(torch.abs(Amp2) <= 1.0), "Amp2 out of range [0, 1]"
        
    Phs2 = torch.angle(Complex2) # output range is [-pi, pi]
    Phs2 = Phs2 - torch.floor(Phs2/(2*torch.pi)) * (2*torch.pi) # [0, 2pi]
    assert torch.all(torch.abs(Phs2) <= 2 * math.pi) and torch.all(Phs2 >= 0.0), "Phs2 is out of range [0, 2pi]"
    
    Amp2 = Amp2.unsqueeze(1)         # [BatchSize, 1, 100, 100]
    Phs2 = Phs2.unsqueeze(1)         # [BatchSize, 1, 100, 100]
    assert Amp2.shape == torch.Size([BatchSize, 1, 100, 100]), "Amp2's shape wrong"
    assert Phs2.shape == torch.Size([BatchSize, 1, 100, 100]), "Phs2's shape wrong"
    
    return Amp2, Phs2

def ASM_nullnorm(d=20e-3, PhsHolo=torch.zeros((1,1,100,100)), AmpHolo=torch.zeros((1,1,100,100)), Lam=6.4e-4, fs=1 / (320e-6), BatchSize=BatchSize ): #wave propagation without any normalizatiom
    '''
    ASM (Angular Spectrum Method) is a model fomulating wave propagation between two holpgram plane
    d is positive means propagate from source hologram to target hologram
    d is negative means propagate from target hologram to source hologram
    Args:
        d: its signal determines whether the ASM or Inverse ASM is applied
           its absolute value determines the propagation distance
        PhsHolo: phase hologram
        AmpHolo: amplitude hologram
        Lam is wavelength and fs is sample frequency
        m and n are the number of meta-cells on  x and y axis of PZT
    '''
    assert Lam > 0, "Wavelength < 0"
    assert abs(d) >= 10e-3 and abs(d) <= 40e-3, "the d is out of range"
    
    Holo = torch.cat([PhsHolo, AmpHolo], dim=1)
    assert Holo.shape == torch.Size([BatchSize, 2, 100, 100]), "Holo.shape != (BS, 2, 100, 100)"
    m, n = Holo.shape[-2], Holo.shape[-1]
    assert m == 100 and n == 100, "The width and/or height of Holo is wrong"

    Phs = PhsHolo.squeeze(1)  # [BatchSize, 100, 100]
    Amp = AmpHolo.squeeze(1)  # [BatchSize, 100, 100]
    assert Phs.shape == torch.Size([BatchSize, 100, 100]), "phs.shape != torch.Size([BatchSize, 1, 100, 100])"
    assert Amp.shape == torch.Size([BatchSize, 100, 100]), "phs.shape != torch.Size([BatchSize, 1, 100, 100])"

    Re = Amp * torch.cos(Phs) # [BatchSize, 100, 100]
    Im = Amp * torch.sin(Phs) # [BatchSize, 100, 100]
    Complex = Re + 1j * Im

    # FFT
    Complex_freqdomian = torch.fft.fftshift(torch.fft.fftn(Complex))
    # Propagator
    [Freq_x, Freq_y] = torch.meshgrid((torch.arange(m)-m/2) * (fs/m), (torch.arange(n)-n/2) * (fs/n))
    #assert torch.all(1 / (Lam**2) - Freq_x**2 - Freq_y**2 >= 0) == True, "[1 / (Lam**2) - Freq_x**2 - Freq_y**2 < 0] in ASM"

    w_of_Freqx_Freqy = torch.sqrt((torch.abs(1 / (Lam**2) - Freq_x**2 - Freq_y**2)))
    Propagator = torch.zeros((m, n), dtype=torch.complex128)
    if torch.all((1 / (Lam**2) - Freq_x**2 - Freq_y**2)>= 0) == True:
        Propagator = torch.exp(1j * 2 * np.pi * w_of_Freqx_Freqy * d).to(device)
    else:
        Propagator = torch.where((1 / (Lam**2) - Freq_x**2 - Freq_y**2)<0 , 0 , np.exp(1j * 2 * np.pi * w_of_Freqx_Freqy * d )).to(device)
    # Propagator = torch.exp(1j * 2 * np.pi * w_of_Freqx_Freqy * d).to(device)
    # Transform to another hologram plane
    Complex_freqdomian2 = Complex_freqdomian * Propagator
    # IFFT
    Complex2 = torch.fft.ifftn(torch.fft.ifftshift(Complex_freqdomian2))

    Amp2 = torch.abs(Complex2)
    # Normalization Option1
    # Amp2 = Amp2 / (Amp2.max()+0.001)

    # Amp2 = Amp2 / (torch.max(torch.max(Amp2, -1)[0], -1)[0]).unsqueeze(-1).unsqueeze(-1)

    #Normalization Option2

    # Max = torch.max(torch.max(Amp2, -1)[0], -1)[0] # [BatchSize]
    # assert Max.size() == torch.Size([BatchSize]), "Max1.size() is wrong"
    # Max = Max.unsqueeze(-1).unsqueeze(-1)
    # assert Max.size() == torch.Size([BatchSize, 1, 1]), "Max.size() is wrong"
    # Amp2 = Amp2 / (Max+0.00001)
    # # assert torch.abs(Amp2.max() - 1.0) <= 0.00001, "Amp2.max() != 1.0"
    # assert torch.all(torch.abs(Amp2) <= 1.0), "Amp2 out of range [0, 1]"
        
    Phs2 = torch.angle(Complex2) # output range is [-pi, pi]
    Phs2 = Phs2 - torch.floor(Phs2/(2*torch.pi)) * (2*torch.pi) # [0, 2pi]
    assert torch.all(torch.abs(Phs2) <= 2 * math.pi) and torch.all(Phs2 >= 0.0), "Phs2 is out of range [0, 2pi]"
    
    Amp2 = Amp2.unsqueeze(1)         # [BatchSize, 1, 100, 100]
    Phs2 = Phs2.unsqueeze(1)         # [BatchSize, 1, 100, 100]
    assert Amp2.shape == torch.Size([BatchSize, 1, 100, 100]), "Amp2's shape wrong"
    assert Phs2.shape == torch.Size([BatchSize, 1, 100, 100]), "Phs2's shape wrong"
    
    return Amp2, Phs2


def accuracy(AC, AE):  #(100, 100), np
    CoVar_like = np.sum(AE**2 * AC**2)
    VarMul_like = np.sqrt(np.sum(AE**4) * np.sum(AC**4))
    return CoVar_like / VarMul_like

def efficacy(AC, AE_nmlzd):
    FG_mask = np.where(AE_nmlzd>0.5, 1.0, 0.0)
    FG_sum = np.sum(AC**2 * FG_mask)
    Total = np.sum(AC**2)
    return FG_sum / Total

def accuracy_torch(AC, AE):  #(100, 100), np
    CoVar_like = torch.sum(AE**2 * AC**2)
    VarMul_like = torch.sqrt(torch.sum(AE**4) * torch.sum(AC**4))
    return CoVar_like / VarMul_like

def efficacy_torch(AC, AE_nmlzd):
    FG_mask = torch.where(AE_nmlzd>0.5, 1.0, 0.0)
    FG_sum = torch.sum(AC**2 * FG_mask)
    Total = torch.sum(AC**2)
    return FG_sum / Total

def gaussian(window_size, sigma):
    gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def ssim_torch(img1, img2, window_size=11, channel=1, size_average = True):

    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous()).to(device)

    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def get_amp_ratio(Ae_Batch = torch.zeros((BatchSize, 1, 100, 100))):
    assert (Ae_Batch >= 0.0).all() and (Ae_Batch <= 1.0).all(), "The Ae_Batch is out of range [0, 1]"
    Expected_Total_Energy = (torch.ones((BatchSize, 1)) * 2500).to(device) # [BatchSize, 1]
    TrueAE_Total_Energy = torch.sum(Ae_Batch, dim=(2,3))                   # [BatchSize, 1]
    if (TrueAE_Total_Energy==0.0).any():
        raise ValueError(
                "The true total energy in expected hologram is zero, please check Ae data ")
    Ratio = torch.sqrt(Expected_Total_Energy / TrueAE_Total_Energy)        # [BatchSize, 1]
    return Ratio.unsqueeze(-1).unsqueeze(-1)                               # [BatchSize, 1, 1, 1]

def get_nmlzd_amp_by_energy(Ac_Batch = torch.zeros((BatchSize, 1, 100, 100)), Ae_Batch = torch.zeros((BatchSize, 1, 100, 100))):
    Ratio = get_amp_ratio(Ae_Batch)                                        # [BatchSize, 1]
    nmlz_Ac_Batch = Ac_Batch / Ratio                                    # [BatchSize, 1, 100, 100]
    # nmlz_Ac_Batch might be predicted out of range [0, 1]
    nmlz_flip_Ac_Batch = torch.where(nmlz_Ac_Batch > 1.0, 2 - nmlz_Ac_Batch, nmlz_Ac_Batch) # [BatchSize, 1, 100, 100]
    assert (nmlz_flip_Ac_Batch>0.0).all() or (nmlz_flip_Ac_Batch<1.0).all(), "After normalizing and flipping Ac, its range still out of range [0, 1]"
    return nmlz_flip_Ac_Batch

# ####################
def Amplitude_Revised(Ac_byEnergy, Ae_scaleup):
    Ae_scaleup_max = (torch.max(torch.max(Ae_scaleup, -1)[0], -1)[0]).unsqueeze(-1).unsqueeze(-1)                # [BatchSize, 1, 1, 1]
    Ac_byEnergy_revised = torch.where(Ac_byEnergy > Ae_scaleup_max, 2*Ae_scaleup_max - Ac_byEnergy, Ac_byEnergy)
    assert (Ac_byEnergy_revised >= 0.0).all() and (Ac_byEnergy_revised <= Ae_scaleup_max).all(), "Ac_byEnergy_revised is out of range of 0~Ae_scaleup_max"
    Ac_byEnergy_nmlzd = Ac_byEnergy_revised / Ae_scaleup_max
    assert (Ac_byEnergy_nmlzd >= 0.0).all() and (Ac_byEnergy_nmlzd <= 1.0).all(), "Ac_byEnergy_nmlzd is out of range of 0~1"
    return Ac_byEnergy_nmlzd

#### not suitable for BAOH
def Amplitude_Normalization_Scaleup(Ac_Batch = torch.zeros((BatchSize, 1, 100, 100)), Ae_Batch = torch.zeros((BatchSize, 1, 100, 100)), normalized = 'None'): # normalized = 'Max' or 'Mean' or 'Energy' 
    assert (Ae_Batch >= 0.0).all() and (Ae_Batch <= 1.0).all(), "The Ae_Batch is out of range [0, 1]" 
    if normalized == 'Energy_Average':
        Total_Energy = (torch.ones((BatchSize, 1)) * 2500).to(device)                   # [BatchSize, 1]
        Ae_fg_mask = torch.where(Ae_Batch > 0.5, 1.0, 0.0)
        fg_pixel_num = torch.sum(torch.where(Ae_fg_mask==1.0, 1.0, 0.0), dim=(2,3))     # [BatchSize, 1]
        bg_pixel_num = torch.sum(torch.where(Ae_fg_mask==0.0, 1.0, 0.0), dim=(2,3))     # [BatchSize, 1]
        assert torch.sum(fg_pixel_num + bg_pixel_num) == BatchSize * 50 * 50, "The sum of foreground and background is wrong"
        Normlizer = torch.sqrt(Total_Energy / fg_pixel_num)
        Normlizer = Normlizer.unsqueeze(-1).unsqueeze(-1)
    elif normalized == 'Energy_Proportionated':
        Expected_Total_Energy = (torch.ones((BatchSize, 1)) * 2500).to(device) # [BatchSize, 1]
        TrueAE_Total_Energy = torch.sum(Ac_Batch, dim=(2,3)).to(device)        # [BatchSize, 1]
        Ratio = torch.sqrt(Expected_Total_Energy / TrueAE_Total_Energy)        # [BatchSize, 1]

        Total_Energy = (torch.ones((BatchSize, 1)) * 2500).to(device) # [BatchSize, 1]
        Ae_Batch_energy = Ae_Batch ** 2      # Ae_Batch is amplitude distribution while Ae_Batch_energy is energy distribution
        Total_Ae_Batch_energy = torch.sum(Ae_Batch_energy, dim=(2,3)) # [BatchSize, 1]
        Ae_Batch_energy_max = torch.max(torch.max(Ae_Batch_energy, -1)[0], -1)[0]          # [BatchSize, 1]
        Ae_Batch_energy_max_proportion = Ae_Batch_energy_max / Total_Ae_Batch_energy       # [BatchSize, 1]
        Ac_Batch_energy_max = Ae_Batch_energy_max_proportion * Total_Energy                # [BatchSize, 1]
        Ac_Batch_amplitude_max = torch.sqrt(Ac_Batch_energy_max)                           # [BatchSize, 1]
        Normlizer = Ac_Batch_amplitude_max.unsqueeze(-1).unsqueeze(-1)                        # [BatchSize, 1, 1, 1]
        assert (Normlizer > torch.zeros_like(Normlizer)).all(), "The Normlizer is zero"
        Energy_Ratio = Total_Energy / Total_Ae_Batch_energy
        Amplitude_Ratio = torch.sqrt(Energy_Ratio)
        assert ((Ac_Batch_amplitude_max - Amplitude_Ratio) <= 1e-3).all(), "Energy_Proportionated method is wrong"
    elif normalized == 'Max':
        Max = torch.max(torch.max(Ac_Batch, -1)[0], -1)[0] # [BatchSize, 1]
        assert Max.shape == torch.Size([BatchSize, 1]), "The shape of Max is not [BatchSize, 1]"
        Max = Max.unsqueeze(-1).unsqueeze(-1)              # [BatchSize, 1, 1, 1]
        assert Max.size() == torch.Size([BatchSize, 1, 1, 1]), "Max.size() is wrong"
        Normlizer = Max
    elif normalized == 'Mean':
        Ae_Batch_fg_mask = torch.where(Ae_Batch>0.5, 1.0, 0.0).to(device)
        num_fg_pixel = torch.sum(Ae_Batch_fg_mask, dim=(-1, -2))
        Mean = (torch.sum(Ac_Batch*Ae_Batch_fg_mask, dim=(2, 3))/num_fg_pixel)   # [BatchSize, 1]
        Normlizer = Mean.unsqueeze(-1).unsqueeze(-1)                # [BatchSize, 1, 1, 1]
    return Normlizer
# def Amp_Norm_energy(Ae_input=torch.zeros((BatchSize, 1, 100, 100)), Ac_reconstructed=torch.zeros((BatchSize, 1, 100, 100)),As=torch.zeros((BatchSize, 1, 100, 100))):
#     n = torch.sum(torch.where(As == 1.0, 1.0, 0.0))
#     Total_Energy = (torch.ones((BatchSize, 1)) * n).to(device)
#     Ae_Batch_energy = Ae_input ** 2
    

def Amplitude_Normalization(Ac_Batch = torch.zeros((BatchSize, 1, 100, 100)), Ae_Batch = torch.zeros((BatchSize, 1, 100, 100)), normalized = 'None'): # normalized = 'Max' or 'Mean' or 'Energy' 
    if normalized == 'Energy_Average':
        Total_Energy = (torch.ones((BatchSize, 1)) * 2500).to(device)
        Ae_Batch_binary = torch.where(Ae_Batch > 0.5, 1.0, 0.0)
        fg_pixel_num = torch.sum(torch.where(Ae_Batch_binary==1.0, 1.0, 0.0), dim=(2,3))     # [BatchSize, 1]
        bg_pixel_num = torch.sum(torch.where(Ae_Batch_binary==0.0, 1.0, 0.0), dim=(2,3))     # [BatchSize, 1]
        assert torch.sum(fg_pixel_num + bg_pixel_num) == BatchSize * 50 * 50, "The sum of foreground and background is wrong"
        Normlizer = torch.sqrt(Total_Energy / fg_pixel_num)
        Normlizer = Normlizer.unsqueeze(-1).unsqueeze(-1)
    elif normalized == 'Energy_Proportionated':
        Total_Energy = (torch.ones((BatchSize, 1)) * 2500).to(device) # [BatchSize, 1]
        Ae_Batch_energy = Ae_Batch ** 2      # Ae_Batch is amplitude distribution while Ae_Batch_energy is energy distribution
        Total_Ae_Batch_energy = torch.sum(Ae_Batch_energy, dim=(2,3)) # [BatchSize, 1]
        Ae_Batch_energy_max = torch.max(torch.max(Ae_Batch_energy, -1)[0], -1)[0]          # [BatchSize, 1]
        Ae_Batch_energy_max_proportion = Ae_Batch_energy_max / Total_Ae_Batch_energy       # [BatchSize, 1]
        Ac_Batch_energy_max = Ae_Batch_energy_max_proportion * Total_Energy                # [BatchSize, 1]
        Ac_Batch_amplitude_max = torch.sqrt(Ac_Batch_energy_max)                           # [BatchSize, 1]
        Normlizer = Ac_Batch_amplitude_max.unsqueeze(-1).unsqueeze(-1)                        # [BatchSize, 1, 1, 1]
        assert (Normlizer > torch.zeros_like(Normlizer)).all(), "The Normlizer is zero"

        Energy_Ratio = Total_Energy / Total_Ae_Batch_energy
        Amplitude_Ratio = torch.sqrt(Energy_Ratio)
        assert ((Ac_Batch_amplitude_max - Amplitude_Ratio) <= 1e-3).all(), "Energy_Proportionated method is wrong"
        # print("Ae_fg_pixel_num is {}".format(torch.sum(torch.where((torch.where(Ae_Batch > 0.5, 1.0, 0.0))==1.0, 1.0, 0.0), dim=(2,3))))
        # print("Total_Ae_Batch_energy is {}".format(Total_Ae_Batch_energy))
        # print("Ae_Batch_energy_max is {}".format(Ae_Batch_energy_max))
        # print("Ae_Batch_energy_max_proportion is {}".format(Ae_Batch_energy_max_proportion))
        # print("Ac_Batch_energy_max is {}".format(Ac_Batch_energy_max))
        # print("Ac_Batch_amplitude_max is {}".format(Ac_Batch_amplitude_max))
        # print("The scaleup Ae is {}".format(Normlizer * Ae_Batch))
        # print("The scaleup Ae max is {}".format(torch.max(torch.max((Normlizer * Ae_Batch), -1)[0], -1)[0]))
        # print("Ac_Batch_energy (Predicted) is {}".format(torch.max(torch.max(Ac_Batch, -1)[0], -1)[0]))
        # print("The normalized Ac_Batch is {}".format(torch.max(torch.max((Ac_Batch / Normlizer), -1)[0], -1)[0]))
        # print("The mean value between Ac_Batch_normalzd and Ae_Barch is {}".format(torch.mean(torch.abs((Ac_Batch / Normlizer) - Ae_Batch), dim=(-1,-2))))
        # Ae_Batch_numpy = Ae_Batch.squeeze(1).cpu().numpy()  # [BatchSize, 100, 100]
        # Ac_Batch_numpy_normalized = (Ac_Batch / Normlizer).squeeze(1).detach().cpu().numpy()
        # Diff = np.abs(Ae_Batch_numpy - Ac_Batch_numpy_normalized)
        # print(Diff.mean())
        # for index in range(BatchSize):
        #     plt.subplot(1, 3, 1)
        #     plt.imshow(Ae_Batch_numpy[index])
        #     plt.colorbar(fraction=0.046, pad=0.04)  # 调整colorbar的长度与image一致
        #     plt.axis('off')
        #     plt.subplot(1, 3, 2)
        #     plt.imshow(Ac_Batch_numpy_normalized[index])
        #     plt.colorbar(fraction=0.046, pad=0.04)  # 调整colorbar的长度与image一致
        #     plt.axis('off')
        #     plt.subplot(1, 3, 3)
        #     plt.imshow(Diff[index])
        #     plt.colorbar(fraction=0.046, pad=0.04)  # 调整colorbar的长度与image一致
        #     plt.axis('off')
        #     plt.savefig('./' + str(index))
        #     plt.clf()
        # exit()
        
    elif normalized == 'Max':
        Max = torch.max(torch.max(Ac_Batch, -1)[0], -1)[0] # [BatchSize, 1]
        assert Max.shape == torch.Size([BatchSize, 1]), "The shape of Max is not [BatchSize, 1]"
        Max = Max.unsqueeze(-1).unsqueeze(-1)              # [BatchSize, 1, 1, 1]
        assert Max.size() == torch.Size([BatchSize, 1, 1, 1]), "Max.size() is wrong"
        Normlizer = Max
    elif normalized == 'Mean':
        Ae_Batch_fg_mask = torch.where(Ae_Batch>0.5, 1.0, 0.0).to(device)
        num_fg_pixel = torch.sum(Ae_Batch_fg_mask, dim=(-1, -2))
        Mean = (torch.sum(Ac_Batch*Ae_Batch_fg_mask, dim=(2, 3))/num_fg_pixel)   # [BatchSize, 1]
        Normlizer = Mean.unsqueeze(-1).unsqueeze(-1)                # [BatchSize, 1, 1, 1]
    Ac_Batch_normalzd = Ac_Batch / Normlizer
    return Ac_Batch_normalzd

def Amplitude_Scaleup(Ac_Batch = torch.zeros((BatchSize, 1, 100, 100)), Ae_Batch = torch.zeros((BatchSize, 1, 100, 100)), scaleup = 'None'):   # scaleup =  'Max' or 'Mean' or 'Energy' 
    if scaleup == 'Energy_Average':
        Total_Energy = (torch.ones((BatchSize, 1)) * 2500).to(device)
        Ae_Batch_binary = torch.where(Ae_Batch > 0.5, 1.0, 0.0)
        fg_pixel_num = torch.sum(torch.where(Ae_Batch_binary==1.0, 1.0, 0.0), dim=(2,3))     # [BatchSize, 1]
        Scaleuper = torch.sqrt(Total_Energy / fg_pixel_num)
        Scaleuper = Scaleuper.unsqueeze(-1).unsqueeze(-1)
    elif scaleup == 'Energy_Proportionated':
        Total_Energy = (torch.ones((BatchSize, 1)) * 2500).to(device) # [BatchSize, 1]
        Ae_Batch_energy = Ae_Batch ** 2      # Ae_Batch is amplitude distribution while Ae_Batch_energy is energy distribution
        Total_Ae_Batch_energy = torch.sum(Ae_Batch_energy, dim=(2,3)) # [BatchSize, 1]
        Ae_Batch_energy_max = torch.max(torch.max(Ae_Batch_energy, -1)[0], -1)[0]          # [BatchSize, 1]
        Ae_Batch_energy_max_proportion = Ae_Batch_energy_max / Total_Ae_Batch_energy       # [BatchSize, 1]
        Ac_Batch_energy_max = Ae_Batch_energy_max_proportion * Total_Energy                # [BatchSize, 1]
        Ac_Batch_amplitude_max = torch.sqrt(Ac_Batch_energy_max)                           # [BatchSize, 1]
        Scaleuper = Ac_Batch_amplitude_max.unsqueeze(-1).unsqueeze(-1)                        # [BatchSize, 1, 1, 1]
        assert (Scaleuper > torch.zeros_like(Scaleuper)).all(), "The Normlizer is zero"
        Energy_Ratio = Total_Energy / Total_Ae_Batch_energy
        Amplitude_Ratio = torch.sqrt(Energy_Ratio)
        assert ((Ac_Batch_amplitude_max - Amplitude_Ratio) <= 1e-3).all(), "Energy_Proportionated method is wrong"
    elif scaleup == 'Max':
        Max = torch.max(torch.max(Ac_Batch, -1)[0], -1)[0] # [BatchSize, 1]
        assert Max.shape == torch.Size([BatchSize, 1]), "The shape of Max is not [BatchSize, 1]"
        Max = Max.unsqueeze(-1).unsqueeze(-1)              # [BatchSize, 1, 1, 1]
        assert Max.size() == torch.Size([BatchSize, 1, 1, 1]), "Max.size() is wrong"
        Scaleuper = Max
    elif scaleup == 'Mean':
        Ae_Batch_fg_mask = torch.where(Ae_Batch>0.5, 1.0, 0.0).to(device)
        num_fg_pixel = torch.sum(Ae_Batch_fg_mask, dim=(-1, -2))
        Mean = (torch.sum(Ac_Batch*Ae_Batch_fg_mask, dim=(2, 3))/num_fg_pixel)   # [BatchSize, 1]
        Scaleuper = Mean.unsqueeze(-1).unsqueeze(-1)                # [BatchSize, 1, 1, 1]
    # elif scaleup == 'Mean':
        
    Ae_Batch_scaleuped = Ae_Batch * Scaleuper
    return Ae_Batch_scaleuped


def Amplitude_Revised(AmpHolo_normlzd_byEnergy):
    # print("AmpHolo_normlzd_byEnergy.min() = {}".format(AmpHolo_normlzd_byEnergy.min()))
    # print("AmpHolo_normlzd_byEnergy.max() = {}".format(AmpHolo_normlzd_byEnergy.max()))
    AmpHolo_normlzd_byEnergy_revised = torch.where(AmpHolo_normlzd_byEnergy > 1.0, 2-AmpHolo_normlzd_byEnergy, AmpHolo_normlzd_byEnergy)
    # print("AmpHolo_normlzd_byEnergy_revised.min() = {}".format(AmpHolo_normlzd_byEnergy_revised.min()))
    # print("AmpHolo_normlzd_byEnergy_revised.max() = {}".format(AmpHolo_normlzd_byEnergy_revised.max()))
    # exit()
    assert AmpHolo_normlzd_byEnergy_revised.min() >= 0.0 and AmpHolo_normlzd_byEnergy_revised.max() <= 1.0, "AmpHolo_normlzd_byEnergy_revised is out of range [0,1]"
    return AmpHolo_normlzd_byEnergy_revised

# #################### This is for pool collection (not related to ENN )


def AeA1Collection(iter_i, savepath, data):
    # The size of collected AeA1 dataset is the same as training dataset
    # The arg data is supposed to be in the range of [0, 1] denoted as Ae&A1
    data_save_path = savepath + os.sep + 'StackedAeA1'
    isExist = os.path.exists(data_save_path)
    if not isExist:
        os.makedirs(data_save_path)
    assert data.shape == torch.Size([BatchSize, 2, 100, 100]), "data.shape != torch.Size([BatchSize, 2, 100, 100])"
    assert data.max() <= 1.0 and data.min() >= 0.0, "data is out of range [0, 1] in AeA1Collection function"
    assert ((data[:,0,:,:] == 1) + (data[:,0,:,:] == 0)).all(), "Ae is not binary image"
    data = data.detach().cpu().numpy() # [BatchSize, 2, 100, 100]
    for index in range(BatchSize):  # data[index,:,:,:].shape = (2, 100, 100)
        np.save(data_save_path + "/" + str((iter_i*BatchSize + index)) + ".npy", data[index,:,:,:])

def AePsCollection(iter_i, savepath, data):
    # The size of collected AePs dataset is the same as training dataset
    # data[:, 0] is supposed to be in the range of [0, 1]
    # data[:, 1] is supposed to be in the range of [0, 2pi]
    data_save_path = savepath + os.sep + 'StackedAePs'
    isExist = os.path.exists(data_save_path)
    if not isExist:
        os.makedirs(data_save_path)
    assert data.shape == torch.Size([BatchSize, 2, 100, 100]), "data.shape != torch.Size([BatchSize, 2, 100, 100])"
    assert data[:,0].min() >= 0.0 and data[:,0].max() <= 1.0, "Amp is out of range [0, 1]"
    assert ((data[:,0] == 1) + (data[:,0] == 0)).all(), "Ae is not a binary image"
    assert data[:,1].min() >= 0.0 and data[:,1].max() <= 2*torch.pi, "Phs is out of range [0, 2pi]"
    data = data.detach().cpu().numpy() # [BatchSize, 2, 100, 100]
    for index in range(BatchSize):  # data[index,:,:,:].shape = (2, 100, 100)
        np.save(data_save_path + "/" + str((iter_i*BatchSize + index)) + ".npy", data[index,:,:,:])

def AePsAbinaryCollection(iter_i, savepath, data):
    # The size of collected AePsAbinary dataset is the same as training dataset
    # data[:, 0] (Ae) is supposed to be in the range of [0, 1]
    # data[:, 1] (Ps) is supposed to be in the range of [0, 2pi]
    # data[:, 2] (Abinary) is supposed to be in the range of [0, 1]
    data_save_path = savepath + os.sep + 'StackedAePsA1binary'
    isExist = os.path.exists(data_save_path)
    if not isExist:
        os.makedirs(data_save_path)
    assert data.shape == torch.Size([BatchSize, 3, 100, 100]), "data.shape != torch.Size([BatchSize, 3, 100, 100])"
    assert data[:,0].min() >= 0.0 and data[:,0].max() <= 1.0, "Amp is out of range [0, 1]"
    assert ((data[:,0] == 1) + (data[:,0] == 0)).all(), "Ae is not a binary image"
    assert data[:,1].min() >= 0.0 and data[:,1].max() <= 2*torch.pi, "Phs is out of range [0, 2pi]"
    assert data[:,2].min() >= 0.0 and data[:,2].max() <= 1.0, "A1binary is not binary"
    assert ((data[:,2] == 1) + (data[:,2] == 0)).all(), "Abinary is not a binary image"
    data = data.detach().cpu().numpy()
    for index in range(BatchSize):  # data[index,:,:,:].shape = (3, 100, 100)
        np.save(data_save_path + "/" + str((iter_i*BatchSize + index)) + ".npy", data[index,:,:,:])
        
def ExperienceCollection(epoch_i, iter_i, savepath, data, training_dataset, capacity):
    # The size of collected experiences depends on arg capacity
    # data (Ac, Psoutput) or (Ae, Psretrieved) is supposed to be in the range of [0, 1] and [0, 2pi] respectively
    data_save_path = savepath + os.sep + training_dataset # training_dataset = Pool1: (Ac, Psoutput) or training_dataset = Pool2: (Ae, Psretrieved)
    isExist = os.path.exists(data_save_path)
    if not isExist:
        os.makedirs(data_save_path)
    data = data.detach().cpu().numpy()  # [BatchSize, 2, 100, 100]
    assert np.all(data[:,0] > (0-1e-3)) and np.all(data[:,0] < (1+1e-3)), "data Amp is out of range"
    assert np.all(data[:,1] > (0-1e-3)) and np.all(data[:,1] < (2*np.pi+1e-3)), "data Phs is out of range"
    if training_dataset == 'Pool2':
        assert ((data[:,0] == 1) + (data[:,0] == 0)).all(), "Ae is not a binary image"
    for index in range(BatchSize):  # data[index,:,:,:].shape = (2, 100, 100)
        np.save(data_save_path + "/" + str((epoch_i*len(indices_train) + iter_i*BatchSize + index) % capacity) + ".npy", data[index,:,:,:])
        
def DataCollection(save_root_path, save_folder_name, data, batch_i, capacity):
    '''
    This function is used for data collection.
    The data will be saved as .npy format.
    save_root_path: '/public/home/zhongchx/Dataset_2D/ExperiencePool_' + TIME + "_" + NET + "_" + NOTE
    save_folder_name: 'StackedAeA1' or 'StackedAePs' or 'StackedAePsA1binary' or 'ExperiencePool1' or 'ExperiencePool2'
    data: The data need to be save with size of [BatchSize, *, 100, 100]
    batch_i: the batch_ith to save
    capacity: When saved data is up to capacity, the old data will be substituted by the news
    '''
    data_save_path = save_root_path + os.sep + save_folder_name #os.sep:跨平台的文件分隔符
    isExist = os.path.exists(data_save_path)  
    if not isExist:
        os.makedirs(data_save_path)
    data = data.detach().cpu().numpy() #detach(): 返回一个新的Tensor，但返回的结果是没有梯度的 cpu() :把gpu上的数据转到 cpu 上。 numpy() :将tensor格式转为 numpy
    if save_folder_name == 'StackedAeA1':
        # assert ((data[:,0] == 1) + (data[:,0] == 0)).all(), "Ae is not binary image"
        assert data[:,1].min() >= 0.0 and data[:,1].max() <= 1.0, "A1 is out of range [0,1]"
    elif save_folder_name == 'StackedAePs':
        # assert ((data[:,0] == 1) + (data[:,0] == 0)).all(), "Ae is not binary image"
        assert np.all(np.abs(data[:,1]) <= 2 * math.pi), "Ps is out of range [0,2pi]"
    elif save_folder_name == 'StackedAePsA1binary':
        # assert ((data[:,0] == 1) + (data[:,0] == 0)).all(), "Ae is not binary image"
        assert np.all(np.abs(data[:,1]) <= 2 * math.pi), "Ps is out of range [0,2pi]"
        assert ((data[:,2] == 1) + (data[:,2] == 0)).all(), "A1binary is not binary image"
    elif save_folder_name == 'ExperiencePool1':
        assert data[:,0].min() >= 0.0 and data[:,0].max() <= 1.0, "Ac is out of range [0, 1]"
        assert np.all(np.abs(data[:,1]) <= 1), "As is out of range [0,1]"
    elif save_folder_name == 'ExperiencePool2':
        assert ((data[:,0] == 1) + (data[:,0] == 0)).all(), "Ae is not binary image"
        assert np.all(np.abs(data[:,1]) <= 1), "As is out of range [0,1]"  #data[:,1]第二列所有数据 
    else:
        print("Some mistakes happen during Data Collection")
        exit()
    for i in range(BatchSize):
        np.save(data_save_path + "/" + str((batch_i*BatchSize + i) % capacity) + ".npy", data[i,:,:,:])

#########################
def plot_subimage(total_row, total_column, sub_index, img, title, bar_min=-1, bar_max=-1, title_size=6):
    if bar_min==-1 or bar_max==-1:
        bar_min = img.min()
        bar_max = img.max()
    plt.subplot(total_row, total_column, sub_index)
    plt.gca().set_title(title)
    plt.gca().title.set_size(title_size)
    plt.subplots_adjust(wspace =0.5, hspace =0.5) #调整子图间距
    plt.imshow(img, vmin=bar_min, vmax=bar_max)
    plt.colorbar(fraction=0.046, pad=0.04)  # 调整colorbar的长度与image一致
    plt.axis('off')


def save_AS_xlsx(save_path, save_OutAmp_np, save_Retriphase_np, save_Amp_np,  save_img_index):
    
    writer = pd.ExcelWriter(save_path + str(save_img_index) + '.xlsx')
    
    rows, cols = save_OutAmp_np.shape
    data = []
    for i in range(rows):
        for j in range(cols):
            element = save_OutAmp_np[i, j]
            row_col = (i, j)
            data.append(row_col + (element,))
            
    df = pd.DataFrame(data, columns=["Row", "Column", "Value"])
    df.to_excel(writer, "ATA_OutAmp1", float_format='%.5f')

    

    save_OutAmp_np = save_OutAmp_np.reshape(10000)
    # assert np.all(save_OutAmp_np >= 0) and np.all(save_OutAmp_np <= 1), "save_OutAmp_np is out of [0, 1]"  #x
    save_OutAmp_list = pd.DataFrame(save_OutAmp_np, columns=["ATA_OutAmp"])
    save_OutAmp_list.to_excel(writer, "ATA_OutAmp", float_format='%.5f')

    # save_Retriphase_np = save_Retriphase_np.reshape(10000)
    # assert np.all(save_Retriphase_np >= 0) and np.all(save_Retriphase_np <= 2*np.pi), "save_Retriphase_np is out of [0, 2pi]"
    # save_Retriphase_list = pd.DataFrame(save_Retriphase_np, columns=["ATA_Retriphase"])
    # save_Retriphase_list.to_excel(writer, "ATA_Retriphase", float_format='%.5f')

    save_Amp_np = save_Amp_np.reshape(10000)
    assert np.all(save_Amp_np >= 0) and np.all(save_Amp_np <= 1), "save_Amp_np is out of [0, 1]"
    save_Amp_list = pd.DataFrame(save_Amp_np, columns=["Expected_AmpHolo"])
    save_Amp_list.to_excel(writer, "Expected_AmpHolo", float_format='%.5f')

    # save_PPOH_A1retrievedThreshold = save_PPOH_A1retrievedThreshold_np.reshape(10000)
    # assert ((save_PPOH_A1retrievedThreshold_np == 0) + (save_PPOH_A1retrievedThreshold_np == 1)).all(), "save_PPOH_A1retrievedThreshold is out 0 or 1"
    # save_PPOH_phs_list = np.where(save_PPOH_A1retrievedThreshold==1, save_Retriphase_np/(2*np.pi)*255, (save_Retriphase_np*0+1)*500)
    # save_PPOH_phs_list = pd.DataFrame(save_PPOH_phs_list, columns=["PPOH_phase"])
    # save_PPOH_phs_list.to_excel(writer, "PPOH_phase", float_format='%.1f')

    writer.save()
    writer.close()

def save_AEPC_A1PS_ACPC_xlsx(save_path, AE, PC_out, A1, AS, AC, PC_rec, save_img_index):
    
    writer = pd.ExcelWriter(save_path + str(save_img_index) + '.xlsx')

    save_AE = AE.reshape(10000)
    save_AE_list = pd.DataFrame(save_AE, columns=["AE"])
    save_AE_list.to_excel(writer, "AE", float_format='%.5f')

    save_PC_out = PC_out.reshape(10000)
    save_PC_out_list = pd.DataFrame(save_PC_out, columns=["PC_out"])
    assert np.all(save_PC_out >= 0) and np.all(save_PC_out <= 2*np.pi), "save_PC_out is out of [0, 2pi]"
    save_PC_out_list.to_excel(writer, "PC_out", float_format='%.5f')

    save_A1 = A1.reshape(10000)
    save_A1_list = pd.DataFrame(save_A1, columns=["A1"])
    save_A1_list.to_excel(writer, "A1", float_format='%.5f')

    save_AS = AS.reshape(10000)
    save_PS_list = pd.DataFrame(save_AS, columns=["PS"])
    assert np.all(save_AS >= 0) and np.all(save_AS <= 1), "save_AS is out of [0, 1]"
    save_PS_list.to_excel(writer, "PS", float_format='%.5f')

    save_AC = AC.reshape(10000)
    save_AC_list = pd.DataFrame(save_AC, columns=["AC"])
    save_AC_list.to_excel(writer, "AC", float_format='%.5f')

    save_PC_rec = PC_rec.reshape(10000)
    save_PC_rec_list = pd.DataFrame(save_PC_rec, columns=["PC_rec"])
    assert np.all(save_PC_rec >= 0) and np.all(save_PC_rec <= 2*np.pi), "save_PC_rec is out of [0, 2pi]"
    save_PC_rec_list.to_excel(writer, "PC_rec", float_format='%.5f')

    writer.save()
    writer.close()

def write_list_to_elsx(My_writer, List, Columns_Name, Sheet_Name):
        data = pd.DataFrame(List, columns=[Columns_Name])
        data.to_excel(My_writer, Sheet_Name, float_format='%.5f')

###########################################################绘图可视化##################################################
def single_curve(NOTE, epochs, list=[], label='', xlabel='', ylabel='', savename=''):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1,len(list)+1,1), list, c='b', marker='v', ms=4, label=label)
    plt.xticks(range(1, len(list)+1, 1), rotation = 90)
    plt.xlim(0, len(list)+1)
    plt.ylim(min(list)-0.2, max(list)+0.2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig('/public/home/liuqing2022/hologram/Results/LossDegragation/PlotCurve/' + NOTE + '_' + str(epochs) + savename)
    plt.clf()

def double_curves(NOTE, epochs, list1=[], list2=[], label1='', label2='', xlabel='', ylabel='', savename=''):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1,len(list1)+1,1), list1, c='red', marker='s', ms=4, label=label1)
    plt.plot(range(1,len(list2)+1,1), list2, c='g', marker='o', label=label2)
    plt.xticks(range(1, max(len(list1), len(list2)) + 1, 1), rotation = 90)
    plt.xlim(0, max(len(list1), len(list2)) + 1)
    plt.ylim(min(min(list1), min(list2)) - 0.2, max(max(list1), max(list2)) + 0.2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig('/public/home/liuqing2022/hologram/Results/LossDegragation/PlotCurve/' + NOTE + '_' + str(epochs) + savename)
    plt.clf()

def list_generator(mean, dis, number):
    return np.random.normal(mean, dis * dis, number)

y1 = list_generator(0.8531, 0.0956, 70)
y2 = list_generator(0.8631, 0.0656, 80)
y3 = list_generator(0.8731, 0.1056, 90)
y4 = list_generator(0.8831, 0.0756, 100)
y5 = list_generator(0.8831, 0.0756, 110)
y1 = pd.Series(np.array(y1))
y2 = pd.Series(np.array(y2))
y3 = pd.Series(np.array(y3))
y4 = pd.Series(np.array(y4))
y5 = pd.Series(np.array(y5))

def MyBoxplot(Note, PSNRlist_for_Boxplot, SSIMlist_for_Boxplot, MSElist_for_Boxplot, ACClist_for_Boxplot, EFFlist_for_Boxplot):
    data = pd.DataFrame({"PSNR": PSNRlist_for_Boxplot, "SSIM": SSIMlist_for_Boxplot, "MSE": MSElist_for_Boxplot, "ACC": ACClist_for_Boxplot, "EFF": EFFlist_for_Boxplot,})
    data.boxplot()
    plt.ylabel("Value")
    plt.xlabel("Metrics") # 我们设置横纵坐标的标题。
    plt.savefig('./Results/Visulaization/' + Note + '/MetricsBoxplot')

def MyCompareBoxplot(Note, PSNRlist_for_Boxplot_1, SSIMlist_for_Boxplot_1, MSElist_for_Boxplot_1, ACClist_for_Boxplot_1, EFFlist_for_Boxplot_1, PSNRlist_for_Boxplot_2, SSIMlist_for_Boxplot_2, MSElist_for_Boxplot_2, ACClist_for_Boxplot_2, EFFlist_for_Boxplot_2):
    data = []
    data.append(PSNRlist_for_Boxplot_1)
    data.append(PSNRlist_for_Boxplot_2)
    #箱型图名称
    labels = ["OutputPs with Au", "RetrievedPs with Au"]
    # 箱型图的颜色 RGB （均为0~1的数据）
    colors = [(202/255.,96/255.,17/255.), (255/255.,217/255.,102/255.)]
    bplot = plt.boxplot(data, patch_artist=True,labels=labels,positions=(1,1.4),widths=0.3) 
    #将三个箱分别上色
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    
    # x_position=[1]
    # x_position_fmt=["PSNR"]
    # plt.xticks([i + 0.8 / 2 for i in x_position], x_position_fmt)
    plt.ylabel('PSNR')
    plt.grid(linestyle="--", alpha=0.3)  #绘制图中虚线 透明度0.3

    ax2 = plt.twinx()

    data2 = []
    data2.append(SSIMlist_for_Boxplot_1)
    data2.append(SSIMlist_for_Boxplot_2)
    bplot2 = ax2.boxplot(data2, patch_artist=True, labels=labels,positions=(2.5,2.9),widths=0.3) 
    for patch, color in zip(bplot2['boxes'], colors):
        patch.set_facecolor(color)
    
    data3 = []
    data3.append(MSElist_for_Boxplot_1)
    data3.append(MSElist_for_Boxplot_2)
    bplot3 = ax2.boxplot(data3, patch_artist=True, labels=labels,positions=(4,4.4),widths=0.3) 
    for patch, color in zip(bplot3['boxes'], colors):
        patch.set_facecolor(color)

    data4 = []
    data4.append(ACClist_for_Boxplot_1)
    data4.append(ACClist_for_Boxplot_2)
    bplot4 = ax2.boxplot(data4, patch_artist=True, labels=labels,positions=(5.5,5.9),widths=0.3) 
    for patch, color in zip(bplot4['boxes'], colors):
        patch.set_facecolor(color)

    data5 = []
    data5.append(EFFlist_for_Boxplot_1)
    data5.append(EFFlist_for_Boxplot_2)
    bplot5 = ax2.boxplot(data5, patch_artist=True, labels=labels,positions=(7,7.4),widths=0.3) 
    for patch, color in zip(bplot5['boxes'], colors):
        patch.set_facecolor(color)

    x_position=[1, 2.5, 4, 5.5, 7]
    x_position_fmt=["PSNR", "SSIM", "RMSE", "Accuracy", "Efficacy"]
    plt.xticks([i + 0.8 / 2 for i in x_position], x_position_fmt)
    plt.ylabel('percent (%)')
    plt.grid(linestyle="--", alpha=0.3)  #绘制图中虚线 透明度0.3
    plt.legend(bplot['boxes'],labels,loc='lower right')
    
    plt.savefig('./Results/Visulaization/' + Note + '/MetricsBoxplot_comparison')
    plt.clf()

def Normalize_AmpHolo(AmpHolo, method):
    if method == 'min_max':
        BS = AmpHolo.shape[0]
        assert AmpHolo.shape[1] == 1, "The channel of AmpHolo is not 1, which is {}".format(AmpHolo.shape[1])
        AmpHolo_min = (torch.min(torch.min(torch.min(AmpHolo, dim=-1)[0], dim=-1)[0], dim=-1)[0]).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        AmpHolo_max = (torch.max(torch.max(torch.max(AmpHolo, dim=-1)[0], dim=-1)[0], dim=-1)[0]).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        assert AmpHolo_min.shape == torch.Size([BS, 1, 1, 1]) and AmpHolo_max.shape == torch.Size([BS, 1, 1, 1]), "The shape of AmpHolo_min ({})/max ({}) is wrong".format(AmpHolo_min.shape, AmpHolo_max.shape) 
        AmpHolo_nmlzd = (AmpHolo - AmpHolo_min) / (AmpHolo_max - AmpHolo_min)
        assert AmpHolo_nmlzd.min() >= 0.0 and AmpHolo_nmlzd.max() <= 1.0, "AmpHolo_nmlzd ranges of ({}, {}) not (0, 1)".format(AmpHolo_nmlzd.min(), AmpHolo_nmlzd.max())
    elif method == 'energy':
        BS, Nx, Ny = AmpHolo.shape[0], AmpHolo.shape[2], AmpHolo.shape[3]
        assert AmpHolo.shape[1] == 2, "The channel of AmpHolo is not 2, which is {}".format(AmpHolo.shape[1])
        Target_AmpHolo = AmpHolo[:,0].unsqueeze(1)
        Recon_AmpHolo = AmpHolo[:,1].unsqueeze(1)
        assert Target_AmpHolo.shape == torch.Size([BS, 1, Nx, Ny]) and Recon_AmpHolo.shape == torch.Size([BS, 1, Nx, Ny]), "The shape of Target_AmpHolo and Recon_AmpHolo are wrong, they are {} and {} ".format(Target_AmpHolo.shape, Recon_AmpHolo.shape)  
        Target_eng = torch.sum(Target_AmpHolo**2, dim=(-1, -2, -3), keepdim=True)
        Recon_eng = torch.sum(Recon_AmpHolo**2, dim=(-1, -2, -3), keepdim=True)
        amp_ratio = torch.sqrt(Recon_eng / Target_eng)
        AmpHolo_nmlzd = Recon_AmpHolo / amp_ratio
        assert AmpHolo_nmlzd.min() >= 0.0, "AmpHolo_nmlzd is smaller than 0, whose minimum is {}".format(AmpHolo_nmlzd.min())
    else:
        print("The normalization method in Normalize_AmpHolo function of amplitude hologram is not been specified! ")
    return AmpHolo_nmlzd
# def plotmybox(list_num, lis_dict):
#     color = [(255/255.,217/255.,102/255.)]
#     boxposition = 0
#     x_position_fmt = []
#     for key_i, value_i in lis_dict.items():
#         boxposition += 1
#         bplot = plt.boxplot(value_i, patch_artist=True,positions=(boxposition),widths=0.3)
#         bplot['boxes'].set_facecolor(color)
#         x_position_fmt.append(key_i)
#     x_position=[1,2,3,4,5]
#     plt.xticks([i + 0.8 / 2 for i in x_position], x_position_fmt)
#     plt.ylabel('percent (%)')
#     plt.grid(linestyle="--", alpha=0.3)  #绘制图中虚线 透明度0.3
#     plt.legend(bplot['boxes'],labels,loc='lower right')  #绘制表示框，右下角绘制
#     plt.savefig(fname="pic.png",figsize=[10,10])  
#     plt.show()


#IASA(NOTE, ATA_Amp=test_output_Amp[0].unsqueeze(1), Expected_Amp=test_input[0].unsqueeze(1).to(device), Z_distance=20e-3, Lam=6.4e-4, fs=1/(50e-3/50), iterations=10)###test_output_phs?
def IASA(NOTE, ATA_Amp=torch.zeros((1,1,100,100)), Expected_Amp=torch.zeros((1,1,100,100)), Z_distance=20e-3, Lam=6.4e-4, fs=1/(50e-3/50), iterations=10):#######绘制每一次IASA图像以及对应PSNR值
    IASA_save_path = "./Results/Visulaization/" + NOTE + "/continueIASA/"
    isExist = os.path.exists(IASA_save_path)
    if not isExist:
        os.makedirs(IASA_save_path)
    Pu = torch.zeros_like(ATA_Amp)
    # Au = torch.ones_like(PTA_Phs)
    for i in range(iterations):

        # def plot_subimage(total_row, total_column, sub_index, img, title, bar_min=-1, bar_max=-1, title_size=6):
        plot_subimage(4, 5, i//5*5*2 + i%5 + 1, ATA_Amp.squeeze(0).squeeze(0).cpu().numpy(), str(i)+'_As', 0.0, 1)
        Ac, Pc = ASM(d=Z_distance, PhsHolo= Pu, AmpHolo=ATA_Amp, Lam=Lam, fs=fs, BatchSize=1)
        Ac_max = torch.max(torch.max(Ac, -1)[0], -1)[0]
        # print('Ac_max is ' , Ac_max)  ####print
        Ac_img_normlzd = (Ac / Ac_max).squeeze(0).squeeze(0).cpu().numpy()
        Ae_img = Expected_Amp.squeeze(0).squeeze(0).cpu().numpy()
        # Calculate PSNR
        psnr = peak_signal_noise_ratio(Ac_img_normlzd, Ae_img)
        # print(psnr)
        psnr = int(psnr * 100) / 100
        # Calculate SSIM
        ssim = structural_similarity(Ac_img_normlzd, Ae_img, multichannel=True)
        # Calculate MSE
        mse = mean_squared_error(Ac_img_normlzd, Ae_img)
        plot_subimage(4, 5, 5 + i//5*5*2 + i%5 + 1, Ac.squeeze(0).squeeze(0).cpu().numpy(), str(i)+' PSNR='+str(psnr))
        ATA_Amp, P1 = ASM(d=-Z_distance, PhsHolo=Pc, AmpHolo=Expected_Amp, Lam=Lam, fs=fs, BatchSize=1)
    plt.savefig(IASA_save_path  + str(iterations) + 'iterations')
    plt.clf()


def saltpepper_noise(image, proportion):
    '''
    This function is used to add saltpepper noise to an image
    Args:
        image: initial pure image
        proportion: the ratio of noise
    Return:
        image_copy: saltpapper_out image
        sp_noise_plate: saltpapper noise
    '''
    image_copy = image.copy()
    img_Y, img_X = image.shape # 求得其高宽
    X = np.random.randint(img_X,size=(int(proportion*img_X*img_Y),)) # 噪声点的 X 坐标
    Y = np.random.randint(img_Y,size=(int(proportion*img_X*img_Y),)) # 噪声点的 Y 坐标
    image_copy[Y, X] = np.random.choice([0, 255], size=(int(proportion*img_X*img_Y),)) # 噪声点的坐标赋值
    sp_noise_plate = np.ones_like(image_copy) * 127 # 噪声点的坐标赋值
    sp_noise_plate[Y, X] = image_copy[Y, X]  # 将噪声给噪声容器
    return image_copy, sp_noise_plate # 这里也会返回噪声，注意返回值

def gaussian_noise(img, fg_mask, bg_mask, mean, sigma):
    '''
    This function is used to add Gauss Noise to an image
    传入:
        img   :  原图
        mean  :  均值
        sigma :  标准差
    返回:
        gaussian_out : 噪声处理后的图片
        noise        : 对应的噪声
    '''
    if img.max() == 255:
        img = img / 255 # 将图片灰度标准化
    elif img.max() == 1:
        img = img
    else:
        print(img.max() != 255 and img.max() != 1)
    noise_fg = np.random.normal(mean, sigma, img.shape) # 为前景产生高斯 noise
    noise_bg = np.random.normal(mean, sigma, img.shape) # 为背景产生高斯 noise
    ################################ Add in 22/07/15 ################################
    # The decreasement of foreground approximates increasement of background (From energy perspective)
    # i = 1
    # while np.abs(np.sum(np.abs(noise_fg)*fg_mask) / np.sum(np.abs(noise_bg)*bg_mask) - 1) > 0.2:
    #     noise_bg = np.random.normal(mean, sigma, img.shape) # 重新为背景产生高斯 noise
    #     i += 1
    #     print(i)
    noise_fg_abs = np.abs(noise_fg)
    noise_bg_abs = np.abs(noise_bg)
    fg_noise_negative = noise_fg_abs * fg_mask * (-1)
    bg_noise_positive = noise_bg_abs * bg_mask
    gaussian_out = img + fg_noise_negative + bg_noise_positive
    ################################################################################
    # gaussian_out = img + noise # 将噪声和图片叠加
    # gaussian_out = np.clip(gaussian_out, 0, 1) # 将超过 1 的置 1，低于 0 的置 0
    assert gaussian_out.min() >= 0.0 and gaussian_out.max() <= 1.0, "gaussian_out is out of range [0, 1]"
    gaussian_out = np.uint8(gaussian_out*255) # 将图片灰度范围的恢复为 0-255
    # noise = np.uint8(noise*255) # 将噪声范围搞为 0-255
    return gaussian_out, (fg_noise_negative+bg_noise_positive) # 这里也会返回噪声，注意返回值

def UpdateA1byKxAretri(A1):
    k = 0.5 # This parameter is from diff-PAT (Scientific Report), k = 0.1, 0.5, 0.8, 2
    A1 = A1 + k * (torch.ones_like(A1) - A1)
    assert A1.shape == torch.Size([BatchSize, 1, 100, 100]), "The shape of A1 is not (BatchSize, 1, 100, 100)"
    print("Ae and its coresponding updated A1 from retrieved A1 are stacked and saved")
    return A1

def UpdateA1byPPOH(A1, threshold = 0.25):
    x = torch.ones_like(A1).to(device)
    y = torch.zeros_like(A1).to(device)
    A1 = torch.where(A1>threshold, x, y)
    assert A1.shape == torch.Size([BatchSize, 1, 100, 100]), "The shape of A1 is not (BatchSize, 1, 100, 100)"
    print("Ae and its coresponding binary A1 are stacked and saved")
    return A1



'''
def PrepareUsedFolders(METHOD, Continue, TIME, NET, Method, Update_version, Lossfunc, PPOH, net):
    ############################ Initial Experience Pools ###############################
    # METHOD: SSVL                 no initial any pool
    # METHOD: TwoPoolRL            Pool1: {(Ac, Ps_netout}; Pool2: {(Ae, Ps_retrieved)}
    # METHOD: IL                   Pool1: {(Ac, Ps_netout}; Pool2: {(Ae, Ps_retrieved)}; Pool_AeA1: {(Ae, A1)}
    # METHOD: I-RL                 Pool_AePs: {(Ae, Ps)}

    NOTE = Method + '_' + Update_version + '_L' + Lossfunc + '_Continue' + str(Continue) + '_PPOH' + str(PPOH)
    ExperiencePool = '/public/home/zhongchx/Dataset_2D/ExperiencePool_' + TIME + "_" + NET + "_" + NOTE
    if not os.path.exists(ExperiencePool):
        os.makedirs(ExperiencePool)
            
    if METHOD == 'TwoPoolRL':
        Capacity_Pool1 = 64  # (64 = 16 * 4), 1600 is the training set
        Capacity_Pool2 = 64  # (64 = 16 * 4)
        pool1_epochs = 1     # training epoch on Experience pool 1
        pool2_epochs = 1     # training epoch on Experience pool 2

    elif METHOD == 'IL':
        Capacity_Pool1 = 64  # (64 = 16 * 4), 1600 is the training set
        Capacity_Pool2 = 64  # (64 = 16 * 4)
        Capacity_Pool_AeA1 = len(train_loader)
        pool1_epochs = 1     # training epoch on Experience pool 1
        pool2_epochs = 1     # training epoch on Experience pool 2
        for iter_i, data in enumerate(train_loader):
            Ae = data
            Au = torch.ones_like(Ae)
            assert Ae.shape == torch.Size([BatchSize, 1, 100, 100]), "Ae.shape is not (BatchSize, 1, 100, 100)"
            AeA1Collection(iter_i, ExperiencePool, data=torch.cat((Ae, Au), 1))
        Dataset_Name = 'StackedAeA1'
        AeAu_dataset = MyDataset(Dataset_Name=Dataset_Name, TIME=TIME, NET=NET, NOTE=NOTE)
        train_loader = Data.DataLoader(dataset = AeAu_dataset, batch_size = BatchSize, num_workers=nw, drop_last=True,)

    elif METHOD == 'I-RL':
        if not PPOH:
            Capacity_Pool_AePs = len(train_loader)
        elif PPOH:
            Capacity_Pool_AePsA1binary = len(train_loader)
        for iter_i, data in enumerate(train_loader):
            Ae = data
            Ps = net(Ae.to(device))
            Ac, Pc = ASM(d=20e-3, PhsHolo=Ps, AmpHolo=torch.ones_like(Ps))
            A1, Ps = ASM(d=-20e-3, PhsHolo=Pc, AmpHolo=Ae.to(device))
            if Update_version == 'ResidualAmpabs': # With abs
                ResidualAc = torch.abs(Ae.to(device) - Ac)
                assert ResidualAc.min() >= 0.0 and ResidualAc.max() <= 1.0, "ResidualAc is in the range of [0, 1]"
                _, ResudualRetriPs = ASM(d=-20e-3, PhsHolo=Pc, AmpHolo=ResidualAc)
            elif Update_version == 'ResidualAmpNOabs': # Without abs
                ResidualAc = Ae.to(device) - Ac
                assert ResidualAc.min() >= -1.0 and ResidualAc.max() <= 1.0, "ResidualAc is in the range of [-1.0, 1]"
                _, ResudualRetriPs = ASM(d=-20e-3, PhsHolo=Pc, AmpHolo=ResidualAc)
            elif Update_version == 'ResidualAmpFBG': # add for FG, minus for BG
                withoutAbs_ResidualAc = Ae.to(device) - Ac
                x = torch.abs(withoutAbs_ResidualAc)
                y = torch.zeros_like(withoutAbs_ResidualAc)
                bg_withoutAbs_ResidualAc = torch.where((withoutAbs_ResidualAc<0), x, y)
                fg_withoutAbs_ResidualAc = torch.where((withoutAbs_ResidualAc>0), x, y)
                _, bg_ResudualRetriPs = ASM(d=-20e-3, PhsHolo=Pc, AmpHolo=bg_withoutAbs_ResidualAc)
                _, fg_ResudualRetriPs = ASM(d=-20e-3, PhsHolo=Pc, AmpHolo=fg_withoutAbs_ResidualAc)
                ResudualRetriPs = (fg_ResudualRetriPs - bg_ResudualRetriPs)
            elif Update_version == 'ResidualAmpBG':
                withoutAbs_ResidualAc = Ae.to(device) - Ac
                x = torch.abs(withoutAbs_ResidualAc)
                y = torch.zeros_like(withoutAbs_ResidualAc)
                bg_withoutAbs_ResidualAc = torch.where((withoutAbs_ResidualAc<0), x, y)
                _, bg_ResudualRetriPs = ASM(d=-20e-3, PhsHolo=Pc, AmpHolo=bg_withoutAbs_ResidualAc)
                ResudualRetriPs = bg_ResudualRetriPs
            elif Update_version == 'DirectRetrievePs':
                _, Ps = ASM(d=-20e-3, PhsHolo=Pc, AmpHolo=Ae.to(device))
                ResudualRetriPs = 0.0
            # Superpose the residual phase and net output phase 
            Ps = Ps + ResudualRetriPs
            # Rescale output_phs based on its periofic nature
            Ps = Ps - torch.floor(Ps/(2*torch.pi)) * (2*torch.pi)
            if not PPOH:
                AePsCollection(iter_i, ExperiencePool, data=torch.cat((Ae.to(device), Ps), 1))
                Dataset_Name = 'StackedAePs'
                AePs_dataset = MyDataset(Dataset_Name=Dataset_Name, TIME=TIME, NET=NET, NOTE=NOTE)
                train_loader = Data.DataLoader(dataset = AePs_dataset, batch_size = BatchSize, num_workers=nw, drop_last=True,)
            elif PPOH:
                A1, Psretri = ASM(d=-20e-3, PhsHolo=Pc, AmpHolo=Ae.to(device))
                x = torch.ones_like(A1)
                y = torch.zeros_like(A1)
                A1binary = torch.where(A1>0.25, x, y)
                AePsAbinaryCollection(iter_i, ExperiencePool, data=torch.cat((Ae.to(device), Ps, A1binary), 1))
                Dataset_Name = 'StackedAePsA1binary'
                AePsA1binary_dataset = MyDataset(Dataset_Name=Dataset_Name, TIME=TIME, NET=NET, NOTE=NOTE)
                train_loader = Data.DataLoader(dataset = AePsA1binary_dataset, batch_size = BatchSize, num_workers=nw, drop_last=True,)
'''
