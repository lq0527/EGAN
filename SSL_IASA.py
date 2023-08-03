#coding:utf-8
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' 
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt


from DataLoader import BatchSize, device, train_loader, test_loader, valid_loader, nw
from AuxiliaryFunction import *
from LossFunction import *
from EvaluationMetric import Validation, PSNR, Validation1
from UNet_V3_original import UNet_V3, init_weights
# from GANmodel import init_weights,FusionGenerator

def IASA(As, Ae,IASA_iters):
#########################################################IASA  
    IASA_A1 = As     
    for IASA_i in range(IASA_iters):    
        IASA_Ac, IASA_Pc = ASM(d=20e-3, PhsHolo=(torch.zeros_like(IASA_A1)).to(device), AmpHolo = IASA_A1.to(device))
        # IASA_Ac_MEAN = torch.sum(IASA_Ac*fgMask, dim=(2, 3))/num_fgpixel           # [BatchSize, 1]
        # IASA_Ac_MEAN = IASA_Ac_MEAN.unsqueeze(-1).unsqueeze(-1)                    # [BatchSize, 1, 50, 50]
        # IASA_A1, IASA_Ps = ASM(d=-20e-3, PhsHolo=IASA_Pc, AmpHolo=input_Ae.to(device)*IASA_Ac_MEAN)
        IASA_A1, IASA_Ps = ASM(d=-20e-3, PhsHolo=IASA_Pc, AmpHolo=(Ae).to(device))

        smaller_condition_mask = torch.where(IASA_Ps <= 1.5*np.pi, 1, 0)
        bigger_condition_mask =  torch.where(IASA_Ps >=0.5*np.pi, 1, 0)
        all_condition = (smaller_condition_mask) + (bigger_condition_mask)#+smaller_condition_mask1+bigger_condition_mask1
        IASA_A1 =  torch.where(all_condition == 2, 0, 1)
    A1 = IASA_A1
    #### Store {(Ac_nmlzd, output_As)} data pairs into Experience pool 1#
    # IASA_Ac, IASA_Pc = ASM(d=20e-3, PhsHolo=(torch.zeros_like(A1)).to(device), AmpHolo = A1.to(device))
    return A1

def loss_fg_bg_mse(Pred, fg_bg):
   
    fg_mask = fg_bg == 1
    bg_mask = fg_bg == 0
    fg_mean = Pred[fg_mask].mean()
    bg_mean = Pred[bg_mask].mean()

    var = Pred.var()
    loss = (1 - fg_mean)**2 + bg_mean**2 + var
    return loss

criter_cosmae = LOSS_COS_MAE()
criter_piecewisemae = PIECEWISE_MAE()
criter_mae = nn.L1Loss()
criter_mse = nn.MSELoss()
criter_BCEll = nn.BCEWithLogitsLoss()
criter_BCE = nn.BCELoss()

criter_mse_fg = LOSS_MSE_FG()
criter_mae_fg = LOSS_MAE_FG()
criter_fbg = LOSS_FG_BG_v1()
criter_ssim = LOSS_SSIM()

criter_tv = Loss_TV()
criter_var = LOSS_VAR()
criter_fl = FocalLoss()

# Declare some variables
Index = 1                                                                                                                                                                                                                  
Weights_SSVL = 'Option7_4'
output_process = '0to1' # sigmoid: The output phase is [0, 2pi], 0to1: The output amplitdue is [0, 1]
TIME = 'TEST_0802_IASA_SSL' +  '_' + str(Index)   #x
DATA, NET, Method, Update_version ='Tom&Jerry', 'UnetV3', 'onlySSL_reg', 'FixedPu' # or mnist2k_Binarized || TWOPOOLS,  TWOPOOLSplusSSVL,  ONEPOOLplusSSVL #x
Lossfunc_TWOPOOLS, Lossfunc_SSVL, Continue, BOH = 'DirectAmpMse', 'mse', False, True
img_ch, output_ch = 1, 1
Best_PSNR = 0.0
base_epochs = 500 #500
SSVL_lr, POOL1_lr, POOL2_lr = 1e-3, 1e-3, 1e-3
# SSVL_lr_degrate, POOL1_lr_degrate, POOL2_lr_degrate = False, False, False
SSVL_lr_degrate, POOL1_lr_degrate, POOL2_lr_degrate = True, True, True

# #train from scratch
net = UNet_V3(img_ch=img_ch, output_ch=output_ch, output_process=output_process)
# net = FusionGenerator(input_nc=1,output_nc=1,ngf=16)

init_weights(net, init_type='kaiming', gain=0.02)    # initial network parameters
#train from before(no need for initialization)
# net_pth1 = './model_pth/' + NOTE + '/800_799.pth'
# net, train_from = Init_net_from_before(my_img_ch=1, my_output_ch=1, my_output_process='0to1',net_path=net_pth1), 'from_before'

net = net.to(device)


SSVL_signal, POOL1_signal, POOL2_signal = True, False, False
NOTE = TIME + "_" + NET + "_" + Method + '_' + Update_version + '_L' + Lossfunc_TWOPOOLS + '_' + Lossfunc_SSVL + '_Continue' + str(Continue) + '_BOH' + str(BOH) #x
# NOTE1 = TIME + "_netpath"   #x


# net = Init_net_from_before(my_img_ch=1, my_output_ch=1, my_output_process='sigmoid', net_path=net_pth1)

train_lossDegragation_TXTNotes = '/public/home/liuqing2022/hologram/Results/LossDegragation/TXTNotes/' + NOTE + '.txt'
writer = SummaryWriter('runs/' + NOTE)
train_iters = len(train_loader)  
valid_iters = len(valid_loader)
with open(train_lossDegragation_TXTNotes,'a') as file0:
    print("Today is {}, the used data is {}, the used net is {}, the used method is {}".format(TIME, DATA, NET, Method), file=file0)
    print("Base epoch is {}, training iteration is {}, validation iteration is {}".format(base_epochs, train_iters, valid_iters), file=file0) 
    print("The batch size is {}, the device is {}".format(BatchSize, device), file=file0)
    print("Other thing about training: {}".format(NOTE), file=file0)



SSVL_Freeze, SSVL_frozen_layers = False, []

SSVL_best_recon_psnr = 0.0

SSVL_total_loss_list = []
PSNR_list = []
PSNR_binary_list = []

net_path = './model_pth/' + NOTE + ".pth"
final_path =  './model_pth/' + NOTE + '/' + str(int(base_epochs)) + "_final.pth"
# net = UNet_V3(img_ch=img_ch, output_ch=output_ch, output_process=output_process)
lr = 0.001
beta = 0.5
optimizer = optim.Adam(net.parameters(), lr=lr, betas=(beta, 0.999))

with open(train_lossDegragation_TXTNotes,'a') as file0:
            print("++++++++++++++++++++++++++++++++++++", file=file0)
            print("        SSVL Training START!        ", file=file0)
            print("++++++++++++++++++++++++++++++++++++", file=file0)
            print("     train epoch             G_loss                G_loss     ", file=file0)

for base_epoch_i in range(base_epochs):         
    if SSVL_signal :
        SSVL_signal, POOL1_signal, POOL2_signal = True, False, False   # TWOPOOLSplusSSVL
        net.train()

        SSVL_sum_loss = 0.0
        
        SSVL_train_bar = tqdm(train_loader)
        SSVL_iters = len(train_loader)
        
        for SSVL_iter_i, SSVL_data in enumerate(SSVL_train_bar):
        ##################Train Net
            SSVL_input_Ae = SSVL_data    
            SSVL_Pu = torch.zeros_like(SSVL_input_Ae)
           
           ##################Train Discriminator
            
            if output_ch == 1:
                # SSVL_output_Ps = net((SSVL_input_Ae*SSVL_input_Ae_ratio).to(device))                         # torch.Size([BatchSize, 1, 50, 50])
                SSVL_output_As = net((SSVL_input_Ae).to(device))
                
                # SSVL_output_As_binary = G_binary((SSVL_input_Ae).to(device))
                
                
            elif output_ch == 2:
                SSVL_output = net(SSVL_input_Ae.to(device))                        # torch.Size([BatchSize, 1, 50, 50])
                SSVL_output_As = SSVL_output[:,0].unsqueeze(1)                         # torch.Size([BatchSize, 1, 50, 50])
                assert SSVL_output_As.min() >= 0.0 and SSVL_output_As.max() <= 1.0, "SSVL_output_Ps is out of range [0, 1]"
          
            SSVL_Ac, SSVL_Pc = ASM(d=20e-3, PhsHolo=SSVL_Pu.to(device), AmpHolo=SSVL_output_As.to(device))
            ######################################## Modification #############################################################
            if base_epoch_i <50:
                Binary_As = IASA(SSVL_output_As, SSVL_input_Ae,30)
            else:
                Amp_retrieve, Phs_retrieve = ASM(d=-20e-3, PhsHolo=SSVL_Pc.to(device), AmpHolo=SSVL_input_Ae.to(device))
                smaller_condition_mask = torch.where(Phs_retrieve <= 1.5*np.pi, 1, 0)
                bigger_condition_mask =  torch.where(Phs_retrieve >=0.5*np.pi, 1, 0)
                all_condition = (smaller_condition_mask) + (bigger_condition_mask)#+smaller_condition_mask1+bigger_condition_mask1
                Binary_As =  torch.where(all_condition == 2, 0, 1)
            
            loss1 = criter_mse(SSVL_input_Ae.to(device),SSVL_Ac.to(device))
            loss2 = loss_fg_bg_mse(SSVL_output_As.to(device),Binary_As.to(device))
            
            # SSVL_output_As_binary = SSVL_output_As
            # SSVL_output_As_binary = torch.where(SSVL_output_As>torch.mean(SSVL_output_As),1,0).float().detach()
           
            loss = loss1 + 0.1 * loss2
            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()   
            
    
            SSVL_sum_loss += loss.item()
            
            SSVL_train_bar.desc = "Train epoch [{}/{}]".format(base_epoch_i + 1, base_epochs)
            SSVL_train_bar.set_postfix({"loss":loss.item()})
             
            
            writer.add_scalar('Train/loss_total', SSVL_sum_loss / SSVL_iters, base_epoch_i)
          
        
            
        with open(train_lossDegragation_TXTNotes,'a') as file0:
            print("       {:.3f}        {:.6f}        ".format(base_epoch_i+1, SSVL_sum_loss / SSVL_iters),file=file0)
           
            SSVL_total_loss_list.append(SSVL_sum_loss / train_iters)

    if (base_epoch_i+1) % 10 == 0:
        with open(train_lossDegragation_TXTNotes,'a') as file0:
            print("======================================", file=file0)
            print("     Validation on val_data Start     ", file=file0)
            print("======================================", file=file0)
    
        net.eval()
        with torch.no_grad():
            PSNR_val_sum = 0.0
            PSNR_val_sum_binary = 0.0
            val_bar = tqdm(valid_loader)
            val_iters = len(valid_loader)
            for val_iter_i, val_data in enumerate(val_bar):
                val_input_Ae = val_data                                  # [BatchSize, 1, 50, 50]
                #val_input_Ae_ratio = (torch.sqrt(torch.ones((BatchSize, 1))*10000 / torch.sum(val_input_Ae**2, dim=(-1,-2)))).unsqueeze(-1).unsqueeze(-1)
                # val_output_As_binary = net1((val_input_Ae).to(device))
                val_output_As=net((val_input_Ae).to(device))
             
                
                val_output_As_binary = torch.where(val_output_As>torch.mean(val_output_As),1,0)
                
                val_Pu = torch.zeros_like(val_output_As_binary)
                
                val_recons_Ac, val_recons_Pc = ASM(d=20e-3, PhsHolo = (val_Pu).to(device), AmpHolo = val_output_As.to(device))
                val_recons_Ac_binary, val_recons_Pc = ASM(d=20e-3, PhsHolo = (val_Pu).to(device), AmpHolo = val_output_As_binary.to(device))
                PSNR_val_iter = PSNR(val_recons_Ac.to(device), val_input_Ae.to(device))
                PSNR_val_iter_binary = PSNR(val_recons_Ac_binary.to(device), val_input_Ae.to(device))
                # val_recons_Ac_normlzd = Amplitude_Normalization(val_recons_Ac, val_input_Ae.to(device), normalized = 'Energy_Proportionated') 
                # PSNR_val_iter = PSNR(val_recons_Ac_normlzd, val_input_Ae.to(device))
                PSNR_val_sum += PSNR_val_iter
                PSNR_val_sum_binary += PSNR_val_iter_binary
                val_bar.desc = "Validation epoch[{}/{}] PSNR for no binarization:{:.3f}, PSNR for binarization:{:.3f}".format(base_epoch_i + 1, base_epochs, PSNR_val_iter,PSNR_val_iter_binary ) #输出每次epoch的validation的PSNR值
            PSNR_val_avg = PSNR_val_sum / val_iters
            PSNR_val_avg_binary = PSNR_val_sum_binary / val_iters
            
            PSNR_list.append(PSNR_val_avg)
            PSNR_binary_list.append(PSNR_val_avg_binary)
            writer.add_scalar('Val/Ac PSNR', PSNR_val_avg, base_epoch_i)
            writer.add_scalar('Val/Ac from binarized As PSNR', PSNR_val_avg_binary, base_epoch_i)
            
            with open(train_lossDegragation_TXTNotes,'a') as file0:
                print("Epoch[{}/{}]  Val Ac PSNR:{:.3f} Val Ac(binary) PSNR:{:.3f}".format(base_epoch_i + 1, base_epochs, PSNR_val_avg,PSNR_val_avg_binary), file=file0)
               
                print("======================================", file=file0)

            if PSNR_val_avg_binary > Best_PSNR:
                Best_PSNR = PSNR_val_avg_binary
                net_best = net
                torch.save(net.state_dict(), './model_pth/' + NOTE + "_best.pth")   #存放PSNR最好的路径
            if not os.path.exists('./model_pth/' + NOTE + '/'):
                os.mkdir('./model_pth/' + NOTE + '/')
            # if (base_epoch_i+1) % 20 == 0:
            #     torch.save(net.state_dict(), './model_pth/' + NOTE + '/' + str(int(base_epoch_i+1)) + "_" + str(base_epoch_i) + ".pth") #每20个epoch存一个路径
            torch.save(net.state_dict(), './model_pth/' + NOTE + '/' + str(int(base_epochs)) + "_final.pth")
            torch.save(net.state_dict(), './model_pth/' + NOTE + ".pth")

            
            
            
        # model = UNet_V3(img_ch=1, output_ch=1,output_process = '0to1')
        # model_binary = UNet_V3(img_ch=1, output_ch=1,output_process = 'sign')
        ################# visualization ##################################
        for i in range(4):
            net_path = './model_pth/' + NOTE + ".pth"
    
            # model.load_state_dict(torch.load(net_path))
            # model_binary.load_state_dict(torch.load(net_path))
            
            input1 = tiff.imread('./img/test_img/TJ_test/'+str(i)+'.tiff')
            input = torch.tensor(input1)
            input = torch.reshape(input,(1,1,100,100))
            # model.eval()
            # model_binary.eval()
            net.eval()
            model = net
    
            with torch.no_grad():
                output_As = model(input.to(device))
          
                output_As_binary = torch.where(output_As>torch.mean(output_As),1,0)
                # output_As_binary = model_binary(input)

                
                
            Pu = torch.zeros_like(output_As)  #Phase = uniform pi
            Ac, Pc = ASM(d=20e-3, PhsHolo = Pu.to(device), AmpHolo = output_As.to(device),BatchSize=1)
            Ac_binary, Pc_binary = ASM(d=20e-3, PhsHolo = Pu.to(device), AmpHolo = output_As_binary.to(device),BatchSize=1)
            
            psnr = PSNR(Ac.to(device), input.to(device),BatchSize=1)
            psnr = psnr.cpu().detach().numpy() 
            psnr = np.round(psnr,3)
            print('The psnr for AC from unbinary AS is',psnr)
            
            psnr_binary = PSNR(Ac_binary.to(device), input.to(device),BatchSize=1)
            psnr_binary = psnr_binary.cpu().detach().numpy() 
            psnr_binary = np.round(psnr_binary,3)
            print('The psnr for AC from binarized AS is',psnr_binary)
            


            Ac1=Ac[0].squeeze(0).cpu().detach().numpy() 
            Ac_binary1=Ac_binary[0].squeeze(0).cpu().detach().numpy() 
            
            output_As1=output_As[0].squeeze(0).cpu().detach().numpy() 
            output_As_binary1=output_As_binary[0].squeeze(0).cpu().detach().numpy() 
            
            input1=input[0].squeeze(0).cpu().detach().numpy() 
            diff = Ac1 - input1
            output_save = './img/test_img/'+TIME
            if not os.path.exists(output_save):
                os.makedirs(output_save)  #如果路径不存在，创建路径
            # output_iter_save = output_save + '/' + str(base_epoch_i)
            # if not os.path.exists(output_iter_save):
            #     os.makedirs(output_iter_save)  #如果路径不存在，创建路径
                
            plt.subplot(2,3,1), plt.title('input_AE')
            plt.imshow(input1,vmin=0, vmax=1), plt.axis('off')
            plt.subplot(2,3,2), plt.title('output_As1')
            plt.imshow(output_As1,vmin=0, vmax=1), plt.axis('off')
            plt.subplot(2,3,3), plt.title('Ac_'+str(psnr))
            plt.imshow(Ac1,vmin=0, vmax=1), plt.axis('off')
            
            plt.subplot(2,3,4), plt.title('output_As_binary')
            plt.imshow(output_As_binary1,vmin=0, vmax=1), plt.axis('off')
            plt.subplot(2,3,5), plt.title('Ac_binary_'+str(psnr_binary))
            plt.imshow(Ac_binary1,vmin=0, vmax=1), plt.axis('off')

            plt.savefig(output_save + '/_pic' + str(i) + 'epoch_'+ str(base_epoch_i+ 1)+'_psnr'+str(psnr)+'_psnr_binary'+str(psnr_binary)+'.png')
            # plt.colorbar()
            plt.clf()
       
with open(train_lossDegragation_TXTNotes,'a') as file0:
    print("++++++++++++++++++++++++++++++++++++", file=file0)
    print("        SSVL Training END!        ", file=file0)
    print("++++++++++++++++++++++++++++++++++++", file=file0)
    print("\n", file=file0)

            


with open(train_lossDegragation_TXTNotes,'a') as file0:

  
    print("SSVL_total_loss_list:{}".format(SSVL_total_loss_list), file=file0)

    print("PSNR_list: {}".format(PSNR_list), file=file0)
    print("PSNR_binary_list: {}".format(PSNR_binary_list), file=file0)
    

excel_writer = pd.ExcelWriter('/public/home/liuqing2022/hologram/Results/LossDegragation/Plotcurve/' + NOTE + '_' + str(base_epochs) +'.xlsx')
write_list_to_elsx(My_writer=excel_writer, List=SSVL_total_loss_list, Columns_Name='total_loss', Sheet_Name='total_loss')
write_list_to_elsx(My_writer=excel_writer, List=PSNR_list, Columns_Name='PSNR_list', Sheet_Name='PSNR_list')   #???
write_list_to_elsx(My_writer=excel_writer, List=PSNR_binary_list, Columns_Name='PSNR_binary_list', Sheet_Name='PSNR_binary_list')   #???

excel_writer.save()
excel_writer.close()
