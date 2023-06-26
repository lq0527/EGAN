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
from torch.nn.functional import cosine_similarity

from DataLoader import BatchSize, device, train_loader, test_loader, valid_loader, nw
from AuxiliaryFunction import *
from LossFunction import *
from EvaluationMetric import Validation, PSNR, Validation1
from GANmodel import Generator,Evaluator,init_weights
      
def get_similarity_score(output1, output2):
    # 计算两张图片的特征向量之间的余弦相似度
    dot_product = torch.sum(output1 * output2, dim=(1, 2, 3))
    norm1 = torch.norm(output1, p=2, dim=(1, 2, 3))
    norm2 = torch.norm(output2, p=2, dim=(1, 2, 3))
    similarity_score = dot_product / (norm1 * norm2)
    similarity_score = similarity_score.unsqueeze(1)
    return similarity_score


criter_mse = nn.MSELoss()
criter_BCEll = nn.BCEWithLogitsLoss()
criter_BCE = nn.BCELoss()

# Declare some variables
Index =1                                                                                                                                                                                                                 
Weights_SSVL = 'Option7_4'
output_process = '0to1' # sigmoid: The output phase is [0, 2pi], 0to1: The output amplitdue is [0, 1]
TIME = 'TEST_0606' +  '_' + str(Index)   #x
DATA, NET, Method, Update_version ='Tom&Jerry', 'EGan', 'RELU', 'FixedPu' # or mnist2k_Binarized || TWOPOOLS,  TWOPOOLSplusSSVL,  ONEPOOLplusSSVL #x
Lossfunc_TWOPOOLS, Lossfunc_SSVL, Continue, BOH = 'Null', 'mse', False, True
img_ch, output_ch = 1, 1

base_epochs = 1000  #500
best_PSNR = 0.0
# ###############OUTPUT###################
NOTE = TIME + "_" + NET + "_" + Method + '_' + Update_version + '_L' + Lossfunc_TWOPOOLS + '_' + Lossfunc_SSVL + '_Continue' + str(Continue) + '_BOH' + str(BOH) #x
# ###############store path###################
net_pth_G = './model_pth/' + NOTE + "_Generator.pth"
net_pth_E = './model_pth/' + NOTE + '_Evaluator.pth'

train_lossDegragation_TXTNotes = '/public/home/liuqing2022/hologram/Results/LossDegragation/TXTNotes/' + NOTE + '.txt'
writer = SummaryWriter('runs/' + NOTE)
train_iters = len(train_loader)  
valid_iters = len(valid_loader)
with open(train_lossDegragation_TXTNotes,'a') as file0:
    print("Today is {}, the used data is {}, the used net is {}, the used method is {}".format(TIME, DATA, NET, Method), file=file0)
    print("Base epoch is {}, training iteration is {}, validation iteration is {}".format(base_epochs, train_iters, train_iters), file=file0)
    print("The batch size is {}, the device is {}".format(BatchSize, device), file=file0)
    print("Other thing about training: {}".format(NOTE), file=file0)

# Declare variable to record info during training 
PSNR_list = []
PSNR_binary_list = []
sum_loss_d_list = []
sum_loss_g_list = []
sum_loss_g_bce_list = []

NOTE = TIME + "_" + NET + "_" + Method + '_' + Update_version + '_L' + Lossfunc_TWOPOOLS + '_' + Lossfunc_SSVL + '_Continue' + str(Continue) + '_BOH' + str(BOH) #x

final_path =  './model_pth/' + NOTE + '/' + str(int(base_epochs)) + "_final.pth"


E = Generator(img_ch=2,output_process='0to1')
G = Generator(img_ch=1,output_process='0to1')
E = E.to(device)
G = G.to(device)


lr1= 0.0001
lr2= 0.0001

beta1 = 0.5
E_optimizer = optim.Adam(E.parameters(), lr=lr1, betas=(beta1, 0.999))
g_optimizer = optim.Adam(G.parameters(), lr=lr2, betas=(beta1, 0.999))
real_label = 1.
fake_label = 0.
real_labels = torch.ones(BatchSize, 1).to(device) #*fake
fake_labels= torch.zeros(BatchSize, 1).to(device) #*real # 

E_loss =torch.tensor([0])
g_loss =0

E_epochs = 2
G_epochs = 5
with open(train_lossDegragation_TXTNotes,'a') as file0:
            print("++++++++++++++++++++++++++++++++++++", file=file0)
            print("        SSVL Training START!        ", file=file0)
            print("++++++++++++++++++++++++++++++++++++", file=file0)
            print("     train epoch        G_loss          D_loss     G_loss_BCE", file=file0)

for base_epoch_i in range(base_epochs):         
    if True :
       
        SSVL_sum_loss_d = 0.0
        SSVL_sum_loss_g = 0.0
        sum_loss_g_bce = 0.0
        
        
        # SSVL_train_bar = tqdm(train_loader)
        SSVL_iters = len(train_loader)
        
        
        for E_epoch_i in range (E_epochs):
            
            SSVL_sum_loss_e = 0.0
            SSVL_sum_loss_g = 0.0
            sum_label = 0.0
            sum_score = 0.0
            
            SSVL_train_bar = tqdm(train_loader)  
            for SSVL_iter_i, SSVL_data in enumerate(SSVL_train_bar):
                
                SSVL_input_Ae = SSVL_data  
                SSVL_Pu = torch.zeros_like(SSVL_input_Ae)
                
                E_optimizer.zero_grad()
                
                #固定网络
                for param in E.parameters():
                    param.requires_grad = True
                
                for param in G.parameters():
                    param.requires_grad = False
               
                SSVL_output_As = G(SSVL_input_Ae.to(device))
                
                SSVL_binary_As=torch.where(SSVL_output_As>0.5,1.0,0.0).detach()
                # SSVL_binary_As=SSVL_output_As
                SSVL_Ac, SSVL_Pc = ASM(d=20e-3, PhsHolo=SSVL_Pu.to(device), AmpHolo = SSVL_binary_As.to(device))

                # ENN的训练过程
                
                AeAs = torch.cat((SSVL_input_Ae.to(device), SSVL_output_As.to(device)), dim=1)
                score = E(AeAs)        
                # 计算余弦相似度损失
                # torch.mean((y_true - y_pred) ** 2)
                labels = (SSVL_input_Ae.to(device) - SSVL_Ac.to(device))
                # labels = get_similarity_score(SSVL_input_Ae.to(device),SSVL_Ac.to(device)).detach()
                E_loss = criter_mse(score, labels)
                sum_label = sum_label+labels
                SSVL_sum_loss_e = SSVL_sum_loss_e+E_loss.item()
                sum_score = sum_score+score
                
                E_loss.backward()
                E_optimizer.step()
                SSVL_train_bar.desc = "Train epoch [{}/{}],Evaluator epoch [{}/{}]".format(base_epoch_i + 1, base_epochs,E_epoch_i + 1,E_epochs)
                SSVL_train_bar.set_postfix({"G_loss":g_loss.item(),"E_loss":E_loss.item()})
                
            print('Epoch[{}/{}], Train E :g_loss:{:.6f},avg e_loss:{:.6f},avg score:{:.6f},label:{:.6f}  '.format(
                    base_epoch_i+1, base_epochs, g_loss.item(),(SSVL_sum_loss_e/SSVL_iters),(sum_score/SSVL_iters).mean(),(sum_label/SSVL_iters).mean()
                  # 打印的是真实图片的损失均值
                ))
            
            
            writer.add_scalar('E_loss', SSVL_sum_loss_e / SSVL_iters, (base_epoch_i+1)*(E_epoch_i+1))
            
        # init_weights(G, init_type='kaiming', gain=0.02) #G初始化
        # print('---------------------G is initialized--------------------------')
        for G_epoch_i in range (G_epochs):
            SSVL_train_bar = tqdm(train_loader) 
            
            SSVL_sum_loss_e = 0.0
            SSVL_sum_loss_g = 0.0
            sum_label1 = 0.0 
            sum_score1 = 0.0 
                
            for SSVL_iter_i, SSVL_data in enumerate(SSVL_train_bar):  
                SSVL_input_Ae = SSVL_data  
                SSVL_Pu = torch.zeros_like(SSVL_input_Ae)     
                #生成网络的训练过程
                for param in G.parameters():
                    param.requires_grad = True
                
                for param in E.parameters():
                        param.requires_grad = False
         
                g_optimizer.zero_grad()
                SSVL_output_As1 = G(SSVL_input_Ae.to(device))
                
                AeAs1 = torch.cat((SSVL_input_Ae.to(device), SSVL_output_As1.to(device)), dim=1)
                score1= E(AeAs1)
                
                score_sum = torch.sum(torch.abs(score1),dim=(2,3))
        
                
                SSVL_binary_As1=torch.where(SSVL_output_As1>0.5,1.0,0.0).detach()
                # SSVL_binary_As=SSVL_output_As
                # SSVL_Ac_1, SSVL_Pc = ASM(d=20e-3, PhsHolo=SSVL_Pu.to(device), AmpHolo = SSVL_output_As1.to(device))
                SSVL_Ac1, SSVL_Pc = ASM(d=20e-3, PhsHolo=SSVL_Pu.to(device), AmpHolo = SSVL_binary_As1.to(device))
                
                # label1 = get_similarity_score(SSVL_input_Ae.to(device),SSVL_Ac1.to(device)).detach()         
                label1 = torch.mean((SSVL_input_Ae.to(device) - SSVL_Ac1.to(device)) ** 2)
                g_loss = criter_mse(score_sum, fake_labels) #+10*criter_mse(SSVL_input_Ae.to(device),SSVL_Ac_1)
                g_loss.backward()
                g_optimizer.step()
                
                sum_label1 = sum_label1+label1
                SSVL_sum_loss_g = SSVL_sum_loss_g+g_loss.item()
                sum_score1 = sum_score1+score1  
     
                
                SSVL_train_bar.desc = "Train epoch [{}/{}],Generator epoch [{}/{}]".format(base_epoch_i + 1, base_epochs,G_epoch_i + 1,G_epochs)
                SSVL_train_bar.set_postfix({"G_loss":g_loss.item(),"E_loss":E_loss.item()})
            print('Epoch[{}/{}],Train G: avg g_loss:{:.6f},d_loss:{:.6f},avg score1:{:.6f},label1:{:.6f}  '.format(
                    base_epoch_i+1, base_epochs, (SSVL_sum_loss_g / SSVL_iters),E_loss.item(),(sum_score1/SSVL_iters).mean(),(sum_label1/SSVL_iters).mean()
                  # 打印的是真实图片的损失均值
                ))    
                
            net = G  ##########
            # net1 = Generator(output_process='sign')  ##########
            # net1.eval()
            with torch.no_grad():
                PSNR_val_sum = 0.0
                PSNR_val_sum_binary = 0.0
                val_bar = tqdm(valid_loader)
                val_iters = len(valid_loader)
                
                for val_iter_i, val_data in enumerate(val_bar):
                    val_input_Ae = val_data                                  # [BatchSize, 1, 50, 50]
                    val_output_As=net((val_input_Ae).to(device))
                    val_output_As_binary = torch.where(val_output_As>=0.5,1,0)
                    # val_output_As = val_output_As /((torch.max(torch.max(val_output_As, -1)[0], -1)[0]).unsqueeze(-1).unsqueeze(-1)+1e-6) 
                    # val_output_As = torch.where(val_output_As>=0.5,1,0)
                    # print('----------------------')
                    # print(val_output_As.size())
                    
                    val_Pu = torch.zeros_like(val_output_As_binary)
                    
                    val_recons_Ac, val_recons_Pc = ASM(d=20e-3, PhsHolo = (val_Pu).to(device), AmpHolo = val_output_As.to(device))
                    val_recons_Ac_binary, val_recons_Pc = ASM(d=20e-3, PhsHolo = (val_Pu).to(device), AmpHolo = val_output_As_binary.to(device))
                    # val_fgMask = torch.where(val_input_Ae > 0.5, 1.0, 0.0).to(device)
                    # val_num_fgpixel = torch.sum(val_fgMask, dim=(-1, -2))                    # [BatchSize, 1]
                    # val_recons_Ac_MAX = torch.max(torch.max(val_recons_Ac, -1)[0], -1)[0]                  # [BatchSize, 1]
                    # val_recons_Ac_MAX = val_recons_Ac_MAX.unsqueeze(-1).unsqueeze(-1)                      # [BatchSize, 1, 50, 50]
                    # val_recons_Ac_MEAN = torch.sum(val_recons_Ac*val_fgMask, dim=(2, 3))/val_num_fgpixel # [BatchSize, 1]
                    # val_recons_Ac_MEAN = val_recons_Ac_MEAN.unsqueeze(-1).unsqueeze(-1)                    # [BatchSize, 1, 50, 50]
                    # val_recons_Ac_MAX_normlzd = val_recons_Ac / val_recons_Ac_MAX
                    # val_recons_Ac_MEAN_normlzd = val_recons_Ac / val_recons_Ac_MEAN
                    # if (val_recons_Ac / val_input_Ae_ratio.to(device)).max() < 2.0:
                    #     val_recons_Ac_normlzd = val_recons_Ac / val_input_Ae_ratio.to(device)
                    #     val_recons_Ac_normlzd = torch.where(val_recons_Ac_normlzd > 1.0, 2-val_recons_Ac_normlzd, val_recons_Ac_normlzd)
                    #     nmlz_method = 'energy'
                    # else:
                    #     print("The range of reconstructed amplitude hologram is not in[0,2]")
                    # val_recons_Ac_normlzd = val_recons_Ac /((torch.max(torch.max(val_recons_Ac, -1)[0], -1)[0]).unsqueeze(-1).unsqueeze(-1)+1e-6) 
                    # nmlz_method = 'max'
                    PSNR_val_iter = PSNR(val_recons_Ac.to(device), val_input_Ae.to(device))
                    PSNR_val_iter_binary = PSNR(val_recons_Ac_binary.to(device), val_input_Ae.to(device))
                    # val_recons_Ac_normlzd = Amplitude_Normalization(val_recons_Ac, val_input_Ae.to(device), normalized = 'Energy_Proportionated') 
                    # PSNR_val_iter = PSNR(val_recons_Ac_normlzd, val_input_Ae.to(device))
                    PSNR_val_sum += PSNR_val_iter
                    PSNR_val_sum_binary += PSNR_val_iter_binary
                    # val_bar.desc = "Validation epoch[{}/{}] PSNR for no binarization:{:.3f}".format(base_epoch_i + 1, base_epochs, PSNR_val_iter ) #输出每次epoch的validation的PSNR值
                    
                    
                    # val_bar.desc = "Validation epoch[{}-{}/{}] AVG PSNR for no binarization:{:.3f}, PSNR for binarization:{:.3f}".format(base_epoch_i + 1,G_epoch_i+1, base_epochs, PSNR_val_sum/val_iter_i,PSNR_val_sum_binary/val_iter_i ) 
            print("Validation epoch[{}-{}/{}] AVG PSNR for no binarization:{:.3f}, PSNR for binarization:{:.3f}".format(
                base_epoch_i + 1,G_epoch_i+1, base_epochs, PSNR_val_sum/val_iter_i,PSNR_val_sum_binary/val_iter_i )        
            )        
                
            
            writer.add_scalar('G_loss', (SSVL_sum_loss_g / SSVL_iters), (base_epoch_i+1)*(G_epoch_i+1))
            
            
            # real_img = SSVL_input_Ae# torch.Size([BatchSize, 1, 100, 100])
            # SSVL_output_As = G((SSVL_input_Ae).to(device)) 
            # # SSVL_binary_As = SSVL_output_As  #不加detach()的话梯度可以传 说明梯度在此时会影响 二值化也是
            # SSVL_binary_As=torch.where(SSVL_output_As>0.5,1.0,0.0)
            # SSVL_Ac, SSVL_Pc = ASM(d=20e-3, PhsHolo=SSVL_Pu.to(device), AmpHolo = SSVL_binary_As.to(device)) 
            # score = E(SSVL_input_Ae.to(device),SSVL_output_As.to(device))       
            # labels = get_similarity_score(SSVL_input_Ae.to(device),SSVL_Ac.to(device)).detach()
            # E_loss = criter_mse(score, labels)
           
            # g_loss = criter_mse(score,real_labels)  
            # E.zero_grad()
            # G.zero_grad()
            # E_loss.backward(retain_graph=True)  #这里还不能分开````
            # g_loss.backward()
            # d_optimizer.step() 
            # g_optimizer.step()   
            # SSVL_sum_loss_d += E_loss.item()
            # SSVL_sum_loss_g += g_loss.item()
            # # sum_loss_g_bce += g_loss_GAN.item()

             
            
        writer.add_scalar('Train/Discriminator_loss_total', SSVL_sum_loss_d / SSVL_iters, base_epoch_i)
        writer.add_scalar('Train/Generator_loss_total', SSVL_sum_loss_g / SSVL_iters, base_epoch_i)
        
        sum_loss_d_list.append(SSVL_sum_loss_d / SSVL_iters)
        sum_loss_g_list.append(SSVL_sum_loss_g / SSVL_iters)
            
   
            
        with open(train_lossDegragation_TXTNotes,'a') as file0:
            print("       {:.3f}        {:.6f}      {:.6f}         {:.6f}     ".format(base_epoch_i+1, SSVL_sum_loss_g / SSVL_iters, SSVL_sum_loss_d / SSVL_iters,sum_loss_g_bce / SSVL_iters),file=file0)
           
                
                
        # if (base_epoch_i + 1) % 5 == 0:
            
        #     with open(train_lossDegragation_TXTNotes,'a') as file0:
        #         print('Epoch[{}/{}],g_loss:{:.6f},d_loss:{:.6f} '
        #             'D real: {:.6f},D fake: {:.6f}'.format(
        #             base_epoch_i, base_epochs, g_loss.item(),d_loss.item(),
        #             real_score.mean().item() , fake_score.mean().item()   
        #         ),file = file0)
            
  
   
    if (base_epoch_i+1) % 10 == 0:
        with open(train_lossDegragation_TXTNotes,'a') as file0:
            print("======================================", file=file0)
            print("     Validation on val_data Start     ", file=file0)
            print("======================================", file=file0)
        # G = Init_net_from_before(my_img_ch=1, my_output_ch=1, my_output_process='0to1', net_path=net_pth_G)    
        # G, train_from = Init_net_from_before(my_img_ch=1, my_output_ch=1, my_output_process='0to1', net_path=net_pth_G,net_kind=Generator), 'continue'
        # net = net.cuda()
        net = G  ##########
        # net1 = Generator(output_process='sign')  ##########
        net.eval()
        
        with torch.no_grad():
            PSNR_val_sum = 0.0
            PSNR_val_sum_binary = 0.0
            val_bar = tqdm(valid_loader)
            val_iters = len(valid_loader)
            for val_iter_i, val_data in enumerate(val_bar):
                val_input_Ae = val_data                                  # [BatchSize, 1, 100, 100]
                val_output_As=net((val_input_Ae).to(device))
                val_output_As_binary = torch.where(val_output_As>=0.5,1,0)
                val_Pu = torch.zeros_like(val_output_As_binary)
                
                val_recons_Ac, val_recons_Pc = ASM(d=20e-3, PhsHolo = (val_Pu).to(device), AmpHolo = val_output_As.to(device))
                val_recons_Ac_binary, val_recons_Pc = ASM(d=20e-3, PhsHolo = (val_Pu).to(device), AmpHolo = val_output_As_binary.to(device))
        
                PSNR_val_iter = PSNR(val_recons_Ac.to(device), val_input_Ae.to(device))
                PSNR_val_iter_binary = PSNR(val_recons_Ac_binary.to(device), val_input_Ae.to(device))
                PSNR_val_sum += PSNR_val_iter
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

            if PSNR_val_avg_binary > best_PSNR:
                best_PSNR = PSNR_val_avg_binary
                net_best = net
                torch.save(net.state_dict(), './model_pth/' + NOTE + "_best.pth")   #存放PSNR最好的路径
            if not os.path.exists('./model_pth/' + NOTE + '/'):
                os.mkdir('./model_pth/' + NOTE + '/')
            torch.save(net.state_dict(), './model_pth/' + NOTE + '/' + str(int(base_epochs)) + "_final.pth")
            # torch.save(net.state_dict(), './model_pth/' + NOTE + ".pth")
                # 保存模型
            torch.save(G.state_dict(), net_pth_G)
            torch.save(E.state_dict(), net_pth_E)
            
            
        model = Generator(img_ch=1, output_ch=1,output_process = '0to1')
        model_binary = Generator(img_ch=1, output_ch=1,output_process = 'sign0.5')
        
        for i in range(2):
            net_path = './model_pth/' + NOTE + "_Generator.pth"
            
            model.load_state_dict(torch.load(net_path))
            model_binary.load_state_dict(torch.load(net_path))
            
            input1 = tiff.imread('./img/test_img/TJ_test/'+str(i)+'.tiff')
            input = torch.tensor(input1)
            input = torch.reshape(input,(1,1,100,100))
            model.eval()
            model_binary.eval()
    
            with torch.no_grad():
                output_As = model(input)
                output_As_binary = model_binary(input)
                
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
            # plt.savefig(output_save + '/_pic' + str(i) + 'epoch_'+ str(base_epoch_i+ 1)+'_psnr'+str(psnr)+'!!!.png')

            plt.savefig(output_save + '/_pic' + str(i) + 'epoch_'+ str(base_epoch_i+ 1)+'_psnr'+str(psnr)+'_psnr_binary'+str(psnr_binary)+'.png')
            # plt.colorbar()
            plt.clf()
       
with open(train_lossDegragation_TXTNotes,'a') as file0:
    print("++++++++++++++++++++++++++++++++++++", file=file0)
    print("        SSVL Training END!        ", file=file0)
    print("++++++++++++++++++++++++++++++++++++", file=file0)
    print("\n", file=file0)

            


with open(train_lossDegragation_TXTNotes,'a') as file0:

    print("Disriminator_loss_list:{}".format(sum_loss_d_list), file=file0)
    print("Generator_loss_list:{}".format(sum_loss_g_list), file=file0)
    # print("G_BCEloss_list:{}".format(sum_loss_g_bce_list), file=file0)
    print("PSNR_list: {}".format(PSNR_list), file=file0)

excel_writer = pd.ExcelWriter('/public/home/liuqing2022/hologram/Results/LossDegragation/Plotcurve/' + NOTE + '_' + str(base_epochs) +'.xlsx')
write_list_to_elsx(My_writer=excel_writer, List=sum_loss_d_list, Columns_Name='Disriminator_loss', Sheet_Name='Disriminator_loss')
write_list_to_elsx(My_writer=excel_writer, List=sum_loss_g_list, Columns_Name='Generator_loss', Sheet_Name='Generator_loss')
# write_list_to_elsx(My_writer=excel_writer, List=sum_loss_g_bce_list, Columns_Name='G_BCEloss', Sheet_Name='G_BCEloss')
write_list_to_elsx(My_writer=excel_writer, List=PSNR_list, Columns_Name='PSNR_list', Sheet_Name='PSNR_list')   #???
excel_writer.close()

net_val = Generator(img_ch=1, output_ch=1, output_process='sign0.5')
# net = UNet_V3(img_ch=1, output_ch=1,output_process = output_process)
init_weights(net_val, init_type='kaiming', gain=0.02)
Validation(NOTE, output_ch, test_loader, net_val)

