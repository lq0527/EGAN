
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchsummary import summary


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)
   
class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),   #(n+2p-f)/s+1 所以该参数未改变图像width and height
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            # nn.Tanh(),
            # nn.PReLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
            # nn.Tanh()
            # nn.PReLU()
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out, size_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            # nn.Upsample(scale_factor=2),
            nn.Upsample(size=size_out),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
            # nn.Tanh()
            # nn.PReLU()
        )

    def forward(self,x):
        x = self.up(x)
        return x


class Generator(nn.Module):
    def __init__(self,img_ch=1,output_ch=1, output_process='0to1'):
        super(Generator,self).__init__()
        self.k = torch.tensor([10]).float().cuda()
        self.t = torch.tensor([0.1]).float().cuda()
        
        self.output_ch = output_ch
        self.output_process = output_process

        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)  #w,h减半，channel不变

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024)
        self.Conv6 = conv_block(ch_in=1024,ch_out=2048)

        self.Up6 = up_conv(ch_in=2048,ch_out=1024, size_out=6)
        self.Up_conv6 = conv_block(ch_in=2048, ch_out=1024)
        
        self.Up5 = up_conv(ch_in=1024,ch_out=512, size_out=12)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256, size_out=25)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128, size_out=50)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64, size_out=100)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)
        
        # self.fc = nn.Linear(2500, 2500)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)  #1,100,100->64,100,100   
        # print('x1,',x1.size())

        x2 = self.Maxpool(x1)#64,100,100->64,50,50
        # print('x2,',x2.size())
        
        x2 = self.Conv2(x2)  #64,50,50->128,50,50
        # print('x2,',x2.size())
        
        x3 = self.Maxpool(x2)  
        # print('x3,',x3.size())
        
        x3 = self.Conv3(x3)
        # print('x3,',x3.size())

        x4 = self.Maxpool(x3)
        # print('x4,',x4.size())
        
        x4 = self.Conv4(x4)
        # print('x4,',x4.size())

        x5 = self.Maxpool(x4)
        # print('x5,',x5.size())
        
        x5 = self.Conv5(x5)
        # print('x5,',x5.size())

        x6 = self.Maxpool(x5)
        # print('x6,',x6.size())
        
        x6 = self.Conv6(x6)  #1024,3,3->2048,3,3
        # print('x6,',x6.size())

        #decoding + concat path
        d6 = self.Up6(x6)  #2048,3,3->1024,6,6
        # print('d6,',d6.size())
        
        d6 = torch.cat((x5,d6),dim=1)  #1024,6,6->2048,6,6
        # print('d6,',d6.size())
        
        d6 = self.Up_conv6(d6)  #2048,6,6->1024,6,6
        # print('d6,',d6.size())
       
        d5 = self.Up5(x5)        #1024,6,6-> 512,12,12                                 #dimension
        # print('d5,',d5.size())
        
        d5 = torch.cat((x4,d5),dim=1)                                              #channel叠加
        # print('d5,',d5.size())
        d5 = self.Up_conv5(d5)
        # print('d5,',d5.size())
        
        d4 = self.Up4(d5)
        # print('d4,',d4.size())
        
        d4 = torch.cat((x3,d4),dim=1)
        # print('d4,',d4.size())
        
        d4 = self.Up_conv4(d4)
        # print('d4,',d4.size())

        d3 = self.Up3(d4)
        # print('d3,',d3.size())
        
        d3 = torch.cat((x2,d3),dim=1)
        # print('d3,',d3.size())
        
        d3 = self.Up_conv3(d3)
        # print('d3,',d3.size())

        d2 = self.Up2(d3)
        # print('d2,',d2.size())
        
        d2 = torch.cat((x1,d2),dim=1)
        # print('d2,',d2.size())
        
        d2 = self.Up_conv2(d2)
        # print('d2,',d2.size())

        d1 = self.Conv_1x1(d2)
        # print('d1,',d1.size())
        

        # d1 = self.fc(torch.flatten(d1, 1))
        # d1 = d1.reshape((-1,1, 50, 50))

        if self.output_ch == 1:
            if self.output_process == '0to1':
                d1 = torch.sigmoid(d1)
                # d1 = d1/torch.max(d1)
            if self.output_process == 'sigmoid':
                d1 = torch.sigmoid(d1) * 2 * math.pi # phs be in the range of [0, 2pi].
            if self.output_process == 'tanh':
                d1 = torch.tanh(d1) * math.pi
            if self.output_process == 'periodic nature':
                d1 = d1 - torch.floor(d1/(2*torch.pi)) * (2*torch.pi) # re-arrange to [0, 2pi]
            if self.output_process == 'direct output':
                d1 = d1
            
            if self.output_process == 'narrow10':
                d1 = torch.sigmoid(10*d1) 
            if self.output_process == 'narrow15':
                d1 = torch.sigmoid(15*d1)     
            if self.output_process == 'narrow20':
                d1 = torch.sigmoid(20*d1)  
                # d1 = d1/torch.max(d1) 
            if self.output_process == 'sign1_mean':
                d1 = torch.sigmoid(d1)
                d1 = d1/torch.max(d1)
                d1 = torch.where(d1 >=torch.mean(d1), 1, 0)
            if self.output_process == 'sign2_mean':
                d1 = torch.sigmoid(20*d1)
                d1 = d1/torch.max(d1)
                d1 = torch.where(d1 >=torch.mean(d1), 1, 0)
                # print(d1.size)
            # if self.output_process == '0to1_norm':
            #     d1 = torch.sigmoid(d1)
            #     d1 = d1/torch.max(d1)
            if self.output_process == 'sign0.5':
                d1 = torch.sigmoid(d1)
                # d1 = d1/torch.max(d1)
                d1 = torch.where(d1 >=0.5, 1, 0)
                # d1 = torch.where(d1 >=0.5, 1, 0)
                # d1 = torch.sign(d1)
                # d1 = d1/torch.max(d1)
                # d1 = (d1+1.0) / 2.0
            # if self.output_process == 'test_sign':
            #     d1 = torch.sigmoid(10*d1)  
            #     d1 = torch.where(d1 >= 0.5, 1, 0)

        elif self.output_ch == 2:
            if self.output_process == 'sigmoid':
                d1[:,0] = torch.sigmoid(d1[:,0]) * 2 * math.pi # phs be in the range of [0, 2pi].
            if self.output_process == 'periodic nature':
                d1[:,0] = d1[:,0] - torch.floor(d1[:,0]/(2*torch.pi)) * (2*torch.pi) # re-arrange to [0, 2pi]
            if self.output_process == 'direct output':
                d1[:,0] = d1[:,0]
            d1[:,1] = torch.sigmoid(d1[:,1])    # re-arrange to [0, 1]
        return d1 


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        
        self.conv_init = nn.Sequential( 
            nn.Conv2d(1, 16, 5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            
            # nn.Conv2d(32, 32, 3, stride=1,padding=1),
            # nn.BatchNorm2d(32),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(0.5),
            
            nn.MaxPool2d(2)
        )
        
        self.conv_1 = nn.Sequential(   
            nn.Conv2d(16, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            
            # nn.Conv2d(64, 64, 3, stride=1, padding=1),
            # nn.BatchNorm2d(64),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(0.5),
            
            nn.MaxPool2d(2)
        )
        
        self.conv_2 = nn.Sequential(   
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            
            # nn.Conv2d(128, 128, 3, stride=1, padding=1),
            # nn.BatchNorm2d(128),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(0.5),
            
            nn.MaxPool2d(2)
        )
        
        self.conv_nonlinear = nn.Sequential(   
            # nn.Conv2d(128, 128, 3, stride=1, padding=1),
            # nn.BatchNorm2d(128),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(0.5),
            
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),


            
        )

        self.fc = nn.Sequential(            
            nn.Linear(16, 1)
            
        )
             
    def forward(self,x):
        x = self.conv_init(x)
        # x = self.conv_1(x)
        # x = self.conv_2(x)
        x = self.conv_nonlinear(x)
        x = F.adaptive_avg_pool2d(x, (1,1))
        x =  x.flatten(1)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x
   
class Evaluator(nn.Module):
    def __init__(self):
        super(Evaluator,self).__init__()
        
        self.conv_init = nn.Sequential( 
            nn.Conv2d(2, 32, 5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.Conv2d(32, 32, 3, stride=1,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.MaxPool2d(2)
        )
        
        self.conv_1 = nn.Sequential(   
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.MaxPool2d(2)
        )
        
        self.conv_2 = nn.Sequential(   
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.MaxPool2d(2)
        )
        
        self.conv_nonlinear = nn.Sequential(   
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.Conv2d(128, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.Tanh(),
        )

        self.fc = nn.Sequential(            
            nn.Linear(16, 1)
            
        )
             
    def forward(self,img1,img2):
        x = torch.cat((img1, img2), dim=1)
        x = self.conv_init(x)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_nonlinear(x)
        x = F.adaptive_avg_pool2d(x, (1,1))
        x =  x.flatten(1)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x


########################### Fusion Net #############################################
def fconv_block(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        nn.Conv2d(in_dim,out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
        act_fn,
    )
    return model


def conv_trans_block(in_dim,out_dim,act_fn,size_out):
    model = nn.Sequential(
        nn.ConvTranspose2d(in_dim,out_dim, kernel_size=3, stride=2, padding=1,output_padding=1),
        nn.Upsample(size=size_out),  ###
        nn.BatchNorm2d(out_dim),
        
        act_fn,
    )
    return model


def maxpool():
    pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    return pool


def conv_block_3(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        fconv_block(in_dim,out_dim,act_fn),
        fconv_block(out_dim,out_dim,act_fn),
        nn.Conv2d(out_dim,out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
    )
    return model

class Conv_residual_conv(nn.Module):

    def __init__(self,in_dim,out_dim,act_fn):
        super(Conv_residual_conv,self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        act_fn = act_fn

        self.conv_1 = fconv_block(self.in_dim,self.out_dim,act_fn)
        self.conv_2 = conv_block_3(self.out_dim,self.out_dim,act_fn)
        self.conv_3 = fconv_block(self.out_dim,self.out_dim,act_fn)

    def forward(self,input):
        conv_1 = self.conv_1(input)
        conv_2 = self.conv_2(conv_1)
        res = conv_1 + conv_2
        conv_3 = self.conv_3(res)
        return conv_3


class FusionGenerator(nn.Module):

    def __init__(self,input_nc, output_nc, ngf):
        super(FusionGenerator,self).__init__()
        self.in_dim = input_nc
        self.out_dim = ngf
        self.final_out_dim = output_nc
        act_fn = nn.LeakyReLU(0.2, inplace=True)
        act_fn_2 = nn.ReLU()

        print("\n------Initiating FusionNet------\n")

        # encoder

        self.down_1 = Conv_residual_conv(self.in_dim, self.out_dim, act_fn)
        self.pool_1 = maxpool()#->50
        self.down_2 = Conv_residual_conv(self.out_dim, self.out_dim * 2, act_fn)
        self.pool_2 = maxpool() #->25
        self.down_3 = Conv_residual_conv(self.out_dim * 2, self.out_dim * 4, act_fn)
        self.pool_3 = maxpool()#->12
        self.down_4 = Conv_residual_conv(self.out_dim * 4, self.out_dim * 8, act_fn)
        self.pool_4 = maxpool()#->6

        # bridge

        self.bridge = Conv_residual_conv(self.out_dim * 8, self.out_dim * 16, act_fn)

        # decoder

        self.deconv_1 = conv_trans_block(self.out_dim * 16, self.out_dim * 8, act_fn_2,size_out=12)
        self.up_1 = Conv_residual_conv(self.out_dim * 8, self.out_dim * 8, act_fn_2)
        self.deconv_2 = conv_trans_block(self.out_dim * 8, self.out_dim * 4, act_fn_2,size_out=25)
        self.up_2 = Conv_residual_conv(self.out_dim * 4, self.out_dim * 4, act_fn_2)
        self.deconv_3 = conv_trans_block(self.out_dim * 4, self.out_dim * 2, act_fn_2,size_out=50)
        self.up_3 = Conv_residual_conv(self.out_dim * 2, self.out_dim * 2, act_fn_2)
        self.deconv_4 = conv_trans_block(self.out_dim * 2, self.out_dim, act_fn_2,size_out=100)
        self.up_4 = Conv_residual_conv(self.out_dim, self.out_dim, act_fn_2)

        # output

        self.out = nn.Conv2d(self.out_dim,self.final_out_dim, kernel_size=3, stride=1, padding=1)
        self.out_2 = nn.Sigmoid()


        # initialization

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)
            
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)


    def forward(self,input):

        down_1 = self.down_1(input)
        pool_1 = self.pool_1(down_1)
        down_2 = self.down_2(pool_1)
        pool_2 = self.pool_2(down_2)
        down_3 = self.down_3(pool_2)
        pool_3 = self.pool_3(down_3)
        down_4 = self.down_4(pool_3)
        pool_4 = self.pool_4(down_4)

        bridge = self.bridge(pool_4)

        deconv_1 = self.deconv_1(bridge)
        skip_1 = (deconv_1 + down_4)/2
        up_1 = self.up_1(skip_1)
        deconv_2 = self.deconv_2(up_1)
        skip_2 = (deconv_2 + down_3)/2
        up_2 = self.up_2(skip_2)
        deconv_3 = self.deconv_3(up_2)
        skip_3 = (deconv_3 + down_2)/2
        up_3 = self.up_3(skip_3)
        deconv_4 = self.deconv_4(up_3)
        skip_4 = (deconv_4 + down_1)/2
        up_4 = self.up_4(skip_4)

        out = self.out(up_4)
        out = self.out_2(out)
    
        return out
        
# #########################################################
# # Use summary to print the network architecture
# myG, myD = Generator(), SiameseNetwork()
# myG, myD = nn.DataParallel(myG), nn.DataParallel(myD)
# CUDA = torch.cuda.is_available()
# if CUDA:
#     myG = myG.cuda()
#     myD = myD.cuda()
# # print("Generator: ")
# # summary(myG, input_size=(1, 100, 100))
# print("\n\nDiscriminator: ")
# summary(myD, [(1, 100, 100),(1, 100, 100)])
#########################################################


#########################################################
# # Use make_dot to visualize the network architecture
# myG = SiameseNetwork() # 创建一个Net实例
# inputs = torch.randn(16, 1, 100, 100) # 定义输入数据
# dot = make_dot(myG(inputs), params=dict(myG.named_parameters())) # 可视化网络结构
# dot.render('Gnerator_Architecture', format='png')
# # # Use SummaryWriter to visualize
# # myG = Generator()
# # inputs= torch.rand(16, 2, 128, 128)
# # writer = SummaryWriter()
# # writer.add_graph(myG, inputs)
# # writer.close()
# #########################################################

