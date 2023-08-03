import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchsummary import summary
from DataLoader import device

# class BinaryLayer(nn.Module):
#     def __init__(self):
#         super(BinaryLayer, self).__init__()
#     @staticmethod 
#     def forward(self, x):
#         return gumbel_softmax(x)
#     @staticmethod 
#     def backward(self, grad_output):
#         return grad_output

def gumbel_softmax(x, temperature=0.1):
    gumbel_noise = torch.rand_like(x)
    gumbel_noise = -torch.log(-torch.log(gumbel_noise + 1e-10) + 1e-10)
    y = F.softmax((x + gumbel_noise) / temperature, dim=-1)
    return y

# class BinaryGenerator(nn.Module):
#     def __init__(self, temperature=5.0):
#         super(BinaryGenerator, self).__init__()
#         self.temperature = temperature

#     def forward(self, x):
#         binary_outputs = self.gumbel_softmax(x)
#         return binary_outputs

# def gumbel_softmax(self, logits):
#         gumbel_noise = torch.rand_like(logits)
#         gumbel_noise = -torch.log(-torch.log(gumbel_noise + 1e-10) + 1e-10)
#         binary_outputs = F.softmax((logits + gumbel_noise) / self.temperature, dim=-1)
#         return binary_outputs

def LearnableBias(x, out_chn):
    bias = nn.Parameter(torch.zeros(1, out_chn, 100, 100), requires_grad=True)
    bias = bias.to(device)
    out = x + bias.expand_as(x)
    return out
 
def BinaryActivation(x):
        out_forward = torch.sign(x)

        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
        out = out_forward.detach() - out3.detach() + out3
        out = (out+1.0)/2.0  ##
        return out

    
class BinaryQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, k, t):
        ctx.save_for_backward(input, k, t)
        out = torch.sign(input)
        out = (out+1.0)/2.0 
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, k, t = ctx.saved_tensors
        grad_input = k * t * (1 - torch.pow(torch.tanh(input * t), 2)) * grad_output
        return grad_input, None, None


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


class UNet_V3(nn.Module):
    def __init__(self,img_ch=3,output_ch=1, output_process='0to1'):
        super(UNet_V3,self).__init__()
        
        self.output_ch = output_ch
        self.output_process = output_process
        self.k = torch.tensor([10]).float().cuda()
        self.t = torch.tensor([0.1]).float().cuda()
 
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

        x2 = self.Maxpool(x1)#64,100,100->64,50,50
        x2 = self.Conv2(x2)  #64,50,50->128,50,50
        
        x3 = self.Maxpool(x2)  
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        x6 = self.Maxpool(x5)
        x6 = self.Conv6(x6)  #1024,3,3->2048,3,3

        #decoding + concat path
        d6 = self.Up6(x6)  #2048,3,3->1024,6,6
        d6 = torch.cat((x5,d6),dim=1)  #1024,6,6->2048,6,6
        d6 = self.Up_conv6(d6)  #2048,6,6->1024,6,6
        
        d5 = self.Up5(x5)        #1024,6,6-> 512,12,12                                 #dimension
        d5 = torch.cat((x4,d5),dim=1)                                              #channel叠加
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

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
            
            if self.output_process == 'learned threshold':
                # d1 = torch.sigmoid(d1)
                # d1 = gumbel_softmax(d1)
                d1 = LearnableBias(d1,1)
                d1 = BinaryActivation(d1)

            if self.output_process == 'narrow25':
                d1 = torch.sigmoid(25*d1)     
            if self.output_process == 'narrow20':
                d1 = torch.sigmoid(20*d1)  
                # d1 = d1/torch.max(d1) 
            if self.output_process == 'sign0.3':
                d1 = torch.sigmoid(d1)
                # d1 = d1/torch.max(d1)
                d1 = torch.where(d1 >= 0.3, 1, 0)
             
            if self.output_process == 'sign2':
                d1 = torch.sigmoid(20*d1)
                d1 = d1/torch.max(d1)
                d1 = torch.where(d1 >= 0.5, 1, 0)
                
            if self.output_process == 'sign0.4':
                d1 = torch.sigmoid(d1)
                # d1 = d1/torch.max(d1)
                d1 = torch.where(d1 >= 0.4, 1, 0)
            if self.output_process == 'sign0.5':
                d1 = torch.sigmoid(d1)
                d1 = d1/torch.max(d1)
                d1 = torch.where(d1 >= 0.5, 1, 0)
            if self.output_process == 'mean':
                d1 = torch.sigmoid(d1)
                # d1 = d1/torch.max(d1)
                d1 = torch.where(d1 >= torch.mean(d1), 1, 0)
            if self.output_process == 'BinaryQuantize':
                d1 = BinaryQuantize.apply(d1, self.k, self.t)


        elif self.output_ch == 2:
            if self.output_process == '0to1':
                d1[:,0] = torch.sigmoid(d1[:,0])  # phs be in the range of [0, 2pi].
            if self.output_process == 'periodic nature':
                d1[:,0] = d1[:,0] - torch.floor(d1[:,0]/(2*torch.pi)) * (2*torch.pi) # re-arrange to [0, 2pi]
            if self.output_process == 'direct output':
                d1[:,0] = d1[:,0]
            d1[:,1] = torch.sigmoid(d1[:,1])    # re-arrange to [0, 1]
        return d1

    #     d1 = 
        
    # def BCE_loss(logit, label):
    # criter = nn.BCELoss()
    # logit = logit.squeeze(dim = 1)
    # total_loss = criter(logit, label.float())
    # return total_loss



# #########################################################
# myUNet = UNet_V3(img_ch=1)
# myUNet = nn.DataParallel(myUNet)
# CUDA = torch.cuda.is_available()
# if CUDA:
#     myUNet = myUNet.cuda()
# summary(myUNet, input_size=(1, 50, 50))
# #########################################################
