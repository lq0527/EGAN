
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchsummary import summary

class dirac(torch.autograd.Function):
    # def __init__(self):
    #     super(mystep,self).__init__()
    #     # self.mystep =mystep
    @staticmethod 
    def forward(ctx,input_x):
        ctx.save_for_backward(input_x)
        output = torch.sign(input_x)
        output = (output+1.0)/2.0
        return output
    @staticmethod 
    def backward(ctx,grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input == 0] = torch.rand_like(grad_input[input == 0]) * 2 - 1
        return grad_input

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
           
class mystep(torch.autograd.Function):
    # def __init__(self):
    #     super(mystep,self).__init__()
    #     # self.mystep =mystep
    @staticmethod 
    def forward(ctx,input_x):
        ctx.save_for_backward(input_x)
        output = torch.sign(input_x)
        output = (output+1.0)/2.0
        return output
    @staticmethod 
    def backward(ctx,grad_output):
        input_x, = ctx.saved_tensors
        grad_output = grad_output*4*torch.sigmoid(4*input_x)*(1-torch.sigmoid(4*input_x))
        return grad_output



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
            if self.output_process == 'mysign':
                # d1 = torch.sigmoid(d1)
                # # d1 = d1/torch.max(d1)
                # d1 = torch.where(d1 >=torch.mean(d1), 1, 0)
                # # d1 = torch.where(d1 >=0.5, 1, 0)
                d1 = mystep.apply(d1)
                # d1 = d1/torch.max(d1)
                # d1 = (d1+1.0) / 2.0
            if self.output_process == 'mysign1':
                d1 = dirac.apply(d1)
            if self.output_process == 'BinaryQuantize':
                d1 = BinaryQuantize.apply(d1, self.k, self.t)

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
     
# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
#         self.pool1 = nn.MaxPool2d(kernel_size=2)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.pool2 = nn.MaxPool2d(kernel_size=2)
#         self.fc1 = nn.Linear(64 * 25 * 25, 128)

#     def forward(self, x):
#         # print('1',x.shape)
#         x = F.relu(self.conv1(x))
#         # print('2',x.shape)
#         x = self.pool1(x)
#         # print('3',x.shape)
#         x = F.relu(self.conv2(x))
#         # print('4',x.shape)
#         x = self.pool2(x)
#         # print('5',x.shape)
#         x = x.view(-1, 64 * 25 * 25)
#         # print('6',x.shape)
#         x = F.relu(self.fc1(x))
#         # print('7',x.shape)
#         return x

# # 构建孪生神经网络
# class SiameseNetwork(nn.Module):
#     def __init__(self):
#         super(SiameseNetwork, self).__init__()
#         self.cnn = CNN()
#         self.fc1 = nn.Linear(128, 64)
#         self.fc2 = nn.Linear(64, 1)

#     def forward(self, x1, x2):
#         feat1 = self.cnn(x1)
#         feat2 = self.cnn(x2)
#         dist = feat1 - feat2
#         # print('8',dist.shape)
#         x = F.relu(self.fc1(dist))
#         # print('9',x.shape)
        
#         x = self.fc2(x)
#         # print(x.shape)
#         return x

# # 训练孪生神经网络
# def train_siamese_network(model, train_loader, criterion, optimizer, num_epochs=10):
#     for epoch in range(num_epochs):
#         running_loss = 0.0
#         for i, data in enumerate(train_loader, 0):
#             inputs1, inputs2, labels = data
#             optimizer.zero_grad()
#             outputs = model(inputs1, inputs2)
#             loss = criterion(outputs.squeeze(), labels)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
#         print('Epoch %d loss: %.3f' % (epoch + 1, running_loss / len(train_loader)))    
        
# def get_similarity_score(output1, output2):
#     # 计算两张图片的特征向量之间的余弦相似度
#     dot_product = torch.sum(output1 * output2, dim=1)
#     norm1 = torch.norm(output1, p=2, dim=1)
#     norm2 = torch.norm(output2, p=2, dim=1)
#     similarity_score = dot_product / (norm1 * norm2)
#     return similarity_score        

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

