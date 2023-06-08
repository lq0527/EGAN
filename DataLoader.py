import os
import numpy as np
import torch
import torch.utils.data as Data
from torch.utils.data import Dataset
import tifffile as tiff

class MyDataset(Dataset):

    def __init__(self, Dataset_Name, TIME='', NET='', NOTE='', Dataset_Path='/public/home/liuqing2022/hologram/img/'):
        '''
        Args:
        Dataset_Name (dtring): "ExpectedAmpHolo" or "ExpectedAmpHolo_2w" or "ExpectedAmpHolo_Binarized" or "ExperiencePool_1" or "ExperiencePool_2"

            1. "ExpectedAmpHolo" or "ExpectedAmpHolo_2w" or "ExpectedAmpHolo_Binarized"
                - {(Ae)}
                - Ae --NN--> Ps' --with Au, ASM--> (Ac', Pc)
            2. "ExperiencePool_1" 
                - {(Ac, P_s)}
                - Ac --NN--> Ps' --with Au, ASM--> (Ac', Pc)
                - loss(P_s, Ps')
            3. "ExperiencePool_2" 
                - {(Ae, P^s)}
                - Ae --NN--> Ps' --with Au, ASM--> (Ac', Pc) --reset with Ae--> (Ae, Pc)--inverse ASM--> (A^1, P^s)
                - loss(P^s, Ps')

            Ae is expected target amplitude hologram
            P_s is network output source phase hologram of phase-only transducer array
            Au is fixed uniform distributed source amplitude hologram
            Ac is reconstructed amplitude hologram by ASM from P_s and Au
            Pc is reconstructed phase hologram by ASM from P_s and Au
            A^1 is retrieved amplitude hologram by inverse ASM from Ae and Pc
            P^s is retrieved phase hologram by inverse ASM from Ae and Pc

        Dataset_Path: "OriginalDataset" or "ExperiencePool"
            1. "OriginalDataset": the path of "ExpectedAmpHolo" dataset {(Ae)}
            2. "ExperiencePool": the path of "ExperiencePool" dataset {(Ac, P_s)} or {(Ae, P^s)}
        
        '''
        self.Dataset_Name = Dataset_Name
        self.Dataset_Path = Dataset_Path
        self.TIME = TIME
        self.NET = NET
        self.NOTE = NOTE

        if self.Dataset_Name == "ExpectedAmpHolo" or self.Dataset_Name == "ExpectedAmpHolo_2w" or self.Dataset_Name == "ExpectedAmpHolo_Binarized" or self.Dataset_Name == "phs":
            all_content = os.listdir(os.path.join(self.Dataset_Path, self.Dataset_Name))
            all_content.sort(key= lambda x:int(x[:-5])) #sort（）函数用于对原列表进行排序,int(x[:-5])用于将文件后缀名.tiff屏蔽
        
        if self.Dataset_Name == "StackedAeA1":
            all_content = os.listdir(os.path.join(self.Dataset_Path, 'ExperiencePool_'+ self.TIME + "_" + self.NET + "_" + self.NOTE + '/StackedAeA1'))
            all_content.sort(key= lambda x:int(x[:-5]))

        if self.Dataset_Name == "StackedAePs":
            all_content = os.listdir(os.path.join(self.Dataset_Path, 'ExperiencePool_'+ self.TIME + "_" + self.NET + "_" + self.NOTE + '/StackedAePs'))
            all_content.sort(key= lambda x:int(x[:-5]))
        
        if self.Dataset_Name == "StackedAePsA1binary":
            all_content = os.listdir(os.path.join(self.Dataset_Path, 'ExperiencePool_'+ self.TIME + "_" + self.NET + "_" + self.NOTE + '/StackedAePsA1binary'))
            all_content.sort(key= lambda x:int(x[:-5]))


        elif self.Dataset_Name == "ExperiencePool1":
            all_content = os.listdir(os.path.join(self.Dataset_Path, 'ExperiencePool_'+ self.TIME + "_" + self.NET + "_" + self.NOTE + '/ExperiencePool1'))
            all_content.sort(key= lambda x:int(x[:-4]))

        elif self.Dataset_Name == "ExperiencePool2":
            all_content = os.listdir(os.path.join(self.Dataset_Path, 'ExperiencePool_'+ self.TIME + "_" + self.NET + "_" + self.NOTE + '/ExperiencePool2'))
            all_content.sort(key= lambda x:int(x[:-4]))
        
        self.Data_list = all_content

    def __getitem__(self, index):  #__getitem__就是获取样本对，模型直接通过这一函数获得一对样本对{x:y}  
        
        if self.Dataset_Name == "ExpectedAmpHolo" or self.Dataset_Name == "ExpectedAmpHolo_2w" or self.Dataset_Name == "ExpectedAmpHolo_Binarized" or self.Dataset_Name == "phs":
            with open(os.path.join(self.Dataset_Path, self.Dataset_Name, self.Data_list[index]), 'rb') as ExpectedAmpHolo:  #os.path.join()函数用于路径拼接文件路径。os.path.join()函数中可以传入多个路径：
                #with open() as f :'rb': 以二进制格式打开一个文件用于只读
                Ae = tiff.imread(ExpectedAmpHolo) # numpy, (100, 100)
                # # ####################### make the expected image is 0 or 1 #######################
                # x = np.ones_like(Ae)
                # y = np.zeros_like(Ae)
                # Ae = np.where(Ae>=0.2, x, y)
                # assert (np.sum(Ae==1.0) + np.sum(Ae==0.0)) == 50*50, "Image is not 0 or 1!"
                # # #################################################################################
                Ae = Ae.astype(np.float32) #astype 修改数据类型  类似的有 dtype: 数组元素的类型;type: 获取数据类型
                if self.Dataset_Name == "ExpectedAmpHolo_Binarized":
                    assert ((Ae == 1.0) + (Ae == 0.0)).all(), "Ae is not binary image"
                assert (Ae <= 1.0).all() and (Ae >= 0.0).all(), "Ae is out of range [0, 1]"
                Ae = torch.from_numpy(Ae)  #torch.from_numpy()方法把数组转换成张量，且二者共享内存，对张量进行修改比如重新赋值，那么原始数组也会相应发生改变。
            Ae = Ae.unsqueeze(0)  # torch.Size([1, 100, 100])
            return Ae
            # with open(os.path.join(self.Dataset_Path, 'rec_amp', self.Data_list[index]), 'rb') as ExpectedAmpHolo:
            #     Ae = tiff.imread(ExpectedAmpHolo) # numpy, (100, 100)
            #     Ae = Ae.astype(np.float32)
            #     if self.Dataset_Name == "ExpectedAmpHolo_Binarized":
            #         assert ((Ae == 1.0) + (Ae == 0.0)).all(), "Ae is not binary image"
            #     Ae = torch.from_numpy(Ae)
            # Ae = Ae.unsqueeze(0)  # torch.Size([1, 100, 100])
            # with open(os.path.join(self.Dataset_Path, 'phs', self.Data_list[index]), 'rb') as ExpectedAmpHolo:
            #     Ps = tiff.imread(ExpectedAmpHolo) # numpy, (100, 100)
            #     assert (Ps <= 1.0).all(), "Ps > 1.0"
            #     Ps = Ps * 2 * np.pi
            #     Ps = Ps.astype(np.float32)
            #     if self.Dataset_Name == "ExpectedAmpHolo_Binarized":
            #         assert ((Ae == 1.0) + (Ae == 0.0)).all(), "Ae is not binary image"
            #     Ps = torch.from_numpy(Ps)
            # Ps = Ps.unsqueeze(0)  # torch.Size([1, 100, 100])
            # return Ae, Ps

        elif self.Dataset_Name == "StackedAeA1":
            with open(os.path.join(self.Dataset_Path, 'ExperiencePool_'+ self.TIME + "_" + self.NET + "_" + self.NOTE + '/StackedAeA1', self.Data_list[index]), 'rb') as StackedAeA1:
                AeAu = np.load(StackedAeA1)
                AeAu = AeAu.astype(np.float32)
                AeAu = torch.from_numpy(AeAu)
            assert AeAu.shape == torch.Size([2, 100, 100]), "AeAu.shape is not (2, 100, 100)"
            Ae = AeAu[0,:,:]    # torch.Size([100, 100])
            # assert ((Ae == 1.0) + (Ae == 0.0)).all(), "Ae is not binary image"
            Au = AeAu[1,:,:]    # torch.Size([100, 100])
            # Save image to check
            # image = Image.fromarray(Ae.numpy()*255)
            # image = image.convert("L")
            # image.save("./" + "test_Ae.jpg")
            # image = Image.fromarray(Au.numpy()*255)
            # image = image.convert("L")
            # image.save("./" + "test_Au.jpg")
            # plt.subplot(1,2,1)
            # plt.imshow(Ae)
            # plt.colorbar(fraction=0.046, pad=0.04)
            # plt.axis('off')
            # plt.subplot(1,2,2)
            # plt.imshow(Au)
            # plt.colorbar(fraction=0.046, pad=0.04)
            # plt.axis('off')
            # plt.savefig('./StachedAeA1.png')
            # exit()
            Ae = Ae.unsqueeze(0)    # torch.Size([1, 100, 100])
            Au = Au.unsqueeze(0)    # torch.Size([1, 100, 100])
            assert Ae.shape == torch.Size([1, 100, 100]) and Au.shape == torch.Size([1, 100, 100]), "The shapes of Ae and Au are wrong (not [1,50,50])"
            return Ae, Au

        elif self.Dataset_Name == "StackedAePs":
            with open(os.path.join(self.Dataset_Path, 'ExperiencePool_'+ self.TIME + "_" + self.NET + "_" + self.NOTE + '/StackedAePs', self.Data_list[index]), 'rb') as StackedAePs:
                AePs = np.load(StackedAePs)
                AePs = AePs.astype(np.float32)
                AePs = torch.from_numpy(AePs)
            assert AePs.shape == torch.Size([2, 100, 100]), "AeAu.shape is not (2, 100, 100)"
            Ae = AePs[0,:,:]      # [100, 100]
            # assert ((Ae == 1.0) + (Ae == 0.0)).all(), "Ae is not binary image"
            Ps = AePs[1,:,:]      # [100, 100]
            Ae = Ae.unsqueeze(0)  # [1, 100, 100]
            Ps = Ps.unsqueeze(0)  # [1, 100, 100]
            assert Ae.shape == torch.Size([1, 100, 100]) and Ps.shape == torch.Size([1, 100, 100]), "The shapes of Ae and Ps are wrong (not [1,50,50])"
            return Ae, Ps

        elif self.Dataset_Name == "StackedAePsA1binary":
            with open(os.path.join(self.Dataset_Path, 'ExperiencePool_'+ self.TIME + "_" + self.NET + "_" + self.NOTE + '/StackedAePsA1binary', self.Data_list[index]), 'rb') as StackedAePsA1binary:
                AePsA1binary = np.load(StackedAePsA1binary)
                AePsA1binary = AePsA1binary.astype(np.float32)
                AePsA1binary = torch.from_numpy(AePsA1binary)
            assert AePsA1binary.shape == torch.Size([3, 100, 100]), "AeAu.shape is not (2, 100, 100)"
            Ae = AePsA1binary[0,:,:]          # [100, 100]
            Ps = AePsA1binary[1,:,:]          # [100, 100]
            A1binary = AePsA1binary[2,:,:]    # [100, 100]
            # assert ((Ae == 1.0) + (Ae == 0.0)).all(), "Ae is not binary image"
            assert ((Ps <= 2*np.pi) + (Ps >= 0.0)).all(), "Ae is not binary image"
            assert ((A1binary == 1.0) + (A1binary == 0.0)).all(), "A1binary is not binary image"
            Ae = Ae.unsqueeze(0)              # [1, 100, 100]
            Ps = Ps.unsqueeze(0)              # [1, 100, 100]
            A1binary = A1binary.unsqueeze(0)  # [1, 100, 100]
            assert Ae.shape == torch.Size([1, 100, 100]) and Ps.shape == torch.Size([1, 100, 100]) and A1binary.shape == torch.Size([1, 100, 100]), "The shapes of Ae, Ps and A1binary are wrong (not [1,50,50])"
            return Ae, Ps, A1binary

        elif self.Dataset_Name == "ExperiencePool1":
            with open(os.path.join(self.Dataset_Path, 'ExperiencePool_'+ self.TIME + "_" + self.NET + "_" + self.NOTE + '/ExperiencePool1', self.Data_list[index]), 'rb') as ExperiencePool_1:
                AcAs = np.load(ExperiencePool_1)
                AcAs = AcAs.astype(np.float32)
                AcAs = torch.from_numpy(AcAs)
            Ac = AcAs[0,:,:]      # [100, 100]
            As = AcAs[1,:,:]      # [100, 100]
            assert ((Ac <= 1.0) + (Ac >= 0.0)).all(), "Ac is out of range [0, 1]"
            assert ((As <= 1.0) + (As >= 0.0)).all(), "As is out of range [0, 1]"
            Ac = Ac.unsqueeze(0)  # [1, 100, 100]
            As = As.unsqueeze(0)  # [1, 100, 100]
            assert Ac.shape == torch.Size([1, 100, 100]) and As.shape == torch.Size([1, 100, 100]), "The shapes of Ac and As are wrong (not [1,100,100])"
            return Ac, As

        elif self.Dataset_Name == "ExperiencePool2":
            with open(os.path.join(self.Dataset_Path, 'ExperiencePool_'+ self.TIME + "_" + self.NET + "_" + self.NOTE + '/ExperiencePool2', self.Data_list[index]), 'rb') as ExperiencePool_2:
                AeAs = np.load(ExperiencePool_2)
                AeAs = AeAs.astype(np.float32)
                AeAs = torch.from_numpy(AeAs)
            Ae = AeAs[0,:,:]      # [100, 100]
            As = AeAs[1,:,:]      # [100, 100]
            assert ((Ae == 1.0) + (Ae == 0.0)).all(), "Ae is not binary"
            assert ((As <= 1.0) + (As >= 0.0)).all(), "As is out of range [0, 1]"
            Ae = Ae.unsqueeze(0)  # [1, 100, 100]
            As = As.unsqueeze(0)  # [1, 100, 100]
            return Ae, As

    def __len__(self):
        return len(self.Data_list)

 # Calculate running time
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()

# Set data path
# Dataset_Path = '/public/home/liuqing2022/hologram/img/ExpectedAmpHolo/'
Dataset_Path = '/public/home/liuqing2022/hologram/img/'
Dataset_Name = 'ExpectedAmpHolo_Binarized'   # 'ExpectedAmpHolo' or 'ExpectedAmpHolo_2w' or "ExpectedAmpHolo_Binarized"
dataset = MyDataset(Dataset_Name=Dataset_Name, Dataset_Path=Dataset_Path)

# Dataset_Path = '/public/home/zhongchx/HA_2D/2DHologramDL/2DHologram_T-ASE-IROS/Dataset/pack_2k/'
# Dataset_Name = 'phs'   # 'ExpectedAmpHolo' or 'ExpectedAmpHolo_2w' or "ExpectedAmpHolo_Binarized"
# dataset = MyDataset(Dataset_Name=Dataset_Name, Dataset_Path=Dataset_Path)

SHUFFLE = True
Random_Seed = 42
BatchSize = 16
# BatchSize = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
nw = min([os.cpu_count(), BatchSize if BatchSize > 1 else 0, 8])  # number of workers
#nw = 1
print('Using {} dataloader workers every process'.format(nw))

# Dataset splitting
n_train = len(dataset) * 0.8
n_valid = len(dataset) * 0.1
n_test = len(dataset) * 0.1 #可能为小数
print("Using {} for training, {} for validation, {} for testing".format(n_train, n_valid, n_test))

# Create data indices for training, validation and testing
Dataset_Size = len(dataset)
indices = list(range(Dataset_Size))  #range(Dataset_size)=创建一个[0,1,2,3,4,5,....,Datasize-1];list()再转为一个列表
split_fortrain = int(n_train)
split_forvalid = int(n_train + n_valid)  #取整数
if SHUFFLE:
    np.random.seed(Random_Seed)  #np.random.seed(n)函数用于生成指定随机数,seed( ) 用于指定随机数生成时所用算法开始的整数值，如果使用相同的seed( )值，则每次生成的随即数都相同，如果不设置这个值，则系统根据时间来自己选择这个值，此时每次生成的随机数因时间差异而不同。
    np.random.shuffle(indices)
indices_train, indices_valid, indices_test = indices[:split_fortrain], indices[split_fortrain:split_forvalid], indices[split_forvalid:]
print(len(indices_train), len(indices_valid), len(indices_test))
# Create PT data samples and loaders
total_sampler = Data.SubsetRandomSampler(indices)
train_sampler = Data.SubsetRandomSampler(indices_train)
valid_sampler = Data.SubsetRandomSampler(indices_valid)
test_sampler = Data.SubsetRandomSampler(indices_test)
'''
 1. epoch:所有的训练样本输入到模型中称为一个epoch; 
 2. iteration:一批样本输入到模型中,成为一个Iteration;
 3. batchszie:批大小,决定一个epoch有多少个Iteration;   
    以50000张图像的训练集为例,若设train_loader = DataLoader(train_dataset,batch_size=4,shuffle=True,num_workers=num_workers)则相当于将训练集平均分成12500份,
    每份有4张图片(batch_size参数设置的就是每份中有多少张图片).train_loader中的每个元素相当于一个分组,一个组中有4张图片,label就是一个分组中的一张图片的标签,故len(train_loader)==12500,len(label)==4
 4. 迭代次数(iteration)=样本总数(epoch)/批尺寸(batchszie)
 5. dataset (Dataset) --- 决定数据从哪读取或者从何读取;
 6. batch_size (python:int, optional) --- 批尺寸(每次训练样本个数,默认为1)  
 7. shuffle (bool, optional) ---每一个 epoch是否为乱序 (default: False);
 8. num_workers (python:int, optional) --- 是否多进程读取数据(默认为0);
 9. drop_last (bool, optional) --- 当样本数不能被batchsize整除时,最后一批数据是否舍弃(default: False)
 10. pin_memory(bool, optional) --- 如果为True会将数据放置到GPU上去(默认为false) 
'''
total_loader = Data.DataLoader(
    dataset = dataset,
    batch_size = BatchSize,
    sampler=total_sampler,
    num_workers=nw,
    drop_last=True,
)
train_loader = Data.DataLoader(
    dataset = dataset,
    batch_size = BatchSize,
    sampler=train_sampler,
    num_workers=nw,
    drop_last=True,
)
valid_loader = Data.DataLoader(  
 
    dataset = dataset,
    batch_size = BatchSize,
    sampler=valid_sampler,
    num_workers=nw,   #线程
    drop_last=True,
)
test_loader = Data.DataLoader(
    dataset = dataset,
    batch_size = BatchSize,
    sampler=test_sampler,
    num_workers=nw,
    drop_last=True,
)
end.record()
torch.cuda.synchronize() #等待当前设备上所有流中的所有核心完成。正确测试代码在cuda运行时间，需要加上torch.cuda.synchronize()，使用该操作来等待GPU全部执行结束，CPU才可以读取时间信息。
print("Using {} ms for dataset loading".format(start.elapsed_time(end)))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using {} device.".format(device))