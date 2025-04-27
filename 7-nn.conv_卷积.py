import torch.nn.functional as F
import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import Conv2d
from torch.utils.tensorboard.writer import SummaryWriter

#示例1  二维卷积简单应用
input = torch.tensor([[1,2,0,3,1],
                     [0,1,2,3,1],
                     [1,2,1,0,0],
                     [5,2,3,1,1],
                     [2,1,0,1,1]])


kernel =  torch.tensor([[1,2,1],
                       [0,1,0],
                       [2,1,0]])

input = torch.reshape(input,(1,1,5,5))   #转变维度，con2d二维卷积需要这个维度
kernel = torch.reshape(kernel,(1,1,3,3))  #依次为 minbatch、chanel、高度、宽度

output = F.conv2d(input,kernel,stride=2,padding=2)
print(output)



#示例2  图像二维卷积
dataset_transfrom = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
]
)

dataset = torchvision.datasets.CIFAR10(root="dataset1",train=False,transform=dataset_transfrom,download=False)
dataset_loader = DataLoader(dataset=dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=False)

class My_moudle(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0)

    def forward(self,input):
        output = self.conv1(input)
        return output

my_moudle = My_moudle()
print(my_moudle)

writer = SummaryWriter("logs")     #存放在logs文件夹下
step = 1
for data in dataset_loader:    #遍历数据集
    img , label = data         #得到图片和标签
    output = my_moudle(img)
    # print(output)
    print(img.shape)
    print(output.shape)
    # torch.Size([64, 3, 32, 32])    输入
    writer.add_image("input",img,step,dataformats='NCHW')
    # torch.Size([64, 6, 30, 30])    输出
    output = torch.reshape(output,(128,3,30,30))   #6通道没办法显示，必须改变形状,增加batch_size,把多余图片分到另一个批次
    writer.add_image("output",output,step,dataformats='NCHW')
    step += 1

writer.close()

