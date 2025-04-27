import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.tensorboard.writer import SummaryWriter

# 池化简单应用
input = torch.tensor([[1,2,0,3,1],
                     [0,1,2,3,1],
                     [1,2,1,0,0],
                     [5,2,3,1,1],
                     [2,1,0,1,1]],dtype=torch.float32)

input = torch.reshape(input,(-1,1,5,5))   # nn.maxpooling需要batch_size

class My_moudle(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3 , ceil_mode=False)    #不足3x3是否保留

    def forward(self,input):
        output = self.maxpool1(input)
        return output

my_moudle = My_moudle()
output = my_moudle(input)
print(output)


# 图像二维卷积
dataset_transfrom = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
]
)

dataset = torchvision.datasets.CIFAR10(root="dataset1",train=False,transform=dataset_transfrom,download=False)
dataset_loader = DataLoader(dataset=dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=False)

writer = SummaryWriter("logs")     #存放在logs文件夹下
step = 1
for data in dataset_loader:
    img , label = data
    output = my_moudle(img)
    print(output)
    writer.add_image("input_pool",img,step,dataformats='NCHW')
    writer.add_image("output_pool",output,step,dataformats='NCHW')
    step += 1

writer.close()