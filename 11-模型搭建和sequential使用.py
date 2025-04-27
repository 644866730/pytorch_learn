import torch
from torch import nn
from torch.utils.tensorboard.writer import SummaryWriter

class My_moudle(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,32,5,padding=2)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32,32,5,padding=2)
        self.maxpool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32,64,5,padding=2)
        self.maxpool3 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(1024,64)
        self.linear2 = nn.Linear(64,10)

       #另一种简单写法
        self.model1 = nn.Sequential(
            nn.Conv2d(3,32,5,padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024,64),
            nn.Linear(64,10)
        )

    def forward(self,x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x =self.maxpool2(x)
        x= self.conv3(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x =self.linear2(x) 

        #用简单写法就可以直接写
        # x = self.model1(x)
        return x
    
my_moudle = My_moudle()
print(my_moudle)

#自定义一个尺寸数据，检查网络是否搭建成功
input = torch.ones(64,3,32,32)
output = my_moudle(input)
print(output)

#可视化模型结构
writer = SummaryWriter("log_model")
writer.add_graph(my_moudle,input)
writer.close()