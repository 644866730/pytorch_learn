import torch
from torch import nn
from torch.utils.tensorboard.writer import SummaryWriter
import torchvision
from torch.utils.data import DataLoader

dataset_transfrom = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
]
)

test_data = torchvision.datasets.CIFAR10(root="dataset1",train=False,transform=dataset_transfrom,download=False)

test_loader = DataLoader(dataset=test_data, batch_size=1)

class My_moudle(nn.Module):
    def __init__(self):
        super().__init__()
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
        x = self.model1(x)
        return x
    
my_moudle = My_moudle()
print(my_moudle)

loss = nn.CrossEntropyLoss()    #交叉熵损失
optim = torch.optim.SGD(my_moudle.parameters(), lr=0.01)   #设置优化器

for epoch in range(20):
    loss_sum = 0.0
    for data in test_loader:
        img,label = data
        
        output = my_moudle(img)
        result_loss = loss(output,label)
        # print(result_loss)
        optim.zero_grad()        #先将上一次梯度清零
        result_loss.backward()     #反向传播得到每个节点的梯度
        optim.step()        #参数更新调优
        loss_sum += result_loss      #每一轮所有数据的误差
    print(loss_sum)