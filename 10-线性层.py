import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn

dataset_transfrom = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
]
)

dataset = torchvision.datasets.CIFAR10(root="dataset1",train=False,transform=dataset_transfrom,download=False)
dataset_loader = DataLoader(dataset=dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=True)


class My_moudle(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(196608,10)
        
    def forward(self,input):
        output = self.linear1(input)
        return output

my_moudle = My_moudle()   
for data in dataset_loader:
    img,label = data
    img = torch.reshape(img,(1,1,1,-1))   #对图像数据进行展平
    # img = torch.flatten(img)     #同样的展平效果
    output = my_moudle(img)
    print(output)
    # print(output.shape)