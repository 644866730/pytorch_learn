import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from my_model import My_moudle
from torch.utils.tensorboard.writer import SummaryWriter

#这种方式更常用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')      #-------------------->定义训练设备

dataset_transfrom = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
]
)

#准备数据
tarin_data = torchvision.datasets.CIFAR10(root="dataset1",train=True,transform=dataset_transfrom,download=False)
test_data = torchvision.datasets.CIFAR10(root="dataset1",train=False,transform=dataset_transfrom,download=False)

train_loader = DataLoader(dataset=tarin_data, batch_size=64)
test_loader = DataLoader(dataset=test_data, batch_size=64)

#创建网络模型    
my_moudle = My_moudle()
my_moudle.to(device)                        #---------------------------------->模型转移到device上                       

#定义损失函数
loss_fn = nn.CrossEntropyLoss()     
loss_fn.to(device)                          #---------------------------------->损失函数转移到device上  

#定义优化器
learning_rate = 0.01
optimizer = torch.optim.SGD(my_moudle.parameters(), lr=learning_rate)

#设置训练参数
total_train_step = 0    
total_test_step = 0   
epoch = 10          
writer = SummaryWriter("logs_train") 

for i in range(epoch):
    total_train_loss = 0.0
    total_test_loss = 0.0
    total_accuracy = 0.0    
    print("-------第{}轮训练开始--------".format(i+1))
    for data in train_loader:
        imgs,labels = data
        imgs = imgs.to(device)                         #---------------------------------->数据转移到device上  
        labels = labels.to(device)                     #---------------------------------->数据转移到device上 

        outputs = my_moudle(imgs)
        loss = loss_fn(outputs,labels)  
        optimizer.zero_grad()          
        loss.backward()               
        optimizer.step()               
        total_train_loss += loss
       
    print("第{}轮整体训练集loss:{}".format(i+1,total_train_loss))
    writer.add_scalar('train_loss',total_train_loss,i+1)
    #测试每一轮训练结果
    with torch.no_grad():
        for data in test_loader:
            imgs , labels = data 
            imgs = imgs.to(device)                         #---------------------------------->数据转移到device上  
            labels = labels.to(device)                     #---------------------------------->数据转移到device上 
            outputs = my_moudle(imgs)
            loss = loss_fn(outputs,labels)
            total_test_loss += loss   
            accuracy = (outputs.argmax(1) == labels).sum()   
            total_accuracy += accuracy     
    print("第{}轮整体训练集accuracy:{}".format(i+1,total_accuracy/len(test_data)))
    writer.add_scalar('test_accuracy',total_accuracy/len(test_data),i+1) 
    print("第{}轮整体测试集loss:{}".format(i+1,total_test_loss))
    writer.add_scalar('test_loss',total_test_loss,i+1)
    if i==9 :
        torch.save(my_moudle,"my_moudle_last.pth")
writer.close()