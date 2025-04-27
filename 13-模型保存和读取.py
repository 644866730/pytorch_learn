import torchvision
import torch

vgg16 = torchvision.models.vgg16()
#保存方式1
torch.save(vgg16,"vgg16_method1.pth")   #模型结构和参数都被保存下来了
#加载模型1
model1 = torch.load("vgg16_method1.pth")
# print(model1)


#保存方式2
torch.save(vgg16.state_dict(),"vgg16_method2.pth")   #只保留参数，用的空间更小
#加载模型2
model2 = torchvision.models.vgg16()
model2.load_state_dict(torch.load("vgg16_method2.pth"))
# model2 = torch.load("vgg16_method2.pth")     直接加载，只会输出参数字典
print(model2)


#陷阱1     在my_model.py里面保存了一个模型
from my_model import *      #如果不将模型导入进来就如报错
model3 = torch.load("my_moudle_13.pth")
print(model3)
