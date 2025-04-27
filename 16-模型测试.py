import torch
from torch import nn
import torchvision
from my_model import My_moudle
from PIL import Image

image_path = r"test_imgs\image.png"
image = Image.open(image_path)
image = image.convert('RGB')
print(image)

dataset_transfrom = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32,32)),
    torchvision.transforms.ToTensor()
]
)

image = dataset_transfrom(image)
print(image.shape)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#加载模型
model = torch.load("my_moudle_last.pth").to(device)
image = torch.randn(1, 3, 32, 32).to(device) 

model.eval()     #测试模型
with torch.no_grad():
    output = model(image)
print(output)
print(output.argmax(1))        #对应的列别序号
