from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard.writer import SummaryWriter


img_path = r"dataset\hymenoptera_data\train\bees\95238259_98470c5b10.jpg"
image = Image.open(img_path)      #用cv库读取也可以

writer = SummaryWriter("logs")

# Totensor   将图片转成tensor格式
trans_totensor = transforms.ToTensor()  #用自定义模板创建转换器 
img_tensor = trans_totensor(image)   #转换
writer.add_image("tensor_img",img_tensor)


# Normalize  可以调整输入数据范围
trans_norm = transforms.Normalize([0.5,0.5,0.5] , [0.2,0.2,0.2])    #提供均值和标准差(三通道)
# norm = (input - 均值)/方差
img_norm = trans_norm(img_tensor)
writer.add_image("Normalize",img_norm)


# Resize    缩放图片
trans_resize = transforms.Resize((512,512))   
img_resize = trans_resize(image)     #需要PIL格式的输入,返回也是PIL格式
img_resize = trans_totensor(img_resize)   #再转成tensor
writer.add_image("img_resize",img_resize,0)

# compose
trans_resize2 = transforms.Resize(512)
# Compose()中的参数需要是一个列表,且是是transforms类型
# 上一个的输出是下一个的输入,PIL -> PIL -> tensor
trans_compose = transforms.Compose([trans_resize2,trans_totensor])
img_resize2 = trans_compose(image)
writer.add_image("img_resize",img_resize,1)


writer.close()