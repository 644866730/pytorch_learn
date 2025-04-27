from torch.utils.tensorboard.writer import SummaryWriter
import numpy as np
from PIL import  Image

writer = SummaryWriter("logs")     #存放在logs文件夹下

image_path = r"dataset\hymenoptera_data\train\bees\16838648_415acd9e3f.jpg"
image_PIL = Image.open(image_path)
image_array = np.array(image_PIL)
writer.add_image("test",image_array,2,dataformats="HWC")     #因为数据类型原因要加dataformats="HWC"
                 #标题    图片      步数   

for i in range(100):
    writer.add_scalar("y=2x",2*i,i)    # 标题，y , x
# 运行后生成的事件文件用以下命令打开：
# tensorboard --logdir=logs 
# tensorboard --logdir=logs --port=6007   修改端口号    

writer.close()
