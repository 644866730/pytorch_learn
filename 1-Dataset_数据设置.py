from torch.utils.data import Dataset
from PIL import Image
import os

class MyData(Dataset):
    def __init__(self,root_dir,label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir,self.label_dir)
        self.image_path = os.listdir(self.path)
        
    def __getitem__(self, index):
        img_name = self.image_path[index]
        img_item_path = os.path.join(self.path,img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img,label
    
    def __len__(self):
        return len(self.image_path)


root_dir = r"dataset\hymenoptera_data\train"
ants_label_dir = "ants"
bees_label_dir = "bees"
ants_dataset = MyData(root_dir , ants_label_dir)
bees_dataset = MyData(root_dir , bees_label_dir)

# img,label = ants_dataset[0]
# img.show()
# img,label = bees_dataset[0]
# img.show()

train_dataset = ants_dataset + bees_dataset     #会将数据集拼接




