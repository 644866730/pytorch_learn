import torchvision
from torch.utils.data import DataLoader

dataset_transfrom = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
]
)

test_data = torchvision.datasets.CIFAR10(root="dataset1",train=False,transform=dataset_transfrom,download=False)

# shuffle:每轮的数据选取顺序是否打乱
# drop_last:是否丢弃每批次余下的数据，例如101个数据，batch_size=4，每轮最后一个数据是否丢弃
test_loader = DataLoader(dataset=test_data, batch_size=4, shuffle=True, num_workers=0, drop_last=False)

img,label = test_data[0]
print(img.shape)
print(label)

