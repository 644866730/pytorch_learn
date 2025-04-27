import torchvision

dataset_transfrom = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
]
)

train_set = torchvision.datasets.CIFAR10(root="dataset1",train=True,transform=dataset_transfrom,download=False)
test_set = torchvision.datasets.CIFAR10(root="dataset1",train=False,transform=dataset_transfrom,download=False)

print(test_set[0])

# img, label = test_set[0]
# img.show()
# print(label)
