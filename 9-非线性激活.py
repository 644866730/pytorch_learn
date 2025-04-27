import torch
from torch import nn

input = torch.tensor([
                [1,-0.5],
                [-1,3]]
)

input = torch.reshape(input,(-1,1,2,2))
print(input.shape)

class My_moudle(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu1 = nn.ReLU(inplace=False)     #是否保留input    

    def forward(self,input):
        output = self.relu1(input)
        return output
    
my_moudle = My_moudle()
output = my_moudle(input)
print(output)