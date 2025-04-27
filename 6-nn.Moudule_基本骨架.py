from torch import nn
import torch

class My_Moudle(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,input):
        output = input+1
        return output
    
moudle = My_Moudle()
x = torch.tensor(1.0)
output = moudle(x)
print(output)