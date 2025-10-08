import torch
import torch.nn as nn

 
class GDSA(nn.Module):
    """Spatial-attention module."""
 
    def __init__(self, channel, kernel_size=3):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.a = int(channel/4)
        self.dilation = [1,2,3,4]
        self.cv2 = nn.Conv2d(self.a, self.a, kernel_size, padding=self.dilation[0]*(kernel_size-1)//2, bias=False, dilation=self.dilation[0],groups=self.a)
        self.cv3 = nn.Conv2d(self.a, self.a, kernel_size, padding=self.dilation[1]*(kernel_size-1)//2, bias=False, dilation=self.dilation[1],groups=self.a)
        self.cv4 = nn.Conv2d(self.a, self.a, kernel_size, padding=self.dilation[2]*(kernel_size-1)//2, bias=False, dilation=self.dilation[2],groups=self.a)
        self.cv5 = nn.Conv2d(self.a, self.a, kernel_size, padding=self.dilation[3]*(kernel_size-1)//2, bias=False, dilation=self.dilation[3],groups=self.a)
        
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()
 
    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        y = list(x.chunk(4, 1))
        y[0] = self.cv2(y[0])
        y[1] = self.cv3(y[1])
        y[2] = self.cv4(y[2])
        y[3] = self.cv5(y[3])
        y = torch.cat(y, 1)
        return y * self.act(self.cv1(torch.cat([torch.mean(y, 1, keepdim=True), torch.max(y, 1, keepdim=True)[0]], 1)))
 
