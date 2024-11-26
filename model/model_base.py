import torch
import torch.nn    as nn
import torchvision as tv


class BasicConvolution(nn.Module):
    def __init__(self, input_ch : int, output_ch : int, kernel_size : int) -> None:
        super().__init__()

        self.convolution   = nn.Conv2d(input_ch, output_ch, kernel_size = kernel_size, padding = 'same')
        self.normalization = nn.BatchNorm2d(output_ch)
        self.activation    = nn.LeakyReLU()
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.convolution(x)
        x = self.normalization(x)
        x = self.activation(x)
        return x

class BasicConvBlock(nn.Module):
    def __init__(self, input_ch : int, output_ch : int, kernel_size : int) -> None:
        super().__init__()
        self.conv = BasicConvolution(input_ch, output_ch, kernel_size)
        self.pool = nn.MaxPool2d(kernel_size = 3, stride = 3)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x

class SimpleCNN(nn.Module):
    def __init__(self, output_class : int) -> None:
        super().__init__()

        self.block_1 = BasicConvBlock(3, 96, 11)
        self.block_2 = BasicConvBlock(96, 256, 5)
        self.conv_3 = nn.Conv2d(256, 384, kernel_size = 3, padding = 'same')
        # self.block_3 = BasicConvBlock(128, 256, 3)
        self.block_4 = BasicConvBlock(384, 256, 3)
        # self.pool = nn.MaxPool2d(kernel_size = 3, stride = 2)
        self.flatten = nn.Flatten()

        self.head = nn.Sequential(
            nn.Dropout(0.5),
            # nn.Linear(384, 384//2),
            # nn.LeakyReLU(),
            nn.Linear(4096, output_class),
            # nn.Linear
        )
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:

        x = self.block_1(x) # (N,  3, 256, 256) => (N,  96,  85,  85)
        x = self.block_2(x) # (N,  96,  85,  85) => (N,  256,  28,  28)
        x = self.conv_3(x)
        x = self.block_4(x) # (N,  384,  28,  28) => (N,  256,   9,   9)        
        x = self.flatten(x) # (N, 256,   4,   4) => (N, 4096)    

        # print(x.size())

        # RESHAPE OPERATION
        bz = x.size(0)      # batch   size
        cz = x.size(1)      # channel size 
        x  = x.view(bz, cz) # (N, 4096,   1,   1) => (N, 4096)  ## 4D Tensor => 2D Tensor
        
        x = self.head(x)    # (N, 4096)           => (N, output_class)
        return x

class BasicMobileNet(nn.Module):
    def __init__(self, output_classes : int) -> None:
        super().__init__()

        self.base = tv.models.mobilenet_v3_small(weights = tv.models.MobileNet_V3_Small_Weights.DEFAULT)
        self.base.classifier = nn.Sequential(
            nn.Linear(576, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, output_classes)
        )
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.base(x)
        return x

if __name__ == "__main__":
    print("Model Base Run")

    t     = torch.rand(1, 3, 64, 64)
    model = SimpleCNN(11)
    y     = model(t)
    print(y)