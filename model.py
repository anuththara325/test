import torch
import torch.nn as nn
import torch.nn.functional as F



class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_act, **kwargs):
        super().__init__()
        self.cnn = nn.Conv2d(
            in_channels,
            out_channels,
            **kwargs,
            bias=True,
        )
        self.act = nn.LeakyReLU(0.2, inplace=True) if use_act else nn.Identity()

    def forward(self, x):
        return self.act(self.cnn(x))


# class UpsampleBlock(nn.Module):
#     def __init__(self, in_c, scale_factor=2):
#         super().__init__()
#         self.upsample = nn.Upsample(scale_factor=scale_factor, mode="nearest")
#         self.conv = nn.Conv2d(in_c, in_c, 3, 1, 1, bias=True)
#         self.act = nn.LeakyReLU(0.2, inplace=True)

#     def forward(self, x):
#         return self.act(self.conv(self.upsample(x)))

import torch.nn as nn

class UpsampleBlock(nn.Module):
    def __init__(self, in_c, scale_factor=2):
        super().__init__()
        self.conv = nn.Conv2d(in_c, in_c * (scale_factor ** 2), 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.pixel_shuffle(self.conv(x)))


class DenseResidualBlock(nn.Module):
    def __init__(self, in_channels, channels=32, residual_beta=0.2):
        super().__init__()
        self.residual_beta = residual_beta
        self.blocks = nn.ModuleList()

        for i in range(5):
            kernel_size = 3 if i % 2 == 0 else 5  # Example: alternating kernel sizes 3 and 5
            use_act = i <= 3  # Use activation for first 4 blocks
            out_channels = channels if i <= 3 else in_channels

            self.blocks.append(
                ConvBlock(
                    in_channels + channels*i,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=kernel_size // 2,  # Assuming padding to keep spatial dimensions unchanged
                    use_act=use_act
                )
            )

    def forward(self, x):
        new_inputs = x
        for block in self.blocks:
            out = block(new_inputs)
            new_inputs = torch.cat([new_inputs, out], dim=1)

        return self.residual_beta * out + x

class RRDB(nn.Module):
    def __init__(self, in_channels, residual_beta=0.2):
        super().__init__()
        self.residual_beta = residual_beta
        self.rrdb = nn.Sequential(*[DenseResidualBlock(in_channels) for _ in range(3)])

    def forward(self, x):
        return self.rrdb(x) * self.residual_beta + x

import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        query = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)  # B x N x C
        key = self.key(x).view(batch_size, -1, width * height)  # B x C x N
        energy = torch.bmm(query, key)  # B x N x N
        attention = F.softmax(energy, dim=-1)  # B x N x N
        value = self.value(x).view(batch_size, -1, width * height)  # B x C x N
        out = torch.bmm(value, attention.permute(0, 2, 1))  # B x C x N
        out = out.view(batch_size, C, width, height)  # B x C x W x H
        out = self.gamma * out + x
        return out

    
class Generator(nn.Module):
    def __init__(self, in_channels=3, num_channels=64, num_blocks=15):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, num_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_channels),
            SelfAttention(num_channels)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, num_channels, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(num_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_channels, num_channels, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(num_channels),
            SelfAttention(num_channels)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, num_channels, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(num_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_channels, num_channels, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(num_channels),
            SelfAttention(num_channels)

        )
        
        self.residuals = nn.Sequential(*[RRDB(num_channels*3) for _ in range(num_blocks)])
        self.conv = nn.Conv2d(num_channels*3, num_channels*3, kernel_size=3, stride=1, padding=1)
        self.attention =  SelfAttention(num_channels*3)
        self.upsamples = nn.Sequential(
            UpsampleBlock(num_channels*3), UpsampleBlock(num_channels*3),
        )
        self.final = nn.Sequential(
            nn.Conv2d(num_channels*3, num_channels*3, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_channels*3, in_channels, 3, 1, 1, bias=True),
        )

    def forward(self, x):
        # Pass input through parallel branches with different kernel sizes and strides
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        
        # Concatenate outputs from parallel branches along the channel dimension
        out = torch.cat([out1, out2, out3], dim=1)
        x = self.conv(self.residuals(out)) + out
        x = self.attention(x)
        x = self.upsamples(x)
        return self.final(x)