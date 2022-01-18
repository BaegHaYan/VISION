import torch
import torch.nn as nn

class GeneratorEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, nker=64):
        super(GeneratorEncoder, self).__init__()

        self.enc1 = DownBlock(in_channels=in_channels, out_channels=nker * 1)
        self.enc2 = DownBlock(in_channels=nker * 1, out_channels=nker * 2)
        self.enc3 = DownBlock(in_channels=nker * 2, out_channels=nker * 4)
        self.enc4 = DownBlock(in_channels=nker * 4, out_channels=nker * 8)
        self.enc5 = DownBlock(in_channels=nker * 8, out_channels=nker * 16)

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.enc5(x)
        x = self.enc6(x)
        return x

class GeneratorDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, nker=64):
        super(GeneratorDecoder, self).__init__()

        self.dec1 = UpBlock(in_channels=in_channels, out_channels=nker * 8)
        self.dec2 = UpBlock(in_channels=nker * 8, out_channels=nker * 4)
        self.dec2 = UpBlock(in_channels=nker * 4, out_channels=nker * 2)
        self.dec2 = UpBlock(in_channels=nker * 2, out_channels=nker * 1)
        self.dec5 = UpBlock(in_channels=nker * 1, out_channels=out_channels)

    def forward(self, x):
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        x = self.dec5(x)
        x = torch.tanh(x)
        return x


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=True, down=True):
        super(DownBlock, self).__init__()

        layers = list()
        layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                            padding=padding, bias=bias)]
        if down:
            layers += [nn.AvgPool2d(kernel_size=2, stride=1)]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        return x

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=True, up=True):
        super(UpBlock, self).__init__()

        layers = list()
        layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                             padding=padding)]
        if up:
            layers += [nn.Upsample(scale_factor=2)]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        return x
