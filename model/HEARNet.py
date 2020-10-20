import torch
import torch.nn as nn



class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.Encoder_channel = [6, 32, 64, 128, 256, 512, ]
        self.Encoder = nn.ModuleDict({f'layer_{i}': nn.Sequential(
            nn.Conv2d(self.Encoder_channel[i], self.Encoder_channel[i + 1], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.Encoder_channel[i + 1]),
            nn.LeakyReLU(0.1)
        ) for i in range(len(self.Encoder_channel) - 1)})


        self.Decoder_inchannel = [512, 512, 256, 128, 64]
        self.Decoder_outchannel = [256, 128, 64, 32, 3]

        self.Decoder = nn.ModuleDict()
        for i in range(len(self.Decoder_inchannel)):
            modules = [
                nn.ConvTranspose2d(self.Decoder_inchannel[i], self.Decoder_outchannel[i], kernel_size=4, stride=2,
                                   padding=1),
                nn.BatchNorm2d(self.Decoder_outchannel[i]),
                nn.LeakyReLU(0.1),
            ]
            self.Decoder[f'layer_{i}'] = nn.Sequential(*modules)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        arr_x = []
        for i in range(5):
            x = self.Encoder[f'layer_{i}'](x)
            arr_x.append(x)

        y = arr_x[4]
        for i in range(5):
            y = self.Decoder[f'layer_{i}'](y)
            if i < 4:
                y = torch.cat((y, arr_x[3 - i]), dim=1)

        return y