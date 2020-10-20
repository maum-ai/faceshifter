import torch
import torch.nn as nn

class MultilevelAttributesEncoder(nn.Module):
    def __init__(self):
        super(MultilevelAttributesEncoder, self).__init__()
        self.Encoder_channel = [3, 32, 64, 128, 256, 512, 1024, 1024]
        self.Encoder = nn.ModuleDict({f'layer_{i}' : nn.Sequential(
                nn.Conv2d(self.Encoder_channel[i], self.Encoder_channel[i+1], kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(self.Encoder_channel[i+1]),
                nn.LeakyReLU(0.1)
            )for i in range(7)})

        self.Decoder_inchannel = [1024, 2048, 1024, 512, 256, 128]
        self.Decoder_outchannel = [1024, 512, 256, 128, 64, 32]
        self.Decoder = nn.ModuleDict({f'layer_{i}' : nn.Sequential(
                nn.ConvTranspose2d(self.Decoder_inchannel[i], self.Decoder_outchannel[i], kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(self.Decoder_outchannel[i]),
                nn.LeakyReLU(0.1)
            )for i in range(6)})

        self.Upsample = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):
        arr_x = []
        for i in range(7):
            x = self.Encoder[f'layer_{i}'](x)
            arr_x.append(x)


        arr_y = []
        arr_y.append(arr_x[6])
        y = arr_x[6]
        for i in range(6):
            y = self.Decoder[f'layer_{i}'](y)
            y = torch.cat((y, arr_x[5-i]), 1)
            arr_y.append(y)

        arr_y.append(self.Upsample(y))

        return arr_y


class ADD(nn.Module):
    def __init__(self, h_inchannel, z_inchannel, z_id_size=256):
        super(ADD, self).__init__()

        self.BNorm = nn.BatchNorm2d(h_inchannel)
        self.conv_f = nn.Conv2d(h_inchannel, h_inchannel, kernel_size=3, stride=1, padding=1)

        self.sigmoid = nn.Sigmoid()

        self.fc_1 = nn.Linear(z_id_size, h_inchannel)
        self.fc_2 = nn.Linear(z_id_size, h_inchannel)

        self.conv1 = nn.Conv2d(z_inchannel, h_inchannel, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(z_inchannel, h_inchannel, kernel_size=3, stride=1, padding=1)

    def forward(self, h_in, z_att, z_id):
        h_bar = self.BNorm(h_in)
        m = self.sigmoid(self.conv_f(h_bar))

        r_id = self.fc_1(z_id).unsqueeze(-1).unsqueeze(-1).expand_as(h_in)
        beta_id = self.fc_2(z_id).unsqueeze(-1).unsqueeze(-1).expand_as(h_in)

        i = r_id*h_bar + beta_id

        r_att = self.conv1(z_att)
        beta_att = self.conv2(z_att)
        a = r_att * h_bar + beta_att

        h_out = (1-m)*a + m*i

        return h_out


class ADDResBlock(nn.Module):
    def __init__(self, h_inchannel, z_inchannel, h_outchannel):
        super(ADDResBlock, self).__init__()

        self.h_inchannel = h_inchannel
        self.z_inchannel = z_inchannel
        self.h_outchannel = h_outchannel

        self.add1 = ADD(h_inchannel, z_inchannel)
        self.add2 = ADD(h_inchannel, z_inchannel)

        self.conv1 = nn.Conv2d(h_inchannel, h_inchannel, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(h_inchannel, h_outchannel, kernel_size=3, stride=1, padding=1)

        if not self.h_inchannel == self.h_outchannel:
            self.add3 = ADD(h_inchannel, z_inchannel)
            self.conv3 = nn.Conv2d(h_inchannel, h_outchannel, kernel_size=3, stride=1, padding=1)

        self.activation = nn.ReLU()

    def forward(self, h_in, z_att, z_id):
        x1 = self.activation(self.add1(h_in, z_att, z_id))
        x1 = self.conv1(x1)
        x1 = self.activation(self.add2(x1, z_att, z_id))
        x1 = self.conv2(x1)

        x2 = h_in
        if not self.h_inchannel == self.h_outchannel:
            x2 = self.activation(self.add3(h_in, z_att, z_id))
            x2 = self.conv3(x2)

        return x1 + x2


class ADDGenerator(nn.Module):
    def __init__(self, z_id_size):
        super(ADDGenerator, self).__init__()

        self.convt = nn.ConvTranspose2d(z_id_size, 1024, kernel_size=2, stride=1, padding=0)
        self.Upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        self.h_inchannel = [1024, 1024, 1024, 1024, 512, 256, 128, 64]
        self.z_inchannel = [1024, 2048, 1024, 512, 256, 128, 64, 64]
        self.h_outchannel = [1024, 1024, 1024, 512, 256, 128, 64, 3]

        self.model = nn.ModuleDict(
            {f"layer_{i}" : ADDResBlock(self.h_inchannel[i], self.z_inchannel[i], self.h_outchannel[i])
        for i in range(8)})


    def forward(self, z_id, z_att):
        x = self.convt(z_id.unsqueeze(-1).unsqueeze(-1))

        for i in range(7):
            x = self.Upsample(self.model[f"layer_{i}"](x, z_att[i], z_id))
        x = self.model["layer_7"](x, z_att[7], z_id)

        return nn.Sigmoid()(x)