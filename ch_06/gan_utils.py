import torch
import torch.nn as nn


IMG_WIDTH = 256
IMG_HEIGHT = 256
NUM_CHANNELS = 3
BATCH_SIZE = 64
N_EPOCHS = 100
SAMPLE_INTERVAL = 18

# prepare patch size for our setup
patch = int(IMG_HEIGHT / 2**4)
PATCH_GAN_SHAPE = (1,patch, patch)


class DownSampleBlock(nn.Module):
    def __init__(self, input_channels, output_channels,normalize=True):
        super(DownSampleBlock, self).__init__()
        layers = [
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False)
            ]
        if normalize:
          layers.append(nn.InstanceNorm2d(output_channels))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Dropout(0.5))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UpSampleBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(UpSampleBlock, self).__init__()
        layers = [
            nn.ConvTranspose2d(
                input_channels,
                output_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False),
        ]
        layers.append(nn.InstanceNorm2d(output_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(0.5))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_connection):
        x = self.model(x)
        x = torch.cat((x, skip_connection), 1)

        return x

class Generator(nn.Module):
    def __init__(self, input_channels=3,out_channels=3):
        super(Generator, self).__init__()

        self.downsample1 = DownSampleBlock(input_channels,64, normalize=False)
        self.downsample2 = DownSampleBlock(64, 128)
        self.downsample3 = DownSampleBlock(128, 256)
        self.downsample4 = DownSampleBlock(256, 512)
        self.downsample5 = DownSampleBlock(512, 512)
        self.downsample6 = DownSampleBlock(512, 512)
        self.downsample7 = DownSampleBlock(512, 512)
        self.downsample8 = DownSampleBlock(512, 512,normalize=False)

        self.upsample1 = UpSampleBlock(512, 512)
        self.upsample2 = UpSampleBlock(1024, 512)
        self.upsample3 = UpSampleBlock(1024, 512)
        self.upsample4 = UpSampleBlock(1024, 512)
        self.upsample5 = UpSampleBlock(1024, 256)
        self.upsample6 = UpSampleBlock(512, 128)
        self.upsample7 = UpSampleBlock(256, 64)

        self.final_layer = nn.Sequential(
            nn.Upsample(scale_factor=2),
            # padding left, right, top, bottom
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        # downsampling blocks
        d1 = self.downsample1(x)
        d2 = self.downsample2(d1)
        d3 = self.downsample3(d2)
        d4 = self.downsample4(d3)
        d5 = self.downsample5(d4)
        d6 = self.downsample6(d5)
        d7 = self.downsample7(d6)
        d8 = self.downsample8(d7)
        # upsampling blocks with skip connections
        u1 = self.upsample1(d8, d7)
        u2 = self.upsample2(u1, d6)
        u3 = self.upsample3(u2, d5)
        u4 = self.upsample4(u3, d4)
        u5 = self.upsample5(u4, d3)
        u6 = self.upsample6(u5, d2)
        u7 = self.upsample7(u6, d1)

        return self.final_layer(u7)

class Discriminator(nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(input_filters, output_filters):
            layers = [
                nn.Conv2d(
                    input_filters,
                    output_filters,
                    kernel_size=4,
                    stride=2,
                    padding=1)
                ]
            layers.append(nn.InstanceNorm2d(output_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(input_channels * 2, output_filters=64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            # padding left, right, top, bottom
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)

