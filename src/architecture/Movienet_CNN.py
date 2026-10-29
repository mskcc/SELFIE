
import torch
import torch.nn as nn
import torch.nn.functional as F


# define the CNN architecture
class MovieNet(nn.Module):
    def __init__(self, Ninputchannels, Noutputchannels):
        super(MovieNet, self).__init__()
        # # Enconder
        self.residual_block_down_1 = ResidualBlockDown(channels=[Ninputchannels, 128, 128])  # 256x256x128
        self.bn1 = nn.BatchNorm2d(128)
        self.maxpool1 = nn.MaxPool2d(2, 2)  # 128x128x128
        self.residual_block_down_2 = ResidualBlockDown(channels=[128, 256, 256])  # 128x128x256
        self.bn2 = nn.BatchNorm2d(256)
        self.maxpool2 = nn.MaxPool2d(2, 2)  # 64x64x256
        self.residual_block_down_3 = ResidualBlockDown(channels=[256, 512, 512])  # 64x64x512
        self.bn3 = nn.BatchNorm2d(512)
        self.maxpool3 = nn.MaxPool2d(2, 2)  # 32x32x512
        self.residual_block_down_4 = ResidualBlockDown(channels=[512, 1024, 1024])  # 32x32x1024
        self.bn4 = nn.BatchNorm2d(1024)
        self.maxpool4 = nn.MaxPool2d(2, 2)  # 16x16x1024
        self.residual_block_down_5 = ResidualBlockDown(channels=[1024, 2048, 2048])  # 16x16x2048
        self.bn5 = nn.BatchNorm2d(2048)  # 16x16x2048
        self.up5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 32x32x2048
        #
        # # Decoder
        self.residual_block_up_6 = ResidualBlockUp(channels=[1024+2048, 1024, 1024])  # 32x32x1024
        self.bn6 = nn.BatchNorm2d(1024)
        self.up6 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 64x64x512
        self.residual_block_up_7 = ResidualBlockUp(channels=[512+1024, 512, 512])  # 64x64x512
        self.bn7 = nn.BatchNorm2d(512)
        self.up7 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 128x128x512
        self.residual_block_up_8 = ResidualBlockUp(channels=[256+512, 256, 256])  # 128x128x256
        self.bn8 = nn.BatchNorm2d(256)
        self.up8 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 256x256x256
        self.residual_block_up_9 = ResidualBlockUp(channels=[128+256, 128, 128])  # 256x256x128
        self.bn9 = nn.BatchNorm2d(128)
        #
        # output
        self.convOutput = nn.Conv2d(128, Noutputchannels, 1, padding=0)  # 60pts
        #
        # dropout layer (p=0.01)
        self.dropout = nn.Dropout(0.01)

    def forward_Movienet(self, x):
        xn = x
        #
        # Encoder
        c1 = self.bn1(self.residual_block_down_1(xn))  # 256x256x128
        p1 = self.maxpool1(c1)  # 128x128x128
        c2 = self.bn2(self.residual_block_down_2(p1))  # 128x128x256
        p2 = self.maxpool2(c2)  # 64x64x256
        c3 = self.bn3(self.residual_block_down_3(p2))  # 64x64x512
        p3 = self.maxpool3(c3)  # 32x32x512
        c4 = self.bn4(self.residual_block_down_4(p3))  # 32x32x1024
        p4 = self.maxpool4(c4)  # 16x16x1024
        c5 = self.bn5(self.residual_block_down_5(p4))  # 16x16x2048
        #
        # Decoder
        p5 = self.up5(c5)  # 32x32x2048
        u6 = torch.cat((p5, c4), 1)  # 32x32x(2048+1024)
        c6 = self.bn6(self.residual_block_up_6(u6))  # 32x32x1024
        p6 = self.up6(c6)  # 64x64x1024
        u7 = torch.cat((p6, c3), 1)  # 64x64x(1024+512)
        c7 = self.bn7(self.residual_block_up_7(u7))  # 64x64x512
        p7 = self.up7(c7)  # 128x128x512
        u8 = torch.cat((p7, c2), 1)  # 128x128x(512+256)
        c8 = self.bn8(self.residual_block_up_8(u8))  # 128x128x256
        p8 = self.up8(c8)  # 256x256x256
        u9 = torch.cat((p8, c1), 1)  # 256x256x(256+128)
        c9 = self.bn9(self.residual_block_up_9(u9))  # 256x256x128
        #
        # Output
        x = self.convOutput(c9)
        x = F.relu(x)
        return x

    def forward(self, x):
        x = self.forward_Movienet(x)
        return x


class ResidualBlockDown(torch.nn.Module):
    """ Helper Class"""

    def __init__(self, channels):
        super(ResidualBlockDown, self).__init__()

        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=channels[0],
                            out_channels=channels[1],
                            kernel_size=(3, 3),
                            stride=(1, 1),
                            padding=1),
            torch.nn.BatchNorm2d(channels[1]),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=channels[1],
                            out_channels=channels[2],
                            kernel_size=(3, 3),
                            stride=(1, 1),
                            padding=1),
            torch.nn.BatchNorm2d(channels[2])
        )

        self.shortcut = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=channels[0],
                            out_channels=channels[2],
                            kernel_size=(3, 3),
                            stride=(1, 1),
                            padding=1),
            torch.nn.BatchNorm2d(channels[2])
        )

    def forward(self, x):
        shortcut = x

        block = self.block(x)
        shortcut = self.shortcut(x)
        x = torch.nn.functional.relu(block + shortcut)

        return x


class ResidualBlockUp(torch.nn.Module):
    """ Helper Class"""

    def __init__(self, channels):
        super(ResidualBlockUp, self).__init__()

        self.block = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=channels[0],
                                     out_channels=channels[1],
                                     kernel_size=(3, 3),
                                     stride=(1, 1),
                                     padding=1),
            torch.nn.BatchNorm2d(channels[1]),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=channels[1],
                            out_channels=channels[2],
                            kernel_size=(3, 3),
                            stride=(1, 1),
                            padding=1),
            torch.nn.BatchNorm2d(channels[2])
        )

        self.shortcut = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=channels[0],
                                     out_channels=channels[2],
                                     kernel_size=(3, 3),
                                     stride=(1, 1),
                                     padding=1),
            torch.nn.BatchNorm2d(channels[2])
        )

    def forward(self, x):
        shortcut = x

        block = self.block(x)
        shortcut = self.shortcut(x)
        x = torch.nn.functional.relu(block + shortcut)

        return x


class BlockUp(torch.nn.Module):
    """ Helper Class"""

    def __init__(self, in_channel, out_channel, bottleneck=False, bias=False):
        super(BlockUp, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.bias = bias

        self.scale_factor = (1, 1, 1)
        self.kernel_size_spatial = (1, 1, 1)
        self.stride_spatial = (1, 1, 1)
        self.padding_spatial = (0, 0, 0)
        self.kernel_size_temp = (3, 1, 1)
        self.stride_temp = (1, 1, 1)
        self.padding_temp = (1, 0, 0)

    def create_block(self):
        self.block = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=self.scale_factor),
            torch.nn.Conv3d(in_channels=self.in_channel,
                            out_channels=self.out_channel,
                            kernel_size=self.kernel_size_spatial,
                            stride=self.stride_spatial,
                            padding=self.padding_spatial,
                            padding_mode='reflect',
                            bias=self.bias),
            torch.nn.InstanceNorm3d(self.out_channel),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Conv3d(in_channels=self.out_channel,
                            out_channels=self.out_channel,
                            kernel_size=self.kernel_size_temp,
                            stride=self.stride_temp,
                            padding=self.padding_temp,
                            padding_mode='reflect',
                            bias=self.bias),
            torch.nn.InstanceNorm3d(self.out_channel),
            torch.nn.LeakyReLU(inplace=True),
            )
        return self.block

    def forward(self, x):
        block = self.block(x)
        return block


class BlockDown2Dt(torch.nn.Module):
    def __init__(
            self, in_channel, out_channel, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), drop_prob=0):
        super(BlockDown2Dt, self).__init__()

        kernel_size_temp = (3, 1, 1)
        stride_temp = (1, 1, 1)

        self.block = torch.nn.Sequential(
                torch.nn.Conv3d(in_channels=in_channel,
                                out_channels=out_channel,
                                kernel_size=(1, 3, 3),
                                stride=stride,
                                padding=(0, 1, 1),
                                padding_mode='reflect',
                                bias=False),
                torch.nn.InstanceNorm3d(out_channel),
                torch.nn.LeakyReLU(inplace=True),
                torch.nn.Conv3d(in_channels=out_channel,
                                out_channels=out_channel,
                                kernel_size=kernel_size_temp,
                                stride=stride_temp,
                                padding=(1, 0, 0),
                                padding_mode='reflect',
                                bias=False),
                torch.nn.InstanceNorm3d(out_channel),
                torch.nn.LeakyReLU(inplace=True),
                torch.nn.Dropout3d(drop_prob),
                torch.nn.Conv3d(in_channels=out_channel,
                                out_channels=out_channel,
                                kernel_size=(1, 3, 3),
                                stride=stride,
                                padding=(0, 1, 1),
                                padding_mode='reflect',
                                bias=False),
                torch.nn.InstanceNorm3d(out_channel),
                torch.nn.LeakyReLU(inplace=True),
                torch.nn.Conv3d(in_channels=out_channel,
                                out_channels=out_channel,
                                kernel_size=kernel_size_temp,
                                stride=stride_temp,
                                padding=(1, 0, 0),
                                padding_mode='reflect',
                                bias=False),
                torch.nn.InstanceNorm3d(out_channel),
            )
        self.Leaky = torch.nn.LeakyReLU()

        self.identity = torch.nn.Sequential(
            torch.nn.Conv3d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=(1, 1, 1),
                bias=False),
            nn.InstanceNorm3d(out_channel))

    def forward(self, x):
        block = self.block(x)
        id = self.identity(x)
        return self.Leaky(block + id)


class BlockUp2Dt(BlockUp):
    def __init__(self, in_channel, out_channel, bottleneck=False, bias=False, no_scaling=False, spatial=False):
        super().__init__(in_channel, out_channel, bottleneck, bias)

        if bottleneck is True:
            self.scale_factor = (1.16, 1, 1)
            self.kernel_size_spatial = (1, 1, 1)
            self.stride_spatial = (1, 1, 1)
            self.padding_spatial = (0, 0, 0)
            self.kernel_size_temp = (3, 1, 1)
            self.stride_temp = (1, 1, 1)
            self.padding_temp = (1, 0, 0)
        else:
            self.scale_factor = (2, 2, 2)
            self.kernel_size_spatial = (1, 3, 3)
            self.stride_spatial = (1, 1, 1)
            self.padding_spatial = (0, 1, 1)
            self.kernel_size_temp = (3, 1, 1)
            self.stride_temp = (1, 1, 1)
            self.padding_temp = (1, 0, 0)
        if no_scaling is True:
            self.scale_factor = (1, 1, 1)
            self.kernel_size_spatial = (1, 3, 3)
            self.stride_spatial = (1, 1, 1)
            self.padding_spatial = (0, 1, 1)
            self.kernel_size_temp = (3, 1, 1)
            self.stride_temp = (1, 1, 1)
            self.padding_temp = (1, 0, 0)
        if spatial is True:
            self.scale_factor = (1, 2, 2)
            self.kernel_size_spatial = (1, 3, 3)
            self.stride_spatial = (1, 1, 1)
            self.padding_spatial = (0, 1, 1)
            self.kernel_size_temp = (3, 1, 1)
            self.stride_temp = (1, 1, 1)
            self.padding_temp = (1, 0, 0)

        self.block = self.create_block()


class BlockDown3D(torch.nn.Module):
    def __init__(
            self, in_channel, out_channel, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), drop_prob=0):
        super(BlockDown3D, self).__init__()

        self.block = torch.nn.Sequential(
                torch.nn.Conv3d(in_channels=in_channel,
                                out_channels=out_channel,
                                stride=stride,
                                kernel_size=kernel_size, 
                                padding=padding,
                                padding_mode='reflect',
                                bias=True),
                torch.nn.InstanceNorm3d(out_channel),
                torch.nn.LeakyReLU(inplace=True), #torch.nn.Tanh(), #
                torch.nn.Conv3d(in_channels=out_channel,
                                out_channels=out_channel,
                                stride=stride,
                                kernel_size=kernel_size, 
                                padding=padding,
                                padding_mode='reflect',
                                bias=True),
                torch.nn.InstanceNorm3d(out_channel),
                torch.nn.LeakyReLU(inplace=True),
                torch.nn.Dropout3d(drop_prob),
                torch.nn.Conv3d(in_channels=out_channel,
                                out_channels=out_channel,
                                stride=stride,
                                kernel_size=kernel_size, 
                                padding=padding,
                                padding_mode='reflect',
                                bias=True),
                torch.nn.InstanceNorm3d(out_channel),
                torch.nn.LeakyReLU(inplace=True),
                torch.nn.Conv3d(in_channels=out_channel,
                                out_channels=out_channel,
                                stride=stride,
                                kernel_size=kernel_size, 
                                padding=padding,
                                padding_mode='reflect',
                                bias=True),
                torch.nn.InstanceNorm3d(out_channel),
            )
        self.Leaky = torch.nn.LeakyReLU()

        self.identity = torch.nn.Sequential(
            torch.nn.Conv3d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=(1, 1, 1),
                bias=False),
            nn.InstanceNorm3d(out_channel))

    def forward(self, x):
        block = self.block(x)
        id = self.identity(x)
        return self.Leaky(block + id)
    


class BlockUp3D(torch.nn.Module):
    """ Helper Class"""

    def __init__(self, in_channel, out_channel, kernel_size=(3, 3, 3), padding = (1, 1, 1), interpolate=False):
        super(BlockUp3D, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.padding = padding
        self.interpolate = interpolate

        self.block = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=self.in_channel,
                            out_channels=self.out_channel,
                            kernel_size=self.kernel_size,
                            padding=self.padding,
                            padding_mode='reflect',
                            bias=True),
            torch.nn.InstanceNorm3d(self.out_channel),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Conv3d(in_channels=self.out_channel,
                            out_channels=self.out_channel,
                            kernel_size=self.kernel_size,
                            padding=self.padding,
                            padding_mode='reflect',
                            bias=True),
            torch.nn.InstanceNorm3d(self.out_channel),
            torch.nn.LeakyReLU(inplace=True),
            )


    def forward(self, x):
        if self.interpolate is True:
            bs = x.shape[0]
            ch = x.shape[1]
            x = x.reshape(-1, x.shape[2], x.shape[3], x.shape[4])  
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            x = x.reshape(bs, ch, x.shape[1], x.shape[2], x.shape[3])  
        block = self.block(x)
        return block


class BlockUp3D_end(torch.nn.Module):
    def __init__(self, in_channel, out_channel1, out_channel2, bias=False, drop_prob=0):
        super(BlockUp3D_end, self).__init__()

        self.block = torch.nn.Sequential(
                torch.nn.Conv3d(in_channels=in_channel,
                                out_channels=out_channel1,
                                kernel_size=(1, 1, 1),
                                stride=(1, 1, 1),
                                padding=(0, 0, 0),
                                padding_mode='reflect',
                                bias=bias),
                torch.nn.InstanceNorm3d(out_channel1),
                torch.nn.LeakyReLU(inplace=True),
                torch.nn.Conv3d(in_channels=out_channel1,
                                out_channels=out_channel2,
                                kernel_size=(1, 1, 1),
                                stride=(1, 1, 1),
                                padding=(0, 0, 0),
                                padding_mode='reflect',
                                bias=bias))

    def forward(self, x):
        block = self.block(x)
        return block


# define the CNN architecture
class TwoDPlusTimeUnet(nn.Module):
    def __init__(self, Ninputchannels, Noutputchannels):
        super(TwoDPlusTimeUnet, self).__init__()

        # # Enconder
        self.block_down_1 = BlockDown2Dt(Ninputchannels, 32)
        self.maxpool1 = nn.MaxPool3d(2)
        self.block_down_2 = BlockDown2Dt(32, 64)
        self.maxpool2 = nn.MaxPool3d(2)
        self.block_down_3 = BlockDown2Dt(64, 128)
        self.maxpool3 = nn.MaxPool3d(
            kernel_size=(5, 1, 1),
            stride=(2, 1, 1),
            padding=(1, 0, 0))
        self.block_down_4 = BlockDown2Dt(128, 256)

        # Decoder
        self.block_up_1 = BlockUp2Dt(256, 128, bottleneck=True, bias=True)
        self.block_up_2 = BlockUp2Dt(256, 64,  bias=True)
        self.block_up_3 = BlockUp2Dt(128, 32,  bias=True)

        self.block_end = BlockUp3D_end(64, 32, 2,  bias=True)

    def forward(self, x):
        # Encoder
        c1 = self.block_down_1(x)  #32x60x256x256
        c2 = self.block_down_2(self.maxpool1(c1))  # 64x30x128x128
        c3 = self.block_down_3(self.maxpool2(c2))  # 128x15x64x64
        c4 = self.block_down_4(self.maxpool3(c3))  # 256x7x64x64

        # Decoder
        up1 = self.block_up_1(c4)  # 128x15x64x64
        cat1 = torch.cat((up1, c3), 1)  # 256x15x64x64
        up2 = self.block_up_2(cat1)  # 64x30x128x128
        cat2 = torch.cat((up2, c2), 1)  # 128x30x128x128
        up3 = self.block_up_3(cat2)  # 32x60x256x256
        cat3 = torch.cat((up3, c1), 1)  # 64x60x256x256

        # Output
        x = self.block_end(cat3)
        return x


# define the CNN architecture
class TwoDPlusTimeUnet_larger(TwoDPlusTimeUnet):
    def __init__(self, Ninputchannels, Noutputchannels):
        super().__init__(Ninputchannels, Noutputchannels)

        self.maxpool3 = nn.MaxPool3d(
            kernel_size=(3, 1, 1),
            stride=(1, 1, 1),
            padding=(0, 0, 0))
        self.block_down_4 = BlockDown2Dt(128, 512)

        # Decoder
        self.block_up_1 = BlockUp2Dt(512, 128, bottleneck=True,  bias=False)
        self.block_up_2 = BlockUp2Dt(256, 64, bias=False)
        self.block_up_3 = BlockUp2Dt(128, 32, bias=False)
        self.block_end = BlockUp3D_end(64, 32, 2,  bias=False)


# define the CNN architecture
class UNet3D(nn.Module):
    def __init__(self, Ninputchannels, Noutputchannels):
            super(UNet3D, self).__init__()

            # # Enconder
            self.block_down_1 = BlockDown3D(Ninputchannels, 32)
            self.block_down_2 = BlockDown3D(32, 64)
            self.maxpool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
            self.block_down_3 = BlockDown3D(64, 128)

            # Decoder             
            self.block_up_1 = BlockUp3D(128, 64, interpolate=True)
            self.block_up_2 = BlockUp3D(128, 32, interpolate=True)
            self.block_end = BlockUp3D_end(64, 32, 2,  bias=True)

    def forward(self, x):
        # x shape: bs, 16, 10, 256, 256
        # Encoder
        c1 = self.block_down_1(x)  # 32x10x256x256
        c2 = self.block_down_2(self.maxpool(c1))  # 64x10x128x128
        c3 = self.block_down_3(self.maxpool(c2))  # 128x10x64x64

        # Decoder
        up1 = self.block_up_1(c3)  # 64x10x128x128
        cat1 = torch.cat((up1,c2), 1)  #128x10x128x128
        up2 = self.block_up_2(cat1)  # 32x10x256x256
        cat2 = torch.cat((up2,c1), 1)  # 64x10x256x256

        # Output
        x = self.block_end(cat2)
        return x