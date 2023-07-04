import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


bias = False

class conv_block(nn.Module):
    # base block
    def __init__(self, ch_in, ch_out, affine=True, actv=nn.LeakyReLU(inplace=True), downsample=False, upsample=False):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.InstanceNorm2d(ch_out, affine=affine),
            actv,
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.InstanceNorm2d(ch_out, affine=affine),
            actv,
        )
        self.downsample = downsample
        self.upsample = upsample
        if self.upsample:
            self.up = up_conv(ch_out, ch_out // 2, affine)

    def forward(self, x):
        x1 = self.conv(x)
        c = x1.shape[1]
        if self.downsample:
            x2 = F.avg_pool2d(x1, 2)
            # half of channels for skip
            return x1[:,:c//2,:,:], x2
        # x1[:,:,:,:]
        if self.upsample:
            x2 = self.up(x1)
            return x2
        return x1


class up_conv(nn.Module):
    # base block
    def __init__(self, ch_in, ch_out, affine=True, actv=nn.LeakyReLU(inplace=True)):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.InstanceNorm2d(ch_out, affine=affine),
            actv,
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Encoder(nn.Module):
    # the Encoder_x or Encoder_r of G
    def __init__(self, in_c, mid_c, layers, affine):
        super(Encoder, self).__init__()
        encoder = []
        for i in range(layers):
            encoder.append(conv_block(in_c, mid_c, affine, downsample=True, upsample=False))
            in_c = mid_c
            mid_c = mid_c * 2
        self.encoder = nn.Sequential(*encoder)

    def forward(self, x):
        res = []
        for layer in self.encoder:
            x1, x2 = layer(x)
            res.append([x1, x2])
            x = x2
        return res

class ShareNet(nn.Module):
    # the Share Block of G
    def __init__(self, in_c, out_c, layers, affine,r):
        super(ShareNet, self).__init__()
        encoder = []
        decoder = []
        for i in range(layers-1):
            encoder.append(conv_block(in_c, in_c * 2, affine, downsample=True, upsample=False))
            decoder.append(conv_block(out_c-r, out_c//2, affine, downsample=False, upsample=True))
            in_c = in_c * 2
            out_c = out_c // 2
            r = r//2
        self.bottom = conv_block(in_c, in_c * 2, affine, upsample=True)
        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)
        self.layers = layers

    def forward(self, x):
        encoder_output = []
        x = x[-1][1]
        for layer in self.encoder:
            x1,x2 = layer(x)
            encoder_output.append([x1, x2])
            x = x2
        bottom_output = self.bottom(x)
        if self.layers == 1:
            return bottom_output
        encoder_output.reverse()
        for i, layer in enumerate(self.decoder):
            x = torch.cat([bottom_output, encoder_output[i][0]], dim=1)
            x = layer(x)
            bottom_output = x
        return x


class Decoder(nn.Module):
    # the Decoder_x or Decoder_r of G
    def __init__(self, in_c, mid_c, layers, affine, r):
        super(Decoder, self).__init__()
        decoder = []
        for i in range(layers-1):
            decoder.append(conv_block(in_c-r, mid_c, affine, downsample=False, upsample=True))
            in_c = mid_c
            mid_c = mid_c // 2
            r = r//2
        decoder.append(conv_block(in_c-r, mid_c, affine, downsample=False, upsample=False))
        self.decoder = nn.Sequential(*decoder)

    def forward(self, share_input, encoder_input):
        encoder_input.reverse()
        x = 0
        for i, layer in enumerate(self.decoder):
            x = torch.cat([share_input, encoder_input[i][0]], dim=1)
            x = layer(x)
            share_input = x
        return x
    
    
    
class Generator(nn.Module):
    # the G of TarGAN

    def __init__(self, in_c, mid_c, layers, s_layers, affine, last_ac=True):
        super(Generator, self).__init__()
        self.img_encoder = Encoder(in_c, mid_c, layers, affine)
        self.img_decoder = Decoder(mid_c * (2 ** layers), mid_c * (2 ** (layers - 1)), layers, affine,64)
        self.target_encoder = Encoder(in_c, mid_c, layers, affine)
        self.target_decoder = Decoder(mid_c * (2 ** layers), mid_c * (2 ** (layers - 1)), layers, affine,64)
        self.share_net = ShareNet(mid_c * (2 ** (layers - 1)), mid_c * (2 ** (layers - 1 + s_layers)), s_layers, affine,256)
        self.out_img = nn.Conv2d(mid_c, 1, 1, bias=bias)
        self.out_tumor = nn.Conv2d(mid_c, 1, 1, bias=bias)
        self.last_ac = last_ac

    def forward(self, img, tumor=None, c=None, mode="train"):
        # c = c.view(c.size(0), c.size(1), 1, 1)
        # c = c.repeat(1, 1, img.size(2), img.size(3))
        # img = torch.cat([img, c], dim=1)
        x_1 = self.img_encoder(img)
        s_1 = self.share_net(x_1)
        res_img = self.out_img(self.img_decoder(s_1,x_1))
        if self.last_ac:
            res_img = torch.tanh(res_img)
        if mode == "train":
            # tumor = torch.cat([tumor, c], dim=1)
            x_2 = self.target_encoder(tumor)
            s_2 = self.share_net(x_2)
            res_tumor = self.out_tumor(self.target_decoder(s_2, x_2))
            if self.last_ac:
                res_tumor = torch.tanh(res_tumor)
            return res_img, res_tumor
        return res_img
    
    
class Discriminator(nn.Module):
# the D_x or D_r of TarGAN ( backbone of PatchGAN )

    def __init__(self, image_size=256, c_dim=1, conv_dim=64, repeat_num=3):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(1, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        layers.append(nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False))

        self.main = nn.Sequential(*layers)


    def forward(self, x):
        h = self.main(x)
        return h