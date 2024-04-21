#使用全局共享参数的多尺度融合方式
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from backbone.cswin_transformer import CSWin_tiny, CSwin_small, CSwin_base
from torch.nn import Parameter
import math
import time

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU()
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU()
        )

class DownConnection(nn.Module):
    def __init__(self, inplanes, planes, stride=2):
        super(DownConnection, self).__init__()
        self.convbn1 = ConvBN(inplanes, planes, kernel_size=3, stride=1)
        self.convbn2 = ConvBN(planes, planes, kernel_size=3, stride=stride)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = ConvBN(inplanes, planes, stride=stride)

    def forward(self, x):
        residual = x
        x = self.convbn1(x)
        x = self.relu(x)
        x = self.convbn2(x)
        x = x + self.downsample(residual)
        x = self.relu(x)

        return x

class SCMF(nn.Module):
    def __init__(self, encoder_channels=(96, 192, 384, 768), atrous_rates=(6, 12)):
        super(SCMF, self).__init__()
        self.conv4 = Conv(encoder_channels[3], encoder_channels[3], kernel_size=1)
        self.conv3 = Conv(encoder_channels[2], encoder_channels[2], kernel_size=1)
        self.conv2 = Conv(encoder_channels[1], encoder_channels[1], kernel_size=1)
        self.conv1 = Conv(encoder_channels[0], encoder_channels[0], kernel_size=1)
        
        self.down11 = DownConnection(encoder_channels[0], encoder_channels[1])
        self.down12 = DownConnection(encoder_channels[1], encoder_channels[2])
        self.down13 = DownConnection(encoder_channels[2], encoder_channels[3])
        self.down21 = DownConnection(encoder_channels[1], encoder_channels[2])
        self.down22 = DownConnection(encoder_channels[2], encoder_channels[3])
        self.down31 = DownConnection(encoder_channels[2], encoder_channels[3])
        
        rate_1, rate_2 = tuple(atrous_rates)
        
        self.lf4 = nn.Sequential(SeparableConvBNReLU(encoder_channels[3], encoder_channels[2], dilation=rate_1), nn.UpsamplingNearest2d(scale_factor=2))
        self.lf3 = nn.Sequential(SeparableConvBNReLU(encoder_channels[2], encoder_channels[1], dilation=rate_2), nn.UpsamplingNearest2d(scale_factor=2))
        self.lf2 = nn.Sequential(SeparableConvBNReLU(encoder_channels[1], encoder_channels[0], dilation=rate_2), nn.UpsamplingNearest2d(scale_factor=2))

    def forward(self, x1, x2, x3, x4):
        out4 = self.conv4(x4) + self.down31(x3) + self.down22(self.down21(x2)) + self.down13(self.down12(self.down11(x1)))
        out3 = self.conv3(x3) + self.down21(x2) + self.down12(self.down11(x1)) + self.lf4(out4)
        out2 = self.conv2(x2) + self.down11(x1) + self.lf3(out3)
        out1 =self.conv1(x1) + self.lf2(out2)

        return out1, out2, out3, out4

class Down_Attention(nn.Module):
    def __init__(self, mode, channels, ratio):
        super().__init__()
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_pooling = nn.AdaptiveMaxPool2d(1)
        if mode == "max":
            self.global_pooling = self.max_pooling
        elif mode == "avg":
            self.global_pooling = self.avg_pooling
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features = channels, out_features = channels // ratio, bias = False),
            nn.ReLU(),
            nn.Linear(in_features = channels // ratio, out_features = channels, bias = False),
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        b, c, _, _ = x.shape
        v = self.global_pooling(x).view(b, c)
        v = self.fc_layers(v).view(b, c, 1, 1)
        v = self.sigmoid(v)
        x = x * v
        return x

class Up_Attention(nn.Module):
    def __init__(self, channels, ratio):
        super().__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features = channels, out_features = channels // ratio, bias = False),
            nn.ReLU(),
            nn.Linear(in_features = channels // ratio, out_features = channels, bias = False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        y = x.view(b, c, h*w)
        index = torch.where(y[..., :] > 0, True, False)
        weight_1 = index.sum(dim = 2) / (h*w)
        weight_2_1 = c*0.0001 + weight_1.sum(dim=1)
        weight_2_1 = weight_2_1.unsqueeze(dim=-1)
        weight_2_2 = 1 / (weight_1 + 0.0001)
        weight_2 = torch.log(weight_2_1 * weight_2_2)
        weight_2 = self.fc_layers(weight_2).view(b, c, 1, 1)
        weight_2 = self.sigmoid(weight_2)
        x = weight_2 * x
        
        return x
              
class Attention(nn.Module):
    def __init__(self, down_mode, channels, ratio):
        super().__init__()
        self.conv1 = SeparableConvBNReLU(in_channels=channels, out_channels=channels)
        self.down = Down_Attention(mode=down_mode, channels=channels, ratio=ratio)
        self.up = Up_Attention(channels, ratio)
        self.conv2 = SeparableConvBNReLU(in_channels=channels, out_channels=channels)

    def forward(self, x):
        x = self.conv1(x)
        x1 = self.down(x)
        x2 = self.up(x)
        out = self.conv2(x1 + x2)
        return out


def l2_norm(x):
    return torch.einsum("bcn, bn->bcn", x, 1 / torch.norm(x, p=2, dim=-2))


class BA_Model(nn.Module):
    def __init__(self, high_level_channel, channel, scale_factor=2, att_scale=8, eps=1e-6):
        super(BA_Model, self).__init__()
    
        
        self.conv_high = Conv(high_level_channel, channel, 1)
        self.up = nn.Sequential(
            ConvBNReLU(channel, channel),
            nn.UpsamplingNearest2d(scale_factor=scale_factor)
        )
        
        self.key_conv = Conv(in_channels=channel, out_channels=channel // att_scale, kernel_size=1)
        self.query_conv = Conv(in_channels=channel, out_channels=channel // att_scale, kernel_size=1)
        self.value_conv = Conv(in_channels=channel, out_channels=channel, kernel_size=1)
    
        self.gamma = Parameter(torch.zeros(1))
        self.l2_norm = l2_norm
        self.eps = eps

    
    def forward(self, low, high):

        
        high = self.conv_high(high)
        high = self.up(high)
        
        batch_size, chnnels, width, height = low.shape
            
        Q = self.query_conv(high).view(batch_size, -1, width * height)
        K = self.key_conv(low).view(batch_size, -1, width * height)
        V = self.value_conv(low).view(batch_size, -1, width * height)
        
        Q = self.l2_norm(Q).permute(0, 2, 1)
        K = self.l2_norm(K)

        tailor_sum = 1 / (width * height + torch.einsum("bnc, bc->bn", Q, torch.sum(K, dim=-1) + self.eps))
        value_sum = torch.einsum("bcn->bc", V).unsqueeze(-1)
        value_sum = value_sum.expand(-1, chnnels, width * height)

        matrix = torch.einsum('bmn, bcn->bmc', K, V)
        matrix_sum = value_sum + torch.einsum("bnm, bmc->bcn", Q, matrix)

        weight_value = torch.einsum("bcn, bn->bcn", matrix_sum, tailor_sum)
        weight_value = weight_value.view(batch_size, chnnels, height, width)
        
        return (low + self.gamma * weight_value).contiguous()
    
    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class BA_Decoder(nn.Module):
    def __init__(self,
                 encoder_channels=(96, 192, 384, 768),
                 dropout=0.05):
        super(BA_Decoder, self).__init__()
        self.ba_1 = BA_Model(encoder_channels[1], encoder_channels[0], 2)
        self.ba_2 = BA_Model(encoder_channels[2], encoder_channels[0], 4)
        self.ba_3 = BA_Model(encoder_channels[3], encoder_channels[0], 8)

        self.dropout = nn.Dropout2d(p=dropout, inplace=False)
        self.detection_head = nn.Sequential(
            ConvBNReLU(encoder_channels[0], encoder_channels[0]),
            nn.UpsamplingBilinear2d(scale_factor=4),
            Conv(encoder_channels[0], 1, kernel_size=1))
        
    def forward(self, x1, x2, x3, x4):
        out1 = self.ba_1(x1, x2)
        out2 = self.ba_2(out1, x3)
        out3 = self.ba_3(out2, x4)

        out = self.dropout(out3)
        bd_out = self.detection_head(out)
        return out, bd_out
        

class Decoder(nn.Module):
    def __init__(self,
                 encoder_channels=(96, 192, 384, 768),
                 dropout=0.05,
                 atrous_rates=(6, 12),
                 num_classes=6):
        super(Decoder, self).__init__()
        self.scmf = SCMF(encoder_channels, atrous_rates)
        
        self.se4 = Attention(down_mode='avg', channels=encoder_channels[3], ratio=2)
        self.se3 = Attention(down_mode='avg', channels=encoder_channels[2], ratio=2)
        self.se2 = Attention(down_mode='avg', channels=encoder_channels[1], ratio=2)
        self.se1 = Attention(down_mode='avg', channels=encoder_channels[0], ratio=2)
        
        self.up4 = nn.Sequential(
            ConvBNReLU(encoder_channels[3], encoder_channels[2]),
            nn.UpsamplingNearest2d(scale_factor=2)
        )
        self.up3 = nn.Sequential(
            ConvBNReLU(encoder_channels[2], encoder_channels[1]),
            nn.UpsamplingNearest2d(scale_factor=2)
        )
        self.up2 = nn.Sequential(
            ConvBNReLU(encoder_channels[1], encoder_channels[0]),
            nn.UpsamplingNearest2d(scale_factor=2)
        )
        
        self.dropout = nn.Dropout2d(p=dropout, inplace=False)
     
        self.segmentation_head = nn.Sequential(
            ConvBNReLU(encoder_channels[0] + encoder_channels[0], encoder_channels[0]),
            # Conv(encoder_channels[0], num_classes, kernel_size=1),
            nn.UpsamplingBilinear2d(scale_factor=4),
            Conv(encoder_channels[0], num_classes, kernel_size=1))

        self.init_weight()


    def forward(self, x1, x2, x3, x4, bd_out):
        out1, out2, out3, out4 = self.scmf(x1, x2, x3, x4)

        out4 = self.se4(out4) + out4
        out3 = out3 + self.up4(out4)
        del out4
        
        out3 = self.se3(out3) + out3
        out2 = out2 + self.up3(out3)
        del out3
        
        out2 = self.se2(out2) + out2
        out1 = out1 + self.up2(out2)
        del out2
        
        out = self.se1(out1) + out1

        out = torch.cat([out, bd_out], dim=1)
        x = self.dropout(out)
        out = self.segmentation_head(x)
        
        return out
    
    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
class MCSNet(nn.Module):
    def __init__(self,
                 backbone='cswin_tiny',
                 dropout=0.05,
                 atrous_rates=(6, 12),
                 num_classes=6
                ):
        super(MCSNet, self).__init__()
        if backbone == 'cswin_tiny': 
            self.backbone = CSWin_tiny()
            self.encoder_channels = (64, 128, 256, 512)
        elif backbone == 'cswin_small':
            self.backbone = CSwin_small()
            self.encoder_channels = (64, 128, 256, 512)
        elif backbone == 'cswin_base':
            self.backbone = CSwin_base()
            self.encoder_channels = (96, 192, 384, 768)
        self.ba_decoder = BA_Decoder(self.encoder_channels, dropout)
        self.decoder = Decoder(self.encoder_channels, dropout, atrous_rates, num_classes)
    
    def forward(self, x):
        x1, x2, x3, x4 = self.backbone(x)
        out, bd_out = self.ba_decoder(x1, x2, x3, x4)
        seg_out = self.decoder(x1, x2, x3, x4, out)
        return bd_out, seg_out

def mcsnet_tiny(pretrained=True, num_classes=4, weight_path='/15T-1/wyj/airs/pretrain_weights/upernet_cswin_tiny.pth'):
    # pretrained weights are load from official repo of Swin Transformer
    model = MCSNet(backbone='cswin_tiny', 
                   num_classes=num_classes)
    if pretrained and weight_path is not None:
        old_dict = torch.load(weight_path)['state_dict']
        model_dict = model.state_dict()
        old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
        model_dict.update(old_dict)
        model.load_state_dict(model_dict)
    return model


def mcsnet_small(pretrained=True, num_classes=4, weight_path='/15T-1/wyj/airs/pretrain_weights/upernet_cswin_small.pth'):
    model = MCSNet(backbone='cswin_small',
                   num_classes=num_classes)
    if pretrained and weight_path is not None:
        old_dict = torch.load(weight_path)['state_dict']
        model_dict = model.state_dict()
        old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
        model_dict.update(old_dict)
        model.load_state_dict(model_dict)
    return model


def mcsnet_base(pretrained=True, num_classes=4, weight_path='pretrain_weights/upernet_cswin_base.pth'):
    model = MCSNet(backbone='cswin_base',
                   num_classes=num_classes)
    if pretrained and weight_path is not None:
        old_dict = torch.load(weight_path)['state_dict']
        model_dict = model.state_dict()
        old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
        model_dict.update(old_dict)
        model.load_state_dict(model_dict)
    return model