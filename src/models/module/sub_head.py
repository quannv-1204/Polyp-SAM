import torch
import torch.nn as nn
import torch.nn.functional as F
from .psa import PSA_p 

######################################################################################################################

class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size = 3, padding = 1, bias=False, dilation=1):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, padding=padding, groups=nin, bias=bias, dilation=dilation)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


# pre-activation based conv block
class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, kSize, stride=1, 
                    padding=0, dilation=1, bias=True, norm='BN', act='ReLU', conv='default', num_groups=1):
        super(Conv, self).__init__()
        if act == 'ELU':
            act = nn.ELU()
        else:
            act = nn.ReLU(True)
        module = []
        if norm == 'GN': 
            module.append(nn.GroupNorm(num_groups=num_groups, num_channels=in_ch))
        else:
            module.append(nn.BatchNorm2d(in_ch, eps=0.001, momentum=0.1, affine=True, track_running_stats=True))
        module.append(act)
        if conv=='default':
            module.append(nn.Conv2d(in_ch, out_ch, kernel_size=kSize, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=num_groups))
        else:
            module.append(depthwise_separable_conv(in_ch, out_ch, kernel_size=kSize, padding=padding, dilation=dilation, bias=bias))
        self.module = nn.Sequential(*module)
    def forward(self, x):
        out = self.module(x)
        return out

def padding(kernel_size, dilation):
    width_pad_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    width_pad_size = width_pad_size // 2 + (width_pad_size % 2 - 1)
    return width_pad_size

class Dilated_bottleNeck(nn.Module):
    def __init__(self, norm, act, in_feat):
        super(Dilated_bottleNeck, self).__init__()

        self.reduction1 = Conv(in_feat, in_feat//2, kSize=1, stride = 1, bias=False, padding=0)
        
        self.aspp_d3 = nn.Sequential(Conv(in_feat//2, in_feat//2, kSize=11, stride=1, padding=padding(11, 1), dilation=1,bias=False, norm=norm, act=act, num_groups=in_feat//2),
                                    Conv(in_feat//2, in_feat//4, kSize=1, stride=1, padding=0, dilation=1,bias=False, norm=norm, act=act))
        self.aspp_d6 = nn.Sequential(Conv(in_feat//2 + in_feat//4, in_feat//2 + in_feat//4, kSize=11, stride=1, padding=padding(11, 2), dilation=2,bias=False, norm=norm, act=act, num_groups=in_feat//2 + in_feat//4),
                                    Conv(in_feat//2 + in_feat//4, in_feat//4, kSize=1, stride=1, padding=0, dilation=1,bias=False, norm=norm, act=act))
        self.aspp_d9 = nn.Sequential(Conv(in_feat, in_feat, kSize=11, stride=1, padding=padding(11, 4), dilation=4,bias=False, norm=norm, act=act, num_groups=in_feat),
                                    Conv(in_feat, in_feat//4, kSize=1, stride=1, padding=0, dilation=1,bias=False, norm=norm, act=act))
        
        self.reduction2 = Conv(((in_feat//4)*3) + (in_feat//2), in_feat//2, kSize=1, stride=1, padding=0,bias=False, norm=norm, act=act)
    def forward(self, x):
        x = self.reduction1(x)
        d3 = self.aspp_d3(x)
        cat1 = torch.cat([x, d3],dim=1)
        d6 = self.aspp_d6(cat1)
        cat2 = torch.cat([cat1, d6],dim=1)
        d9 = self.aspp_d9(cat2)
        out = self.reduction2(torch.cat([x,d3,d6,d9], dim=1))
        return out      # num_channels x H/16 x W/16
    


class SubHead(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        norm = "BN"
        act = 'ReLU'
        kSize = 3
        num_channels = 256
        ###############################################################################################
        self.ASPP = Dilated_bottleNeck(norm, act, num_channels)
        self.psa_1 = PSA_p(num_channels//2, num_channels//2) 
        self.decoder1 = nn.Sequential(Conv(num_channels//2, num_channels//4, kSize, stride=1, padding=kSize//2, bias=False, 
                                            norm=norm, act=act),      
                                        Conv(num_channels//4, num_channels//8, kSize, stride=1, padding=kSize//2, bias=False, 
                                            norm=norm, act=act),    
                                        Conv(num_channels//8, num_channels//16, kSize, stride=1, padding=kSize//2, bias=False, 
                                            norm=norm, act=act),  
                                        Conv(num_channels//16, num_channels//32, kSize, stride=1, padding=kSize//2, bias=False, 
                                            norm=norm, act=act),
                                        Conv(num_channels//32, 1, kSize, stride=1, padding=kSize//2, bias=False, 
                                            norm=norm, act=act)
                                     )

    def forward(self, dense_feat):

        # decoder 1 - Pyramid level 5
        dense_feat = self.ASPP(dense_feat)   
        dense_feat = self.psa_1(dense_feat)                
        mask = self.decoder1(dense_feat)

        return mask
