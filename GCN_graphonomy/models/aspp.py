import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP(nn.Module):
    def __init__(self, backbone, output_stride, BatchNorm, dropout=0.5):
        super(ASPP, self).__init__()
        if backbone == 'drn':
            inplanes = 512
        elif backbone == 'mobilenet':
            inplanes = 320
        else:
            inplanes = 2048
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_avg_pool_conv = nn.Conv2d(inplanes, 256, 1, stride=1, bias=False)
        self.global_avg_pool_bn = BatchNorm(256)
        self.global_avg_pool_act = nn.ReLU()    

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = BatchNorm(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)

        # x.shape=torch.Size([4, 2048, 16, 12]), 
        # x1.shape=torch.Size([4, 256, 16, 12]), 
        # x2.shape=torch.Size([4, 256, 16, 12]), 
        # x3.shape=torch.Size([4, 256, 16, 12]), 
        # x4.shape=torch.Size([4, 256, 16, 12]), 
        #print( "[ASPP] x.shape={}, x1.shape={}, x2.shape={}, x3.shape={}, x4.shape={}".format( x.shape,x1.shape,x2.shape,x3.shape,x4.shape) )

        # x5.shape=torch.Size([4, 256, 1, 1]) -> torch.Size([4, 256, 16, 12])
        x5 = self.global_avg_pool(x)
        #print( "[ASPP / GAP] x5.shape={}".format( x5.shape) )
        x5 = self.global_avg_pool_conv(x5)
        #print( "[ASPP / GAP conv] x5.shape={}".format( x5.shape) )
        if( x5.shape[0] > 1):
            x5 = self.global_avg_pool_bn(x5)
            #print( "[ASPP / GAP bn] x5.shape={}".format( x5.shape) )
        x5 = self.global_avg_pool_act(x5)
        #print( "[ASPP / GAP act] x5.shape={}".format( x5.shape) )
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        #print( "[ASPP / GAP up] x5.shape={}".format( x5.shape) )

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        #print( "[ASPP] x.shape={}".format( x.shape) )
        #print( "[ASPP] x[0,0,:,:]", x[0,0,:,:] )

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def build_aspp(backbone, output_stride, BatchNorm):
    return ASPP(backbone, output_stride, BatchNorm)