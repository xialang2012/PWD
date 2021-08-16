import torch
import torch.nn as nn
import torch.nn.functional as F

channels = [16, 64, 128, 256]

def conv1x1(in_channels, out_channels, stride=1):
    """1x1 convolution"""
    return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.99),
            nn.ReLU()
        )
    
class FullConfusionUnit(nn.Module):

    def __init__(self, in_channels, out_channels, with_conv_shortcut=False, halfChannels=True):
        super(FullConfusionUnit, self).__init__()
        self.halfChannels = halfChannels
        self.with_conv_shortcut = with_conv_shortcut

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.99),
            nn.ReLU()
        )

        self.conv_d = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.99),
            nn.ReLU()
        )

    def forward(self, x):

        out = self.conv(x)
        out = self.conv(out)

        if self.with_conv_shortcut:
            residual = self.conv(x)
            out = out.add(residual)
        else:
            out = out.add(x)

        if self.halfChannels:
            #out_half = F.interpolate(out, scale_factor=1.0/2, mode='bilinear')
            out_half = self.conv_d(out)
        else:
            out_half = None

        return out, out_half

class BasicBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, with_conv_shortcut=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.99),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.99)            
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.99)            
        )

        self.relu = nn.ReLU()

        self.with_conv_shortcut = with_conv_shortcut

    def forward(self, x):

        out = self.conv1(x)

        out = self.conv2(out)

        if self.with_conv_shortcut:
            residual = self.conv3(x)
            out += residual
        else:
            out += x

        out = self.relu(out)
        return out

class FRNet(nn.Module):   

    def __init__(self, in_channels, classes=1) -> None:
        super().__init__()

        self.inHeadChannels = channels[1]

        self.inconv = nn.Sequential(
            nn.Conv2d(in_channels, self.inHeadChannels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.inHeadChannels, eps=0.001, momentum=0.99),
            nn.ReLU()
        )

        # 1*1 convs for stage 1 and 2
        self.conv1_s1 = conv1x1(self.inHeadChannels, channels[0])
        self.conv1_s2 = conv1x1(self.inHeadChannels, channels[1])

        # 1*1 convs for stage 3
        self.conv1_s3_fs2_l2_f = conv1x1(channels[1], channels[0])
        self.conv1_s3_fs3_l2_f = conv1x1(channels[0], channels[1])
        self.conv1_s3_fs3_l3_f = conv1x1(channels[1], channels[2])
        self.conv1_s3_fs2_l2_f_d = conv1x1(channels[1], channels[2])
        self.conv1_s3_fs1_4f = conv1x1(self.inHeadChannels, channels[2])

        # 1*1 convs for stage 4
        self.conv1_s4_fs3_l2_f = conv1x1(channels[1], channels[0])
        self.conv1_s4_fs3_l3_f = conv1x1(channels[2], channels[0])
        self.conv1_s4_fs4_l2_f = conv1x1(channels[0], channels[1])
        self.conv1_s4_fs3_l3_f_1 = conv1x1(channels[2], channels[1])

        self.conv1_s4_fs4_l3_f = conv1x1(channels[1], channels[2])
        self.conv1_s4_fs4_l4_f = conv1x1(channels[2], channels[3])
        self.conv1_s4_fs3_l3_f_2 = conv1x1(channels[2], channels[3])
        self.conv1_s4_fs2_l2_f = conv1x1(channels[1], channels[3])
        self.conv1_s4_fs1_8f = conv1x1(self.inHeadChannels, channels[3])
    
        # basic block for stage 1 to 4
        self.BasicBlock_s1 = BasicBlock(channels[0], channels[0])
        self.BasicBlock_s2 = BasicBlock(channels[1], channels[1])
        self.BasicBlock_s2_short = BasicBlock(channels[1], channels[1], True)
        self.BasicBlock_s3 = BasicBlock(channels[2], channels[2])
        self.BasicBlock_s4 = BasicBlock(channels[3], channels[3])

        # confusion unit for stage 2, 3, 4
        # s3_l1_f, channels[0]
        self.funit_s3_l1_f = FullConfusionUnit(channels[0], channels[0])  
        # s3_l2_f, channels[1]
        self.funit_s3_l2_f = FullConfusionUnit(channels[1], channels[1])
        # s4_l1_f, channels[0]
        self.funit_s4_l1_f = FullConfusionUnit(channels[0], channels[0])
        # s4_l2_f, channels[1]
        self.funit_s4_l2_f = FullConfusionUnit(channels[1], channels[1])
        # s4_l3_f, channels[2]
        self.funit_s4_l3_f = FullConfusionUnit(channels[2], channels[2])
        # s4_l4_f, channels[3]
        self.funit_s4_l4_f = FullConfusionUnit(channels[3], channels[3], halfChannels=None)

        self.maxpool_2 = nn.MaxPool2d(2, 2)
        self.maxpool_4 = nn.MaxPool2d(4, 4)
        self.maxpool_8 = nn.MaxPool2d(8, 8)
        # final class
        self.conFinal = conv1x1(sum(channels), 1)
    
    def forward(self, x):

        s1_f = self.inconv(x)
        s1_2f = self.maxpool_2(s1_f)
        s1_4f = self.maxpool_4(s1_f)
        s1_8f = self.maxpool_8(s1_f)

        # stage2, full & aspp layer 1
        s1_f = self.conv1_s1(s1_f)
        s2_l1_f = self.BasicBlock_s1(s1_f)
        s2_l1_f = self.BasicBlock_s1(s2_l1_f) # L1
        #s2_l1_f = self.BasicBlock_s1(s2_l1_f) # L1

        s2_l2_f = self.conv1_s2(s1_2f)
        #s2_l2_f = self.BasicBlock_s2_short(s1_2f) # L2
        s2_l2_f = self.BasicBlock_s2(s2_l2_f) # L2
        #s2_l2_f = self.BasicBlock_s2(s2_l2_f) # L2

        # stage 3
        s3_l1_f = self.conv1_s3_fs2_l2_f(s2_l2_f)
        s3_l1_f = F.interpolate(s3_l1_f, scale_factor=2, mode='bilinear')
        s3_l1_f += s2_l1_f
        s3_l1_f, s3_l2_f = self.funit_s3_l1_f(s3_l1_f)
        s3_l1_f = self.BasicBlock_s1(s3_l1_f)      # L1
        #s3_l1_f = self.BasicBlock_s1(s3_l1_f)      # L1

        s3_l2_f = self.conv1_s3_fs3_l2_f(s3_l2_f)
        s3_l2_f += s2_l2_f
        s3_l2_f, s3_l3_f = self.funit_s3_l2_f(s3_l2_f)
        s3_l2_f = self.BasicBlock_s2(s3_l2_f)      # L2
        #s3_l2_f = self.BasicBlock_s2(s3_l2_f)      # L2

        s3_l3_f = self.conv1_s3_fs3_l3_f(s3_l3_f)
        # s2_l2_f_d = F.interpolate(s2_l2_f, scale_factor=0.5, mode='bilinear')
        s2_l2_f_d = self.maxpool_2(s2_l2_f)
        s2_l2_f_d = self.conv1_s3_fs2_l2_f_d(s2_l2_f_d)
        s1_4f_c = self.conv1_s3_fs1_4f(s1_4f)
        s3_l3_f = s3_l3_f.add(s2_l2_f_d).add(s1_4f_c)
        s3_l3_f = self.BasicBlock_s3(s3_l3_f)      # L3
        #s3_l3_f = self.BasicBlock_s3(s3_l3_f)      # L3

        # stage 4
        s4_l1_f_d = self.conv1_s4_fs3_l2_f(s3_l2_f)
        s4_l1_f_d = F.interpolate(s4_l1_f_d, scale_factor=2, mode='bilinear')

        s4_l1_f_2up = self.conv1_s4_fs3_l3_f(s3_l3_f)
        s4_l1_f_2up = F.interpolate(s4_l1_f_2up, scale_factor=4, mode='bilinear')

        s4_l1_f = s3_l1_f.add(s4_l1_f_d).add(s4_l1_f_2up)
        s4_l1_f, s4_l2_f = self.funit_s4_l1_f(s4_l1_f)
        s4_l1_f = self.BasicBlock_s1(s4_l1_f)      # L1
        #s4_l1_f = self.BasicBlock_s1(s4_l1_f)      # L1

        #s4_l2_f_d = MaxPooling2D(pool_size=(2, 2))(s4_l2_f)    # downsample
        s4_l2_f_d = self.conv1_s4_fs4_l2_f(s4_l2_f)   # channels
        s4_l2_f_up = self.conv1_s4_fs3_l3_f_1(s3_l3_f)
        s4_l2_f_up = F.interpolate(s4_l2_f_up, scale_factor=2, mode='bilinear')
        s4_l2_f = s4_l2_f_d.add(s4_l2_f_up).add(s3_l2_f)
        s4_l2_f, s4_l3_f = self.funit_s4_l2_f(s4_l2_f)
        s4_l2_f = self.BasicBlock_s2(s4_l2_f)      # L2
        #s4_l2_f = self.BasicBlock_s2(s4_l2_f)      # L2


        #s4_l3_f_d = MaxPooling2D(pool_size=(2, 2))(s4_l3_f)    # downsample
        s4_l3_f_d = self.conv1_s4_fs4_l3_f(s4_l3_f)   # channels  
        s4_l3_f = s4_l3_f_d.add(s3_l3_f)
        s4_l3_f, s4_l4_f = self.funit_s4_l3_f(s4_l3_f)
        s4_l3_f = self.BasicBlock_s3(s4_l3_f)      # L3
        #s4_l3_f = self.BasicBlock_s3(s4_l3_f)      # L3


        #s4_l4_f_d = MaxPooling2D(pool_size=(2, 2))(s4_l4_f)    
        s4_l4_f_d = self.conv1_s4_fs4_l4_f(s4_l4_f)   # channels
        
        s3_l3_f_d = self.conv1_s4_fs3_l3_f_2 (s3_l3_f)   # channels
        #s3_l3_f_d = F.interpolate(s3_l3_f_d, scale_factor=0.5, mode='bilinear')
        s3_l3_f_d = self.maxpool_2(s3_l3_f_d)

        s2_l2_f_2d = self.conv1_s4_fs2_l2_f(s2_l2_f)   # channels
        #s2_l2_f_2d = F.interpolate(s2_l2_f_2d, scale_factor=0.25, mode='bilinear')
        s2_l2_f_2d = self.maxpool_4(s2_l2_f_2d)

        s1_8f = self.conv1_s4_fs1_8f(s1_8f)

        s4_l4_f = s2_l2_f_2d.add(s4_l4_f_d).add(s1_8f).add(s3_l3_f_d)
        s4_l4_f, s4_l5_f = self.funit_s4_l4_f(s4_l4_f)
        s4_l4_f = self.BasicBlock_s4(s4_l4_f)      # L4
        #s4_l4_f = self.BasicBlock_s4(s4_l4_f)      # L4

        # upsampling
        s4_l2_f = F.interpolate(s4_l2_f, scale_factor=2, mode='bilinear')
        s4_l3_f = F.interpolate(s4_l3_f, scale_factor=4, mode='bilinear')
        s4_l4_f = F.interpolate(s4_l4_f, scale_factor=8, mode='bilinear')

        x = torch.cat([s4_l1_f, s4_l2_f, s4_l3_f, s4_l4_f], 1)

        return [self.conFinal(x), 1]

if __name__ == '__main__':
    num_classes = 1
    in_batch, inchannel, in_h, in_w = 1, 3, 256, 256

    # x = np.arange(0, 4*3*256*256)
    # x = x.reshape(3, 256, 256, 4)


    x = torch.randn(in_batch, inchannel, in_h, in_w)
    net = FRNet(inchannel, num_classes)
    print(sum(p.numel() for p in net.parameters()), sum(p.numel() for p in net.parameters() if p.requires_grad))

    out = net(x)[0]

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter('runs/fashion_mnist_experiment_1')

    from torchviz import make_dot
    make_dot(out, params=dict(list(net.named_parameters()))).render("rnn_torchviz", format="png")

    with SummaryWriter(comment='LeNet') as w:
        w.add_graph(net, x)
    print(out.shape)