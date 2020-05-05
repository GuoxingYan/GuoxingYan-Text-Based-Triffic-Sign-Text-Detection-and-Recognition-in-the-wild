import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import math
import torchvision.models
from torchsummary import summary
import hiddenlayer as hl
# from tensorboardX import SummaryWriter
from torchviz import make_dot
from torch.autograd import Variable
from builder import ConvBuilder
#from modules.modulated_deform_conv import ModulatedDeformConv, _ModulatedDeformConv, ModulatedDeformConvPack
from mmcv.cnn import constant_init
from mmdet.ops import DeformConv, ModulatedDeformConv

#from torchstat import stat





class ACBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False):
        super(ACBlock, self).__init__()
        self.deploy = deploy
        if deploy:
            self.fused_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size,kernel_size), stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)
        else:
            self.square_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=(kernel_size, kernel_size), stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=False,
                                         padding_mode=padding_mode)
            self.square_bn = nn.BatchNorm2d(num_features=out_channels)

            self.relu = nn.ReLU(inplace=True)
            self.ver_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 1),#(5, 1),
                                      stride=stride,
                                      padding=(1, 0), dilation=dilation, groups=groups, bias=False,#padding=(2, 0),
                                      padding_mode=padding_mode)

            self.hor_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 3),#(1, 5),
                                      stride=stride,
                                      padding=(0, 1), dilation=dilation, groups=groups, bias=False,#padding=(0, 2),
                                      padding_mode=padding_mode)
            self.ver_bn = nn.BatchNorm2d(num_features=out_channels)
            self.hor_bn = nn.BatchNorm2d(num_features=out_channels)
            
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)


    def forward(self, input):
        if self.deploy:
            return self.fused_conv(input)
        else:
            square_outputs = self.square_conv(input)
            square_outputs = self.relu(self.square_bn(square_outputs))
            # print(square_outputs.size())
            # return square_outputs
            # vertical_outputs = self.ver_conv_crop_layer(input)
            # vertical_outputs = self.ver_conv(vertical_outputs)
            vertical_outputs = self.ver_conv(input)
            vertical_outputs = self.relu(self.ver_bn(vertical_outputs))
            # print(vertical_outputs.size())
            # horizontal_outputs = self.hor_conv_crop_layer(input)
            # horizontal_outputs = self.hor_conv(horizontal_outputs)
            horizontal_outputs = self.hor_conv(input)
            horizontal_outputs = self.relu(self.hor_bn(horizontal_outputs))
            # print(horizontal_outputs.size())
            return square_outputs + vertical_outputs + horizontal_outputs


class ACNetBuilder(ConvBuilder):

    def __init__(self, deploy):
        super(ACNetBuilder, self).__init__()
        self.deploy = deploy

    def switch_to_deploy(self):
        self.deploy = True


    def Conv2d(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', use_original_conv=False):
        if use_original_conv or kernel_size == 1 or kernel_size == (1, 1):
            return super(ACNetBuilder, self).Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                 padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode, use_original_conv=True)
        else:
            return ACBlock(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                       padding=padding, dilation=dilation, groups=groups, padding_mode=padding_mode, deploy=self.deploy)


    def Conv2dBN(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', use_original_conv=False):
        if use_original_conv or kernel_size == 1 or kernel_size == (1, 1):
            return super(ACNetBuilder, self).Conv2dBN(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                 padding=padding, dilation=dilation, groups=groups, padding_mode=padding_mode, use_original_conv=True)
        else:
            return ACBlock(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                       padding=padding, dilation=dilation, groups=groups, padding_mode=padding_mode, deploy=self.deploy)


    def Conv2dBNReLU(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', use_original_conv=False):
        if use_original_conv or kernel_size == 1 or kernel_size == (1, 1):
            return super(ACNetBuilder, self).Conv2dBNReLU(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                 padding=padding, dilation=dilation, groups=groups, padding_mode=padding_mode, use_original_conv=True)
        else:
            se = nn.Sequential()
            se.add_module('acb', ACBlock(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                       padding=padding, dilation=dilation, groups=groups, padding_mode=padding_mode, deploy=self.deploy))
            se.add_module('relu', self.ReLU())
            return se


    def BNReLUConv2d(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', use_original_conv=False):
        if use_original_conv or kernel_size == 1 or kernel_size == (1, 1):
            return super(ACNetBuilder, self).BNReLUConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                 padding=padding, dilation=dilation, groups=groups, padding_mode=padding_mode, use_original_conv=True)
        bn_layer = self.BatchNorm2d(num_features=in_channels)
        conv_layer = ACBlock(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                       padding=padding, dilation=dilation, groups=groups, padding_mode=padding_mode, deploy=self.deploy)
        se = self.Sequential()
        se.add_module('bn', bn_layer)
        se.add_module('relu', self.ReLU())
        se.add_module('acb', conv_layer)
        return se


cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
# cfg = [64, 'M', 128, 'M', 256, 256,  'M', 512, 512, 'M']


def make_layers(cfg, batch_norm=False):
	layers = []
	in_channels = 3
	for v in cfg:
		if v == 'M':
			layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
		else:
			conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
			if batch_norm:
				layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
			else:
				layers += [conv2d, nn.ReLU(inplace=True)]
			in_channels = v
	return nn.Sequential(*layers)

# Asymmetric Convolution Blocks#################################################3
ACNetBuilder = ACNetBuilder(deploy=False)

def make_layers_acb(cfg, batch_norm=False):
	layers = []
	in_channels = 3
	for v in cfg:
		if v == 'M':
			layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
		else:
			conv2d_acb = ACNetBuilder.Conv2dBNReLU(in_channels, v, kernel_size=3, padding=1)
			layers += conv2d_acb
	
			in_channels = v
	return nn.Sequential(*layers)


class VGG(nn.Module):
	def __init__(self, features):
		super(VGG, self).__init__()
		self.features = features
		self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
		self.classifier = nn.Sequential(
			nn.Linear(512 * 7 * 7, 4096),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(4096, 4096),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(4096, 1000),
		)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, 0, 0.01)
				nn.init.constant_(m.bias, 0)

	def forward(self, x):
		x = self.features(x)
		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x


class extractor(nn.Module):
	def __init__(self, pretrained):
		super(extractor, self).__init__()
		vgg16_bn = VGG(make_layers(cfg, batch_norm=True))
		if pretrained:
			vgg16_bn.load_state_dict(torch.load('./pretrained/vgg11_bn-6002323d.pth'))
		self.features = vgg16_bn.features

	def forward(self, x):
		out = []
		for m in self.features:
			x = m(x)
			if isinstance(m, nn.MaxPool2d):
				out.append(x)
		return out[1:]

class extractor_acb(nn.Module):
	def __init__(self, pretrained):
		super(extractor_acb, self).__init__()
		vgg16_bn = VGG(make_layers_acb(cfg, batch_norm=True))
		if pretrained:
			vgg16_bn.load_state_dict(torch.load('./pths/vgg16_bn-6c64b313.pth'))
		self.features = vgg16_bn.features
	
	def forward(self, x):
		out = []
		for m in self.features:
			x = m(x)
			if isinstance(m, nn.MaxPool2d):
				out.append(x)
		return out[1:]

class Conv2dBNReLU(nn.Module):
	def __init__(self,in_channels, out_channels, kernel_size , stride , padding ):
		super(Conv2dBNReLU,self).__init__()
		self.pre = nn.Sequential(
			nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias=False),#in out k s p
			nn.BatchNorm2d(num_features = out_channels),
			nn.ReLU(inplace=True)
		)
	def forward(self,x):
		return self.pre(x)

class BNReLUConv2d(nn.Module):
	def __init__(self,in_channels, out_channels, kernel_size , stride , padding ):
		super(BNReLUConv2d,self).__init__()
		self.pre = nn.Sequential(
			nn.BatchNorm2d(num_features = in_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias=False)#in out k s p
		)
	def forward(self,x):
		return self.pre(x)

class inception_block_dcn_v2(nn.Module):
	def __init__(self,inchannels,channels = 32):
		super(inception_block_dcn_v2,self).__init__()
		self.inchannels = inchannels
		self.channels = channels
		if self.inchannels!=self.channels:
			self.conv1_1 = BNReLUConv2d(inchannels,channels, 1, 1, 0)
			self.conv2_1 = BNReLUConv2d(inchannels,channels, 1, 1, 0)
			self.conv3_1 = BNReLUConv2d(inchannels,channels, 1, 1, 0)
			self.conv4_1 = BNReLUConv2d(inchannels,channels, 1, 1, 0)
			self.skip_layer = BNReLUConv2d(inchannels,channels, 1, 1, 0)
		self.conv1_2 = BNReLUConv2d(channels, channels, 1, 1, 0)
		self.conv1_3_offset = nn.Conv2d(channels,27,3, padding=1)
		self.conv1_3 = ModulatedDeformConv(channels,channels,kernel_size=3,stride=1,padding=1,bias=False)#3*3 dcn代
		self.bn1_3 = nn.BatchNorm2d(channels)
		self.relu1_3 = nn.ReLU()

		self.conv2_2 = BNReLUConv2d(channels, channels, kernel_size=(1, 3), stride =1, padding=(0, 1))
		self.conv2_3 = BNReLUConv2d(channels, channels, kernel_size=(3, 1), stride =1, padding=(1, 0))
		self.conv2_4_offset = nn.Conv2d(channels,27,3, padding=1)
		self.conv2_4 = ModulatedDeformConv(channels,channels,kernel_size=3,stride=1,padding=1,bias=False)
		self.bn2_4 = nn.BatchNorm2d(channels)
		self.relu2_4 = nn.ReLU()

		self.conv3_2 = BNReLUConv2d(channels, channels, kernel_size=(1, 5), stride =1, padding=(0, 2))
		self.conv3_3 = BNReLUConv2d(channels, channels, kernel_size=(5, 1), stride =1, padding=(2, 0))
		self.conv3_4_offset = nn.Conv2d(channels,27,3, padding=1)
		self.conv3_4 = ModulatedDeformConv(channels,channels,kernel_size=3,stride=1,padding=1,bias=False)#3*3 dcn代写
		self.bn3_4 = nn.BatchNorm2d(channels)
		self.relu3_4 = nn.ReLU()

		self.conv4_2 = BNReLUConv2d(channels, channels, kernel_size=(1, 7), stride =1, padding=(0, 3))
		self.conv4_3 = BNReLUConv2d(channels, channels, kernel_size=(7, 1), stride =1, padding=(3, 0))
		self.conv4_4_offset = nn.Conv2d(channels,27,3, padding=1)
		self.conv4_4 = ModulatedDeformConv(channels,channels,kernel_size=3,stride=1,padding=1,bias=False)#3*3 dcn代写
		self.bn4_4 = nn.BatchNorm2d(channels)
		self.relu4_4 = nn.ReLU()

		self.conv1_4_concat = BNReLUConv2d(channels*4,channels, 1, 1, 0)#27*3

		# self.convygx = BNReLUConv2d(channels,channels,3, stride =1, padding=1)


	def forward(self,x):
		if self.inchannels==self.channels:
			y1 = self.relu1_3(self.bn1_3(self.conv1_2(x)))
		else:
			y1 = self.relu1_3(self.bn1_3(self.conv1_2(self.conv1_1(x))))
		offset_mask1 = self.conv1_3_offset(y1)
		offset1 = offset_mask1[:, :18, :, :]
		mask1 = offset_mask1[:, -9:, :, :].sigmoid()
		out1 = self.conv1_3(y1, offset1, mask1)

		if self.inchannels==self.channels:
			y2 = self.relu2_4(self.bn2_4(self.conv2_3(self.conv2_2(x))))
		else:
			y2 = self.relu2_4(self.bn2_4(self.conv2_3(self.conv2_2(self.conv2_1(x)))))
		offset_mask2 = self.conv2_4_offset(y2)
		offset2 = offset_mask2[:, :18, :, :]
		mask2 = offset_mask2[:, -9:, :, :].sigmoid()
		out2 = self.conv2_4(y2, offset2, mask2)

		if self.inchannels==self.channels:
			y3 = self.relu3_4(self.bn3_4(self.conv3_3(self.conv3_2(x))))
		else:
			y3 = self.relu3_4(self.bn3_4(self.conv3_3(self.conv3_2(self.conv3_1(x)))))
		offset_mask3 = self.conv3_4_offset(y3)
		offset3 = offset_mask3[:, :18, :, :]
		mask3 = offset_mask3[:, -9:, :, :].sigmoid()
		out3 = self.conv3_4(y3, offset3, mask3)

		if self.inchannels==self.channels:
			y4 = self.relu4_4(self.bn4_4(self.conv4_3(self.conv4_2(x))))
		else:
			y4 = self.relu4_4(self.bn4_4(self.conv4_3(self.conv4_2(self.conv4_1(x)))))
		offset_mask4 = self.conv4_4_offset(y4)
		offset4 = offset_mask4[:, :18, :, :]
		mask4 = offset_mask4[:, -9:, :, :].sigmoid()
		out4 = self.conv4_4(y4, offset4, mask4)

		cat = self.conv1_4_concat(torch.cat((out1,out2,out3,out4),1))#,out4
		
		# if self.inchannels==self.channels:
		# 	cat = cat + x
		# else:
		# 	cat = cat + self.skip_layer(x)

		# cat = self.convygx(cat)#############
		return cat

class inception_block_dcn_v3(nn.Module):
	def __init__(self,inchannels,channels = 32):
		super(inception_block_dcn_v3,self).__init__()
		self.inchannels = inchannels
		self.channels = channels
  
		self.conv1_2 = BNReLUConv2d(32, channels, 1, 1, 0)
		self.conv1_3_offset = nn.Conv2d(channels,27,3, padding=1)
		self.conv1_3 = ModulatedDeformConv(channels,channels,kernel_size=3,stride=1,padding=1,bias=False)#3*3 dcn代
		self.bn1_3 = nn.BatchNorm2d(channels)
		self.relu1_3 = nn.ReLU()

		self.conv2_2 = BNReLUConv2d(32, channels, kernel_size=(1, 3), stride =1, padding=(0, 1))
		self.conv2_3 = BNReLUConv2d(channels, channels, kernel_size=(3, 1), stride =1, padding=(1, 0))
		self.conv2_4_offset = nn.Conv2d(2*channels,27,3, padding=1)
		self.conv2_4 = ModulatedDeformConv(2*channels,channels,kernel_size=3,stride=1,padding=1,bias=False)
		self.bn2_4 = nn.BatchNorm2d(2*channels)
		self.relu2_4 = nn.ReLU()

################
		self.conv3_1 = BNReLUConv2d(32, channels, kernel_size=(3, 3), stride =1, padding=1)
		self.conv3_2 = BNReLUConv2d(channels, channels, kernel_size=(1, 3), stride =1, padding=(0, 1))
		self.conv3_3 = BNReLUConv2d(channels, channels, kernel_size=(3, 1), stride =1, padding=(1, 0))
		self.conv3_4_offset = nn.Conv2d(2*channels,27,3, padding=1)
		self.conv3_4 = ModulatedDeformConv(2*channels,channels,kernel_size=3,stride=1,padding=1,bias=False)#3*3 dcn代写
		self.bn3_4 = nn.BatchNorm2d(2*channels)
		self.relu3_4 = nn.ReLU()

################
		self.conv4_1 = BNReLUConv2d(32, channels, kernel_size=(3, 3), stride =1, padding=1)
		self.conv4_2 = BNReLUConv2d(channels, channels, kernel_size=(1, 5), stride =1, padding=(0, 2))
		self.conv4_3 = BNReLUConv2d(channels, channels, kernel_size=(5, 1), stride =1, padding=(2, 0))
		self.conv4_4_offset = nn.Conv2d(2*channels,27,3, padding=1)
		self.conv4_4 = ModulatedDeformConv(2*channels,channels,kernel_size=3,stride=1,padding=1,bias=False)#3*3 dcn代写
		self.bn4_4 = nn.BatchNorm2d(2*channels)
		self.relu4_4 = nn.ReLU()

		self.conv1_4_concat = BNReLUConv2d(channels*4,32, 1, 1, 0)#27*3

		self.convygx = BNReLUConv2d(channels,channels,3, stride =1, padding=1)

	def forward(self,x):

		y1 = self.relu1_3(self.bn1_3(self.conv1_2(x)))

		offset_mask1 = self.conv1_3_offset(y1)
		offset1 = offset_mask1[:, :18, :, :]
		mask1 = offset_mask1[:, -9:, :, :].sigmoid()
		out1 = self.conv1_3(y1, offset1, mask1)

		y2 = self.relu2_4(self.bn2_4(torch.cat((self.conv2_2(x),self.conv2_3(x)),1)))
		offset_mask2 = self.conv2_4_offset(y2)
		offset2 = offset_mask2[:, :18, :, :]
		mask2 = offset_mask2[:, -9:, :, :].sigmoid()
		out2 = self.conv2_4(y2, offset2, mask2)
		
		y3_1 = self.conv3_1(x)
		y3 = self.relu3_4(self.bn3_4(torch.cat((self.conv3_2(y3_1),self.conv3_3(y3_1)),1)))
		offset_mask3 = self.conv3_4_offset(y3)
		offset3 = offset_mask3[:, :18, :, :]
		mask3 = offset_mask3[:, -9:, :, :].sigmoid()
		out3 = self.conv3_4(y3, offset3, mask3)

		y4_1 = self.conv4_1(x)
		y4 = self.relu4_4(self.bn4_4(torch.cat((self.conv4_2(y4_1),self.conv4_3(y4_1)),1)))
		offset_mask4 = self.conv4_4_offset(y4)
		offset4 = offset_mask4[:, :18, :, :]
		mask4 = offset_mask4[:, -9:, :, :].sigmoid()
		out4 = self.conv4_4(y4, offset4, mask4)
		cat = self.conv1_4_concat(torch.cat((out1,out2,out3,out4),1))#,out4
		
		# if self.inchannels==self.channels:
		# 	cat = cat + x
		# else:
		# 	cat = cat + self.skip_layer(x)

		cat = self.convygx(cat)#############
		return cat

class inception_block_dcn_v4(nn.Module):
	def __init__(self,inchannels,channels = 32):
		super(inception_block_dcn_v4,self).__init__()
		self.inchannels = inchannels
		self.channels = channels
  
		self.conv1_2 = BNReLUConv2d(32, channels, 1, 1, 0)
		self.conv1_3_offset = nn.Conv2d(channels,27,3, padding=1)
		self.conv1_3 = ModulatedDeformConv(channels,channels,kernel_size=3,stride=1,padding=1,bias=False)#3*3 dcn代
		self.bn1_3 = nn.BatchNorm2d(channels)
		self.relu1_3 = nn.ReLU()

		self.conv2_2 = BNReLUConv2d(32, channels, kernel_size=(1, 3), stride =1, padding=(0, 1))
		self.conv2_3 = BNReLUConv2d(channels, channels, kernel_size=(3, 1), stride =1, padding=(1, 0))
		self.conv2_4_offset = nn.Conv2d(2*channels,27,3, padding=1)
		self.conv2_4 = ModulatedDeformConv(2*channels,channels,kernel_size=3,stride=1,padding=1,bias=False)
		self.bn2_4 = nn.BatchNorm2d(2*channels)
		self.relu2_4 = nn.ReLU()
		
		# self.conv2_new = BNReLUConv2d(channels*2, channels, 1, 1, 0)

################
		# self.conv3_1 = BNReLUConv2d(32, channels, kernel_size=(3, 3), stride =1, padding=1)
		self.conv3_2 = BNReLUConv2d(channels, channels, kernel_size=(1, 5), stride =1, padding=(0, 2))
		self.conv3_3 = BNReLUConv2d(channels, channels, kernel_size=(5, 1), stride =1, padding=(2, 0))
		self.conv3_4_offset = nn.Conv2d(2*channels,27,3, padding=1)
		self.conv3_4 = ModulatedDeformConv(2*channels,channels,kernel_size=3,stride=1,padding=1,bias=False)#3*3 dcn代写
		self.bn3_4 = nn.BatchNorm2d(2*channels)
		self.relu3_4 = nn.ReLU()
		# self.conv3_new = BNReLUConv2d(channels*2, channels, 1, 1, 0)

################
		# self.conv4_1 = BNReLUConv2d(32, channels, kernel_size=(3, 3), stride =1, padding=1)
		self.conv4_2 = BNReLUConv2d(channels, channels, kernel_size=(1, 7), stride =1, padding=(0, 3))
		self.conv4_3 = BNReLUConv2d(channels, channels, kernel_size=(7, 1), stride =1, padding=(3, 0))
		self.conv4_4_offset = nn.Conv2d(2*channels,27,3, padding=1)
		self.conv4_4 = ModulatedDeformConv(2*channels,channels,kernel_size=3,stride=1,padding=1,bias=False)#3*3 dcn代写
		self.bn4_4 = nn.BatchNorm2d(2*channels)
		self.relu4_4 = nn.ReLU()
		# self.conv4_new = BNReLUConv2d(channels*2, channels, 1, 1, 0)

		self.conv1_4_concat = BNReLUConv2d(channels*4,32, 1, 1, 0)#27*3

		self.convygx = BNReLUConv2d(channels,channels,3, stride =1, padding=1)

	def forward(self,x):

		y1 = self.relu1_3(self.bn1_3(self.conv1_2(x)))

		offset_mask1 = self.conv1_3_offset(y1)
		offset1 = offset_mask1[:, :18, :, :]
		mask1 = offset_mask1[:, -9:, :, :].sigmoid()
		out1 = self.conv1_3(y1, offset1, mask1)

		y2 = self.relu2_4(self.bn2_4(torch.cat((self.conv2_2(x),self.conv2_3(x)),1)))
		# y2 = self.conv2_new(y2)##########
		offset_mask2 = self.conv2_4_offset((y2))
		offset2 = offset_mask2[:, :18, :, :]
		mask2 = offset_mask2[:, -9:, :, :].sigmoid()
		out2 = self.conv2_4(y2, offset2, mask2)
		
		# y3_1 = self.conv3_1(x)
		y3 = self.relu3_4(self.bn3_4(torch.cat((self.conv3_2(x),self.conv3_3(x)),1)))
		# y3 = self.conv3_new(y3)##########
		offset_mask3 = self.conv3_4_offset(y3)
		offset3 = offset_mask3[:, :18, :, :]
		mask3 = offset_mask3[:, -9:, :, :].sigmoid()
		out3 = self.conv3_4(y3, offset3, mask3)

		# y4_1 = self.conv4_1(x)
		y4 = self.relu4_4(self.bn4_4(torch.cat((self.conv4_2(x),self.conv4_3(x)),1)))
		# y4 = self.conv4_new(y4)##########
		offset_mask4 = self.conv4_4_offset(y4)
		offset4 = offset_mask4[:, :18, :, :]
		mask4 = offset_mask4[:, -9:, :, :].sigmoid()
		out4 = self.conv4_4(y4, offset4, mask4)
		cat = self.conv1_4_concat(torch.cat((out1,out2,out3,out4),1))#,out4
		
		# if self.inchannels==self.channels:
		# 	cat = cat + x
		# else:
		# 	cat = cat + self.skip_layer(x)

		cat = self.convygx(cat)#############
		return cat

class inception_block_dcn(nn.Module):
	def __init__(self,channels = 32):
		super(inception_block_dcn,self).__init__()
		#self.conv1_1 = Conv2dBNReLU(32,channels,1)
		#self.conv1_2 = Conv2dBNReLU(chconv1_5blockannels,channels,1)

		self.conv1_2 = nn.Conv2d(channels, channels, 1)
		self.bn1_2 = nn.BatchNorm2d(channels)
		self.relu1_2 = nn.ReLU()
		self.conv1_3_offset = nn.Conv2d(channels,27,3, padding=1)
		self.conv1_3 = ModulatedDeformConv(channels,32,kernel_size=3,stride=1,padding=1,bias=False)#3*3 dcn代写


		#self.conv2_1 = Conv2dBNReLU(32,channels,1)
		# self.conv2_2 = Conv2dBNReLU(channels, channels, kernel_size=(1, 3),padding=(0, 1))
		# self.conv2_3 = Conv2dBNReLU(channels, channels, kernel_size=(3, 1),padding=(1, 0))
		self.conv2_2 = nn.Conv2d(channels, channels, kernel_size=(1, 3),padding=(0, 1))
		self.bn2_2 = nn.BatchNorm2d(channels)
		self.relu2_2 = nn.ReLU()
		self.conv2_3 = nn.Conv2d(channels, channels, kernel_size=(3, 1),padding=(1, 0))
		self.bn2_3 = nn.BatchNorm2d(channels)
		self.relu2_3 = nn.ReLU()
		self.conv2_4_offset = nn.Conv2d(channels,27,3, padding=1)
		self.conv2_4 = ModulatedDeformConv(channels,32,kernel_size=3,stride=1,padding=1,bias=False)#3*3 dcn代写


		#self.conv3_1 = Conv2dBNReLU(32,channel,1)
		# self.conv3_2 = Conv2dBNReLU(channels, channels, kernel_size=(1, 5),padding=(0, 2))
		# self.conv3_3 = Conv2dBNReLU(channels, channels, kernel_size=(5, 1),padding=(2, 0))
		self.conv3_2 = nn.Conv2d(channels, channels, kernel_size=(1, 5),padding=(0, 2))
		self.bn3_2 = nn.BatchNorm2d(channels)
		self.relu3_2 = nn.ReLU()
		self.conv3_3 = nn.Conv2d(channels, channels, kernel_size=(5, 1),padding=(2, 0))
		self.bn3_3 = nn.BatchNorm2d(channels)
		self.relu3_3 = nn.ReLU()
		self.conv3_4_offset = nn.Conv2d(channels,27,3, padding=1)
		self.conv3_4 = ModulatedDeformConv(channels,32,kernel_size=3,stride=1,padding=1,bias=False)#3*3 dcn代写


		#self.conv4_1 = Conv2dBNReLU(32,channel,1)
		##########################################################################33
		self.conv4_2 = nn.Conv2d(channels, channels, kernel_size=(1, 7),padding=(0, 3))
		self.bn4_2 = nn.BatchNorm2d(channels)
		self.relu4_2 = nn.ReLU()
		self.conv4_3 = nn.Conv2d(channels, channels, kernel_size=(7, 1),padding=(3, 0))
		self.bn4_3 = nn.BatchNorm2d(channels)
		self.relu4_3 = nn.ReLU()
		self.conv4_4_offset = nn.Conv2d(channels,27,3, padding=1)
		self.conv4_4 = ModulatedDeformConv(channels,32,kernel_size=3,stride=1,padding=1,bias=False)#3*3 dcn代写
		# self.conv4_4 = ModulatedDeformConv(channels,channels,kernel_size=3,stride=1,padding=1,bias=False)#3*3 dcn代写
		############################################################################

		self.conv1_3_concat = nn.Conv2d(32*4,channels,1)#27*3
	
		self.bn1_3_concat = nn.BatchNorm2d(channels)
		self.relu1_3_concat = nn.ReLU()

		self.convygx = nn.Conv2d(channels,channels,3, padding=1)
		self.bnygx = nn.BatchNorm2d(channels)
		self.reluygx = nn.ReLU()



		self.relu = nn.ReLU()
			

	def forward(self,x):
		y1 = self.relu1_2(self.bn1_2(self.conv1_2(x)))
		offset_mask1 = self.conv1_3_offset(y1)
		offset1 = offset_mask1[:, :18, :, :]
		mask1 = offset_mask1[:, -9:, :, :].sigmoid()
		out1 = self.conv1_3(y1, offset1, mask1)
		#y1 = self.conv1_3(self.conv1_2(x))
		y2 = (self.relu2_3(self.bn2_3(self.conv2_3(self.relu2_2(self.bn2_2(self.conv2_2(x)))))))
		offset_mask2 = self.conv2_4_offset(y2)
		offset2 = offset_mask2[:, :18, :, :]
		mask2 = offset_mask2[:, -9:, :, :].sigmoid()
		out2 = self.conv2_4(y2, offset2, mask2)

		y3 = (self.relu3_3(self.bn3_3(self.conv3_3(self.relu3_2(self.bn3_2(self.conv3_2(x)))))))
		offset_mask3 = self.conv3_4_offset(y3)
		offset3 = offset_mask3[:, :18, :, :]
		mask3 = offset_mask3[:, -9:, :, :].sigmoid()
		out3 = self.conv3_4(y3, offset3, mask3)

		y4 = (self.relu4_3(self.bn4_3(self.conv4_3(self.relu4_2(self.bn4_2(self.conv4_2(x)))))))
		offset_mask4 = self.conv4_4_offset(y4)
		offset4 = offset_mask4[:, :18, :, :]
		mask4 = offset_mask4[:, -9:, :, :].sigmoid()
		out4 = self.conv4_4(y4, offset4, mask4)

		cat = self.conv1_3_concat(torch.cat((out1,out2,out3,out4),1))#,out4

		# cat = self.reluygx(self.bnygx(self.convygx(cat)))

		y0 = self.relu(cat)# y0 = self.relu(torch.add(cat,x))
		y0 = self.reluygx(self.bnygx(self.convygx(y0)))
		#print(y0.size())

		return y0
class merge_v2(nn.Module):#conv1_5 = True表示2的5次，正常操作，false就是最后一层去除掉了
	def __init__(self,inception_mid=False,inception_end=False, conv1_5 = True, two = 128, thr = 256, four = 512, five = 512): #64 128 256 384

		super(merge_v2, self).__init__()
		self.inception_end = inception_end
		self.inception_mid = inception_mid
		if self.inception_mid:
			self.conv3 = inception_block_dcn_v2(four+thr,64)
			self.conv4 = inception_block_dcn_v2(64+two, 32)
		else:
			self.conv3 = nn.Conv2d(four+thr, 64, 1)#256+128
			self.bn3 = nn.BatchNorm2d(64)
			self.relu3 = nn.ReLU()
			self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
			self.bn4 = nn.BatchNorm2d(64)
			self.relu4 = nn.ReLU()
			self.conv5 = nn.Conv2d(64+two, 32, 1)#128+64
			self.bn5 = nn.BatchNorm2d(32)
			self.relu5 = nn.ReLU()
			self.conv6 = nn.Conv2d(32, 32, 3, padding=1)
			self.bn6 = nn.BatchNorm2d(32)
			self.relu6 = nn.ReLU()
		if not self.inception_end:
			self.conv7 = nn.Conv2d(32, 32, 3, padding=1)
			self.bn7 = nn.BatchNorm2d(32)
			self.relu7 = nn.ReLU()
		else:
			self.conv_inception = inception_block_dcn_v4(32, 32)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				#print("nn.Conv2d: %s" %m)
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				#print("nn.BatchNorm2d %s" %m)
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

		for m in self.modules():
			# print(m)
			if hasattr(m, 'conv1_3_offset'):
				constant_init(m.conv1_3_offset, 0)
			if hasattr(m, 'conv2_4_offset'):
				constant_init(m.conv2_4_offset, 0)
			if hasattr(m, 'conv3_4_offset'):
				constant_init(m.conv3_4_offset, 0)
			if hasattr(m, 'conv4_4_offset'):
				constant_init(m.conv4_4_offset, 0)

	def forward(self, x):
		y = F.interpolate(x[2], scale_factor=2, mode='bilinear', align_corners=True)
		y = torch.cat((y, x[1]), 1)
		if self.inception_mid:
			y = self.conv3(y)
			y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
			y = torch.cat((y, x[0]), 1)
			y = self.conv4(y)
		else:
			y = self.relu3(self.bn3(self.conv3(y)))
			y = self.relu4(self.bn4(self.conv4(y)))
			y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
			y = torch.cat((y, x[0]), 1)
			y = self.relu5(self.bn5(self.conv5(y)))
			y = self.relu6(self.bn6(self.conv6(y)))
		if not self.inception_end:
		    y = self.relu7(self.bn7(self.conv7(y)))
		else:
			y = self.conv_inception(y)

		return y 



class merge(nn.Module):#conv1_5 = True表示2的5次，正常操作，false就是最后一层去除掉了
	def __init__(self,inception=False, conv1_5 = True, two = 128, thr = 256, four = 512, five = 512):

		super(merge, self).__init__()
		self.inception = inception
		self.conv1_5 = conv1_5
		if self.conv1_5:
			self.conv1 = nn.Conv2d(four+five, 128, 1)# 512+512
			self.bn1 = nn.BatchNorm2d(128)
			self.relu1 = nn.ReLU()
			self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
			self.bn2 = nn.BatchNorm2d(128)
			self.relu2 = nn.ReLU()

			self.conv3 = nn.Conv2d(128+thr, 64, 1)#256+128
		else:
			self.conv3 = nn.Conv2d(four+thr, 64, 1)#256+128

		self.bn3 = nn.BatchNorm2d(64)
		self.relu3 = nn.ReLU()
		self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
		self.bn4 = nn.BatchNorm2d(64)
		self.relu4 = nn.ReLU()

		self.conv5 = nn.Conv2d(64+two, 32, 1)#128+64
		self.bn5 = nn.BatchNorm2d(32)
		self.relu5 = nn.ReLU()
		self.conv6 = nn.Conv2d(32, 32, 3, padding=1)
		self.bn6 = nn.BatchNorm2d(32)
		self.relu6 = nn.ReLU()
		if not self.inception:
			self.conv7 = nn.Conv2d(32, 32, 3, padding=1)
			self.bn7 = nn.BatchNorm2d(32)
			self.relu7 = nn.ReLU()
		else:
			self.conv_inception = inception_block_dcn_v4(32)#(32,32)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
		for m in self.modules():
			if hasattr(m, 'conv1_3_offset'):
				# print(m.conv1_3_offset)
				constant_init(m.conv1_3_offset, 0)
			if hasattr(m, 'conv2_4_offset'):
				constant_init(m.conv2_4_offset, 0)
			if hasattr(m, 'conv3_4_offset'):
				constant_init(m.conv3_4_offset, 0)
			if hasattr(m, 'conv4_4_offset'):
				constant_init(m.conv4_4_offset, 0)


	def forward(self, x):
		if self.conv1_5:
			y = F.interpolate(x[3], scale_factor=2, mode='bilinear', align_corners=True)
			# y = self.t1(x[3])
			y = torch.cat((y, x[2]), 1)
			y = self.relu1(self.bn1(self.conv1(y)))		
			y = self.relu2(self.bn2(self.conv2(y)))
		
			y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
		else:
			y = F.interpolate(x[2], scale_factor=2, mode='bilinear', align_corners=True)
		# y = self.t2(y)
		y = torch.cat((y, x[1]), 1)
		y = self.relu3(self.bn3(self.conv3(y)))		
		y = self.relu4(self.bn4(self.conv4(y)))
		
		y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
		# y = self.t3(y)
		y = torch.cat((y, x[0]), 1)

		y = self.relu5(self.bn5(self.conv5(y)))		
		y = self.relu6(self.bn6(self.conv6(y)))
		if not self.inception:
		    y = self.relu7(self.bn7(self.conv7(y)))
		else:
			y = self.conv_inception(y)
		return y 


class merge_tcnn(nn.Module):
	def __init__(self):
		super(merge_tcnn, self).__init__()

		self.conv1 = nn.Conv2d(1024, 128, 1)
		self.bn1 = nn.BatchNorm2d(128)
		self.relu1 = nn.ReLU()
		self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
		self.bn2 = nn.BatchNorm2d(128)
		self.relu2 = nn.ReLU()

		self.conv3 = nn.Conv2d(384, 64, 1)
		self.bn3 = nn.BatchNorm2d(64)
		self.relu3 = nn.ReLU()
		self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
		self.bn4 = nn.BatchNorm2d(64)
		self.relu4 = nn.ReLU()

		self.conv5 = nn.Conv2d(192, 32, 1)
		self.bn5 = nn.BatchNorm2d(32)
		self.relu5 = nn.ReLU()
		self.conv6 = nn.Conv2d(32, 32, 3, padding=1)
		self.bn6 = nn.BatchNorm2d(32)
		self.relu6 = nn.ReLU()

		self.conv7 = nn.Conv2d(32, 32, 3, padding=1)
		self.bn7 = nn.BatchNorm2d(32)
		self.relu7 = nn.ReLU()
		##############
		self.t1 = nn.ConvTranspose2d(512, 512, 3, stride=2, padding=1, output_padding=1)
		self.t2 = nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1, output_padding=1)
		self.t3 = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1)
		###############
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def forward(self, x):
		
		#y = F.interpolate(x[3], scale_factor=2, mode='bilinear', align_corners=True)
		y = self.t1(x[3])
		y = torch.cat((y, x[2]), 1)
		y = self.relu1(self.bn1(self.conv1(y)))		
		y = self.relu2(self.bn2(self.conv2(y)))
		
		#y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
		y = self.t2(y)
		y = torch.cat((y, x[1]), 1)
		y = self.relu3(self.bn3(self.conv3(y)))		
		y = self.relu4(self.bn4(self.conv4(y)))
		
		#y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
		y = self.t3(y)
		y = torch.cat((y, x[0]), 1)
		y = self.relu5(self.bn5(self.conv5(y)))		
		y = self.relu6(self.bn6(self.conv6(y)))
		
		y = self.relu7(self.bn7(self.conv7(y)))
		return y

class output(nn.Module):
	def __init__(self, scope=512):
		super(output, self).__init__()
		self.conv1 = nn.Conv2d(32, 1, 1)
		self.sigmoid1 = nn.Sigmoid()
		self.conv2 = nn.Conv2d(32, 4, 1)
		self.sigmoid2 = nn.Sigmoid()
		self.conv3 = nn.Conv2d(32, 1, 1)
		self.sigmoid3 = nn.Sigmoid()
		self.scope = 512

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)

	def forward(self, x):
		#print((x))
		score = self.sigmoid1(self.conv1(x))
		loc   = self.sigmoid2(self.conv2(x)) * self.scope
		angle = (self.sigmoid3(self.conv3(x)) - 0.5) * math.pi
		geo   = torch.cat((loc, angle), 1) 
		return score, geo
		
	
class EAST(nn.Module):
	def __init__(self, pretrained=False, conv1_5 = False,inception =True):
		super(EAST, self).__init__()
		self.extractor = extractor(pretrained)
		self.merge     = merge(inception=inception,conv1_5 = conv1_5)
		self.output    = output(512)############
	
	def forward(self, x):

		#print(self.extractor(x).size())
		y = self.merge(self.extractor(x))
		#print(y.size())
		return self.output(y)

class EAST_acb(nn.Module):
	def __init__(self, pretrained=False):
		super(EAST_acb, self).__init__()
		self.extractor = extractor_acb(pretrained)
		self.merge     = merge()
		self.output    = output(512)############
	
	def forward(self, x):
		
		return self.output(self.merge(self.extractor(x)))

class EAST_tcnn(nn.Module):
	def __init__(self, pretrained=False):
		super(EAST_tcnn, self).__init__()
		self.extractor = extractor(pretrained)
		self.merge     = merge_tcnn()
		self.output    = output(512)############
	
	def forward(self, x):
		return self.output(self.merge(self.extractor(x)))
		


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
		

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        f = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        f.append(x)
        x = self.layer2(x)
        f.append(x)
        x = self.layer3(x)
        f.append(x)
        x = self.layer4(x)
        f.append(x)
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        '''
        f中的每个元素的size分别是 bs 256 w/4 h/4， bs 512 w/8 h/8， 
        bs 1024 w/16 h/16， bs 2048 w/32 h/32
        '''
        return f


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load("./pretrained/resnet50-19c8e357.pth"))
    return model


class East_res(nn.Module):
    def __init__(self, pretrained=False):
        super(East_res, self).__init__()
        self.resnet = resnet50(pretrained)##################resnet50(True)
        self.conv1 = nn.Conv2d(3072, 128, 1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(128,64,1)

        self.bn3 = nn.BatchNorm2d(64)

        self.relu3 = nn.ReLU()


        self.conv4 = nn.Conv2d(64, 64, 3 ,padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(320, 64, 1)
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU()

        self.conv6 = nn.Conv2d(64, 32, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(32)
        self.relu6 = nn.ReLU()

        self.conv7 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(32)
        self.relu7 = nn.ReLU()

        self.conv8 = nn.Conv2d(32, 1, 1)
        self.sigmoid1 = nn.Sigmoid()
        self.conv9 = nn.Conv2d(32, 4, 1)
        self.sigmoid2 = nn.Sigmoid()
        self.conv10 = nn.Conv2d(32, 1, 1)
        self.sigmoid3 = nn.Sigmoid()
        self.unpool1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.unpool2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.unpool3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self,x):
        
        f = self.resnet(x)
        h = f[3]  # bs 2048 w/32 h/32
        g = (self.unpool1(h)) #bs 2048 w/16 h/16
        c = self.conv1(torch.cat((g, f[2]), 1))
        c = self.bn1(c)
        c = self.relu1(c)
        
        h = self.conv2(c)  # bs 128 w/16 h/16
        h = self.bn2(h)
        h = self.relu2(h)
        g = self.unpool2(h) # bs 128 w/8 h/8
        c = self.conv3(torch.cat((g, f[1]), 1))
        c = self.bn3(c)
        c = self.relu3(c)

        h = self.conv4(c)  # bs 64 w/8 h/8
        h = self.bn4(h)
        h = self.relu4(h)
        g = self.unpool3(h) # bs 64 w/4 h/4
        c = self.conv5(torch.cat((g, f[0]), 1))
        c = self.bn5(c)
        c = self.relu5(c)
        
        h = self.conv6(c) # bs 32 w/4 h/4
        h = self.bn6(h)
        h = self.relu6(h)
        g = self.conv7(h) # bs 32 w/4 h/4
        g = self.bn7(g)
        g = self.relu7(g)
        
        F_score = self.conv8(g) #  bs 1 w/4 h/4
        F_score = self.sigmoid1(F_score)
        geo_map = self.conv9(g)
        geo_map = self.sigmoid2(geo_map) * 512##########################################3
        angle_map = self.conv10(g)
        angle_map = self.sigmoid3(angle_map)
        angle_map = (angle_map - 0.5) * math.pi / 2###################################

        F_geometry = torch.cat((geo_map, angle_map), 1) # bs 5 w/4 w/4
        return F_score, F_geometry


# crelu
class Conv2dBNCReLU(nn.Module):
	def __init__(self,in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1):
		super(Conv2dBNCReLU,self).__init__()
		self.pre = nn.Sequential(
			nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias=False),#in out k s p
			nn.BatchNorm2d(num_features = out_channels),
		)
		self.relu = nn.ReLU(inplace=True)
	def forward(self, x):
		y = self.pre(x)
		return torch.cat((self.relu(y),self.relu(-y)), 1)
		#return self.relu(torch.cat((self.pre(x), -self.pre(x))))


class Conv2d_1_3_1(nn.Module):
	def __init__(self, in_channels, mid_channels, out_channels, stride = 1, padding = 1,dcn=False):
		super(Conv2d_1_3_1,self).__init__()
		self.layer1 = Conv2dBNReLU(in_channels, mid_channels, kernel_size = 1, stride = stride, padding = 0)
		self.dcn=dcn
		if not self.dcn:
			self.layer2 = Conv2dBNCReLU(mid_channels, mid_channels, kernel_size = 3 , stride = 1, padding = 1)
		else:
			self.conv3x3_dcn_offset = nn.Conv2d(mid_channels,18,kernel_size=3,stride = 1, padding = 1)
			self.conv3x3_dcn = DeformConv(mid_channels,mid_channels,kernel_size=3,stride = 1, padding = 1)
			self.bn3x3_bn = nn.BatchNorm2d(mid_channels)
			self.bn3x3_relu = nn.ReLU()
		self.layer3 = Conv2dBNReLU(mid_channels*2, out_channels, kernel_size = 1, stride = 1, padding = 0)
		self.skip_layer = nn.Conv2d(in_channels,out_channels,1,stride = stride,padding = 0)
	def forward(self, x):
		y = self.layer1(x)
		if not self.dcn:
			y = self.layer2(y)
		else:
			y = self.bn3x3_relu(self.bn3x3_bn(self.conv3x3_dcn(y,self.conv3x3_dcn_offset(y))))
			y = torch.cat((self.bn3x3_relu(y),self.bn3x3_relu(-y)), 1)
		y = self.layer3(y)
		if x.size()[1:]==y.size()[1:]:
			return y+x
		else:
			return y + self.skip_layer(x)


class Inception_PVA_v1(nn.Module):
	def __init__(self, in_channels, mid_channels_3x3_1, mid_channels_3x3_2, mid_channels_5x5_1, mid_channels_5x5_2, out_channels, stride = 1, padding = 1 , acb_block = False, dcn =False,with_modulated_dcn=False):
		super(Inception_PVA_v1,self).__init__()
		self.conv1x1 = Conv2dBNReLU(in_channels,64,kernel_size=1, stride = 1, padding = 0)
		self.acb_block = acb_block
		self.dcn = dcn
		self.with_modulated_dcn = with_modulated_dcn
		if self.dcn:
			if not self.with_modulated_dcn:
				conv_op = DeformConv
				offset_channels = 18
			else:
				conv_op = ModulatedDeformConv
				offset_channels = 27

			self.conv3x3_1 = Conv2dBNReLU(in_channels,mid_channels_3x3_1, kernel_size=1, stride = 1, padding = 0)
			self.conv3x3_dcn_offset = nn.Conv2d(mid_channels_3x3_1,offset_channels,kernel_size=3,stride = 1, padding = 1)
			self.conv3x3_dcn = conv_op(mid_channels_3x3_1,mid_channels_3x3_2,kernel_size=3,stride = 1, padding = 1)
			self.bn3x3_bn = nn.BatchNorm2d(mid_channels_3x3_2)
			self.bn3x3_relu = nn.ReLU()

			self.conv5x5_1 = Conv2dBNReLU(in_channels,mid_channels_5x5_1, kernel_size=1, stride = 1, padding = 0)
			self.conv5x5_dcn_offset_1 = nn.Conv2d(mid_channels_5x5_1,offset_channels,kernel_size=3,stride = 1, padding = 1)
			self.conv5x5_dcn_1 = conv_op(mid_channels_5x5_1,mid_channels_5x5_2,kernel_size=3,stride = 1, padding = 1)
			self.bn5x5_bn_1 = nn.BatchNorm2d(mid_channels_5x5_2)
			self.bn5x5_relu_1 = nn.ReLU()
			self.conv5x5_dcn_offset_2 = nn.Conv2d(mid_channels_5x5_2,offset_channels,kernel_size=3,stride = 1, padding = 1)
			self.conv5x5_dcn_2 = conv_op(mid_channels_5x5_2,mid_channels_5x5_2,kernel_size=3,stride = 1, padding = 1)
			self.bn5x5_bn_2 = nn.BatchNorm2d(mid_channels_5x5_2)
			self.bn5x5_relu_2 = nn.ReLU()

		else:
			if not acb_block:
				self.conv3x3_1 = Conv2dBNReLU(in_channels,mid_channels_3x3_1, kernel_size=1, stride = 1, padding = 0)
				self.conv3x3_2 = Conv2dBNReLU(mid_channels_3x3_1,mid_channels_3x3_2, kernel_size=3, stride = 1, padding = 1)
				self.conv5x5_1 = Conv2dBNReLU(in_channels,mid_channels_5x5_1, kernel_size=1, stride = 1, padding = 0)
				self.conv5x5_2 = Conv2dBNReLU(mid_channels_5x5_1,mid_channels_5x5_2, kernel_size=3, stride = 1, padding = 1)
				self.conv5x5_3 = Conv2dBNReLU(mid_channels_5x5_2,mid_channels_5x5_2, kernel_size=3, stride = 1, padding = 1)
			else:
				self.conv3x3_1 = Conv2dBNReLU(in_channels,mid_channels_3x3_1, kernel_size=1, stride = 1, padding = 0)
				self.conv3x3_2 = Conv2dBNReLU(mid_channels_3x3_1,mid_channels_3x3_2, kernel_size=(1, 3), stride = 1, padding = (0, 1))
				self.conv3x3_3 = Conv2dBNReLU(mid_channels_3x3_2,mid_channels_3x3_2, kernel_size=(3, 1), stride = 1, padding = (1, 0))

				self.conv5x5_1 = Conv2dBNReLU(in_channels,mid_channels_5x5_1, kernel_size=1, stride = 1, padding = 0)
				self.conv5x5_2 = Conv2dBNReLU(mid_channels_5x5_1,mid_channels_5x5_2, kernel_size=(1, 5), stride = 1, padding = (0, 2))
				self.conv5x5_3 = Conv2dBNReLU(mid_channels_5x5_2,mid_channels_5x5_2, kernel_size=(5, 1), stride = 1, padding = (2, 0))

		self.outlayer = Conv2dBNReLU(64 + mid_channels_3x3_2 + mid_channels_5x5_2, out_channels, kernel_size=1, stride = 1, padding =0)
		self.skip_layer = nn.Conv2d(in_channels,out_channels,kernel_size=1, stride = 1, padding =0)
		
    
	def forward(self,x):
		if not self.dcn:
			if not self.acb_block:
				y = torch.cat((self.conv1x1(x), self.conv3x3_2(self.conv3x3_1(x)), self.conv5x5_3(self.conv5x5_2(self.conv5x5_1(x)))), 1)
			else:
				y = torch.cat((self.conv1x1(x), self.conv3x3_3(self.conv3x3_2(self.conv3x3_1(x))), self.conv5x5_3(self.conv5x5_2(self.conv5x5_1(x)))), 1)
		else:
			y1 = self.conv1x1(x)
			if self.with_modulated_dcn:
				y2 = self.conv3x3_1(x)
				offset_mask2 = self.conv3x3_dcn_offset(y2)
				offset2 = offset_mask2[:, :18, :, :]
				mask2 = offset_mask2[:, -9:, :, :].sigmoid()
				y2 = self.bn3x3_relu(self.bn3x3_bn(self.conv3x3_dcn(y2, offset2, mask2)))

				y3 = self.conv5x5_1(x)
				offset_mask3_1 = self.conv5x5_dcn_offset_1(y3)
				offset3_1 = offset_mask3_1[:, :18, :, :]
				mask3_1 = offset_mask3_1[:, -9:, :, :].sigmoid()
				y3 = self.bn5x5_relu_1(self.bn5x5_bn_1(self.conv5x5_dcn_1(y3, offset3_1, mask3_1)))

				offset_mask3_2 = self.conv5x5_dcn_offset_2(y3)
				offset3_2 = offset_mask3_2[:, :18, :, :]
				mask3_2 = offset_mask3_2[:, -9:, :, :].sigmoid()
				y3 = self.bn5x5_relu_2(self.bn5x5_bn_2(self.conv5x5_dcn_2(y3, offset3_2, mask3_2)))

			else:
				y2 = self.conv3x3_1(x)
				y2 = self.bn3x3_relu(self.bn3x3_bn(self.conv3x3_dcn(y2, self.conv3x3_dcn_offset(y2))))
				y3 = self.conv5x5_1(x)
				y3 = self.bn5x5_relu_1(self.bn5x5_bn_1(self.conv5x5_dcn_1(y3, self.conv5x5_dcn_offset_1(y3))))
				y3 = self.bn5x5_relu_2(self.bn5x5_bn_2(self.conv5x5_dcn_2(y3, self.conv5x5_dcn_offset_2(y3))))
			y = torch.cat((y1, y2, y3), 1)

		y = self.outlayer(y) 

		return y + self.skip_layer(x)
		
class Inception_PVA_v2(nn.Module):
	def __init__(self,in_channels, mid_channels_3x3_1, mid_channels_3x3_2, mid_channels_5x5_1, mid_channels_5x5_2, out_channels, stride = 1, padding = 1, acb_block = False, dcn =False,with_modulated_dcn=False):
		super(Inception_PVA_v2,self).__init__()
		self.acb_block = acb_block
		self.conv1x1 = Conv2dBNReLU(in_channels,64,kernel_size=1, stride = 2, padding = 0)#################################

		self.dcn = dcn
		self.with_modulated_dcn = with_modulated_dcn
		if self.dcn:
			if not self.with_modulated_dcn:
				conv_op = DeformConv
				offset_channels = 18
			else:
				conv_op = ModulatedDeformConv
				offset_channels = 27
			self.conv3x3_1 = Conv2dBNReLU(in_channels,mid_channels_3x3_1, kernel_size=1, stride = 2, padding = 0)
			self.conv3x3_dcn_offset = nn.Conv2d(mid_channels_3x3_1,offset_channels,kernel_size=3,stride = 1, padding = 1)
			self.conv3x3_dcn = conv_op(mid_channels_3x3_1,mid_channels_3x3_2,kernel_size=3,stride = 1, padding = 1)
			self.bn3x3_bn = nn.BatchNorm2d(mid_channels_3x3_2)
			self.bn3x3_relu = nn.ReLU()

			self.conv5x5_1 = Conv2dBNReLU(in_channels,mid_channels_5x5_1, kernel_size=1, stride = 2, padding = 0)
			self.conv5x5_dcn_offset_1 = nn.Conv2d(mid_channels_5x5_1,offset_channels,kernel_size=3,stride = 1, padding = 1)
			self.conv5x5_dcn_1 = conv_op(mid_channels_5x5_1,mid_channels_5x5_2,kernel_size=3,stride = 1, padding = 1)
			self.bn5x5_bn_1 = nn.BatchNorm2d(mid_channels_5x5_2)
			self.bn5x5_relu_1 = nn.ReLU()
			self.conv5x5_dcn_offset_2 = nn.Conv2d(mid_channels_5x5_2,offset_channels,kernel_size=3,stride = 1, padding = 1)
			self.conv5x5_dcn_2 = conv_op(mid_channels_5x5_2,mid_channels_5x5_2,kernel_size=3,stride = 1, padding = 1)
			self.bn5x5_bn_2 = nn.BatchNorm2d(mid_channels_5x5_2)
			self.bn5x5_relu_2 = nn.ReLU()

		else:
			if not acb_block:
				self.conv3x3_1 = Conv2dBNReLU(in_channels,mid_channels_3x3_1, kernel_size=1, stride = 2, padding = 0)
				self.conv3x3_2 = Conv2dBNReLU(mid_channels_3x3_1,mid_channels_3x3_2, kernel_size=3, stride = 1, padding = 1)
				self.conv5x5_1 = Conv2dBNReLU(in_channels,mid_channels_5x5_1, kernel_size=1, stride = 2, padding = 0)
				self.conv5x5_2 = Conv2dBNReLU(mid_channels_5x5_1,mid_channels_5x5_2, kernel_size=3, stride = 1, padding = 1)
				self.conv5x5_3 = Conv2dBNReLU(mid_channels_5x5_2,mid_channels_5x5_2, kernel_size=3, stride = 1, padding = 1)
			else:
				self.conv3x3_1 = Conv2dBNReLU(in_channels,mid_channels_3x3_1, kernel_size=1, stride = 2, padding = 0)
				self.conv3x3_2 = Conv2dBNReLU(mid_channels_3x3_1,mid_channels_3x3_2, kernel_size=(1, 3), stride = 1, padding = (0, 1))
				self.conv3x3_3 = Conv2dBNReLU(mid_channels_3x3_2,mid_channels_3x3_2, kernel_size=(3, 1), stride = 1, padding = (1, 0))
				self.conv5x5_1 = Conv2dBNReLU(in_channels,mid_channels_5x5_1, kernel_size=1, stride = 2, padding = 0)
				self.conv5x5_2 = Conv2dBNReLU(mid_channels_5x5_1,mid_channels_5x5_2, kernel_size=(1, 5), stride = 1, padding = (0, 2))
				self.conv5x5_3 = Conv2dBNReLU(mid_channels_5x5_2,mid_channels_5x5_2, kernel_size=(5, 1), stride = 1, padding = (2, 0))

		self.pool_1 =  nn.MaxPool2d(kernel_size=3, stride=2, padding=1)#
		self.pool_2 =  Conv2dBNReLU(in_channels, 128, kernel_size=1, stride = 1, padding = 0)
		self.outlayer = Conv2dBNReLU(64 + mid_channels_3x3_2 + mid_channels_5x5_2 + 128, out_channels, kernel_size=1, stride = 1, padding =0)
		self.skip_layer = nn.Conv2d(in_channels,out_channels,kernel_size=1, stride = 2, padding =0)


    
	def forward(self,x):
		if not self.dcn:
			if not self.acb_block:
				y = torch.cat((self.conv1x1(x), self.conv3x3_2(self.conv3x3_1(x)), self.conv5x5_3(self.conv5x5_2(self.conv5x5_1(x))),self.pool_2(self.pool_1(x))), 1)
			else:
				y = torch.cat((self.conv1x1(x), self.conv3x3_3(self.conv3x3_2(self.conv3x3_1(x))), self.conv5x5_3(self.conv5x5_2(self.conv5x5_1(x))),self.pool_2(self.pool_1(x))), 1)
		else:
			y1 = self.conv1x1(x)
			if self.with_modulated_dcn:
				y2 = self.conv3x3_1(x)
				offset_mask2 = self.conv3x3_dcn_offset(y2)
				offset2 = offset_mask2[:, :18, :, :]
				mask2 = offset_mask2[:, -9:, :, :].sigmoid()
				y2 = self.bn3x3_relu(self.bn3x3_bn(self.conv3x3_dcn(y2, offset2, mask2)))

				y3 = self.conv5x5_1(x)
				offset_mask3_1 = self.conv5x5_dcn_offset_1(y3)
				offset3_1 = offset_mask3_1[:, :18, :, :]
				mask3_1 = offset_mask3_1[:, -9:, :, :].sigmoid()
				y3 = self.bn5x5_relu_1(self.bn5x5_bn_1(self.conv5x5_dcn_1(y3, offset3_1, mask3_1)))

				offset_mask3_2 = self.conv5x5_dcn_offset_2(y3)
				offset3_2 = offset_mask3_2[:, :18, :, :]
				mask3_2 = offset_mask3_2[:, -9:, :, :].sigmoid()
				y3 = self.bn5x5_relu_2(self.bn5x5_bn_2(self.conv5x5_dcn_2(y3, offset3_2, mask3_2)))

			else:
				y2 = self.conv3x3_1(x)
				y2 = self.bn3x3_relu(self.bn3x3_bn(self.conv3x3_dcn(y2, self.conv3x3_dcn_offset(y2))))
				y3 = self.conv5x5_1(x)
				y3 = self.bn5x5_relu_1(self.bn5x5_bn_1(self.conv5x5_dcn_1(y3, self.conv5x5_dcn_offset_1(y3))))
				y3 = self.bn5x5_relu_2(self.bn5x5_bn_2(self.conv5x5_dcn_2(y3, self.conv5x5_dcn_offset_2(y3))))

			y4 = self.pool_2(self.pool_1(x))
			y = torch.cat((y1, y2, y3, y4), 1)

		y = self.outlayer(y)
		return y + self.skip_layer(x)

class PVANet(nn.Module):
	def __init__(self, version = 1,  conv1_5 = True ,acb_block =False, dcn =False,with_modulated_dcn=False):#conv1_5如果是true表示是原来的，如果是128表示是已经减少了最后一大层
		super(PVANet,self).__init__()
		self.pre = Conv2dBNCReLU(3, 16*version, kernel_size=7, stride=2, padding=3)
		self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)#注意这里的padding的设置

		self.layer1 = Conv2d_1_3_1(16*2*version, mid_channels=24*version, out_channels=64*version, stride = 1)
		self.layer2 = Conv2d_1_3_1(64*version, mid_channels=24*version, out_channels=64*version, stride = 1)
		self.layer3 = Conv2d_1_3_1(64*version, mid_channels=24*version, out_channels=64*version, stride = 1)#
		self.layer4 = Conv2d_1_3_1(64*version, mid_channels=48*version, out_channels=128*version, stride = 2,dcn =dcn)
  
		self.layer5 = Conv2d_1_3_1(128*version, mid_channels=48*version, out_channels=128*version, stride = 1,dcn =dcn)
		self.layer6 = Conv2d_1_3_1(128*version, mid_channels=48*version, out_channels=128*version, stride = 1,dcn =dcn)
		self.layer7 = Conv2d_1_3_1(128*version, mid_channels=48*version, out_channels=128*version, stride = 1,dcn =dcn)#
		self.layer8 = Inception_PVA_v2(128*version, 48*version, 128*version, 24*version, 48*version, out_channels=256*version, acb_block = False, dcn =dcn,with_modulated_dcn=with_modulated_dcn)
  
		self.layer9 = Inception_PVA_v1( 256*version, 64*version, 128*version, 24*version, 48*version, out_channels=256*version, acb_block = False, dcn =dcn,with_modulated_dcn=with_modulated_dcn)
		self.layer10 = Inception_PVA_v1( 256*version, 64*version, 128*version, 24*version, 48*version, out_channels=256*version, acb_block = acb_block, dcn =dcn,with_modulated_dcn=with_modulated_dcn)
		self.layer11 = Inception_PVA_v1( 256*version, 64*version, 128*version, 24*version, 48*version, out_channels=256*version, acb_block = acb_block, dcn =dcn,with_modulated_dcn=with_modulated_dcn)

		self.conv1_5 = conv1_5
		if self.conv1_5:
			self.layer12 = Inception_PVA_v2( 256*version, 96*version, 192*version, 32*version, 64*version, out_channels=384*version, acb_block = acb_block, dcn =dcn,with_modulated_dcn=with_modulated_dcn)
			self.layer13 = Inception_PVA_v1( 384*version, 96*version, 192*version, 32*version, 64*version, out_channels=384*version, acb_block = acb_block, dcn =dcn,with_modulated_dcn=with_modulated_dcn)
			self.layer14 = Inception_PVA_v1( 384*version, 96*version, 192*version, 32*version, 64*version, out_channels=384*version, acb_block = acb_block, dcn =dcn,with_modulated_dcn=with_modulated_dcn)
			self.layer15 = Inception_PVA_v1( 384*version, 96*version, 192*version, 32*version, 64*version, out_channels=384*version, acb_block = acb_block, dcn =dcn,with_modulated_dcn=with_modulated_dcn)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
		for m in self.modules():
			if hasattr(m, 'conv3x3_dcn_offset'):
				constant_init(m.conv3x3_dcn_offset, 0)
			if hasattr(m, 'conv5x5_dcn_offset_1'):
				constant_init(m.conv5x5_dcn_offset_1, 0)
			if hasattr(m, 'conv5x5_dcn_offset_2'):
				constant_init(m.conv5x5_dcn_offset_2, 0)

	def forward(self,x):
		y = self.pool(self.pre(x)) #cReLU
		print(y.size())
		y = self.layer7(self.layer6(self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(y)))))))
		if self.conv1_5:
			y = self.layer15(self.layer14(self.layer13(self.layer12(self.layer11(self.layer10(self.layer9(self.layer8(y))))))))
		else:
			y = self.layer11(self.layer10(self.layer9(self.layer8(y))))
		return y

class extractor_PVANet(nn.Module):
	def __init__(self,version=1,conv1_5=True, acb_block =False,dcn =False,with_modulated_dcn=False):
		super(extractor_PVANet, self).__init__()
		self.features = PVANet(version = version,conv1_5 = conv1_5, acb_block = acb_block,dcn =dcn,with_modulated_dcn=with_modulated_dcn)

	def forward(self, x):
		out = []
		for name, module in self.features.named_children():
			x = module(x)
			# print(name)
			if name in ['layer3', 'layer7','layer11','layer15']:##########3['layer3', 'layer7','layer11','layer15']
				out.append(x)
		return out

#inception_mid表示merge的卷积是否是inception，inception_end表示最后的一个是3*3*32还是inception
class EAST_PVANet(nn.Module):
	def __init__(self,inception_mid = False,inception_end=False, version = 1,conv1_5 = True, acb_block =False,dcn =False,with_modulated_dcn=False):
		super(EAST_PVANet,self).__init__()
		self.extractor = extractor_PVANet(version=version,conv1_5 = conv1_5, acb_block = acb_block,dcn =dcn,with_modulated_dcn=with_modulated_dcn)
		
		self.merge = merge(inception=inception_end,conv1_5 = conv1_5, two = 64*version, thr = 128*version, four = 256*version, five = 384*version)

		self.out = output(320)


	def forward(self, x):
		return self.out(self.merge(self.extractor(x)))

from efficientnet_pytorch import EfficientNet


class extractor_EfficientNet(nn.Module):
	def __init__(self):
		super(extractor_EfficientNet, self).__init__()
		self.features =  EfficientNet.from_pretrained('efficientnet-b0')
		
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
	

	def forward(self, x):
		out = []
		for name, module in self.features.named_children():
			# print(name)
			if name == '_blocks':
				for name, module1 in module.named_children():
					x = module1(x)
					if name in ['2', '4', '10', '15']:
						#print(x.size())
						out.append(x)
			else:
				x = module(x)
	
		return out

class EAST_EfficientNet(nn.Module):
	def __init__(self,inception=False):
		super(EAST_EfficientNet,self).__init__()
		self.extractor = extractor_EfficientNet()
		self.merge = merge(inception=inception,conv1_5=True, two = 24, thr = 40, four = 112, five = 320)
		self.out = output(512)

	def forward(self, x):
		y = self.extractor(x)
		y = self.merge(y)
		return self.out(y)


if __name__ == '__main__':
	# writer = SummaryWriter(log_dir= "./visual_weights")

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model=EAST_PVANet(inception_mid = False,inception_end=True, version = 1,conv1_5 = False, acb_block =False,dcn =True,with_modulated_dcn=False).to(device)
	#model = torchvision.models.vgg16_bn().features.to(device)
	#model = torchvision.models.resnet50().to(device)
	# model = EfficientNet.from_pretrained('efficientnet-b0').to(device)
	# out = []
	# for name, module in model.named_children():
	# 	print(name)
	# 	if name in ['_blocks']:
	# 		for name, module1 in module.named_children():
	# 			print(name)
	# 			if name in ['2', '4', '10', '15']:
	# 				out.append(module1)
 
	# print(model)
				
	#summary(model,(3, 224, 224))
	#summary(m,(3, 224, 224))
	# model = torchvision.models.resnet18()
	# hl.build_graph(model, torch.zeros([1, 3,  224, 224]))
	# x = Variable(torch.randn(1,3, 256, 256))
	# vis_graph = make_dot(model(x), params=dict(model.named_parameters()))
	# vis_graph.view()
	# writer.add_graph(model,x)
	# writer.close()
	# score, geo = m(x)
	# print(score.shape)
	# print(geo.shape)
