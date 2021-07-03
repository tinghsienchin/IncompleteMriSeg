import torch
import torch.nn as nn
import torch.nn.functional as F

in_place = True

def conv_block(feat_in, feat_out, kernel = 3, padding = 1, strides = 1, bias = False):

	return nn.Conv3d(feat_in, feat_out, kernel_size= kernel, stride= strides, padding= padding, bias= bias)


class NoBottleneck(nn.Module):
	"""docstring for NoBottleneck"""
	def __init__(self, feat_in, feat_out, strides = 1, downsample = None):
		super(NoBottleneck, self).__init__()
		self.feat_in = feat_in
		self.feat_out = feat_out
		self.strides = strides
		self.Norm1 = nn.GroupNorm(4,feat_in)
		self.conv1 = conv_block(self.feat_in, self.feat_out, kernel = 3, strides = self.strides, padding = 1)
		self.relu = nn.ReLU(inplace = in_place)

		self.Norm2 = nn.GroupNorm(4,feat_out)
		self.conv2 = conv_block(self.feat_out, self.feat_out, kernel = 3, strides = 1, padding = 1)

		self.downsample = downsample

	def forward(self, x):
		residual = x

		out = self.Norm1(x)
		out = self.relu(out)
		out = self.conv1(out)

		out = self.Norm2(out)
		out = self.relu(out)
		out = self.conv2(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out = out + residual

		return out
		



class ResUNet(nn.Module):
	"""
	Implementations based on the Unet3D paper: https://arxiv.org/abs/1606.06650
	"""
	def __init__(self, in_channels, n_classes, base_n_filter, layers, training = True):
		super(ResUNet, self).__init__()

		self.in_channels = in_channels
		self.n_classes = n_classes
		self.base_n_filter = base_n_filter
		self.training = training
		self.conv1 = conv_block(self.in_channels, self.base_n_filter)

		self.layer0 = self._make_layer(NoBottleneck, self.base_n_filter, self.base_n_filter, layers[0], strides = 1)
		self.layer1 = self._make_layer(NoBottleneck, self.base_n_filter, self.base_n_filter * 2, layers[1], strides = 2)
		self.layer2 = self._make_layer(NoBottleneck, self.base_n_filter * 2, self.base_n_filter * 4, layers[2], strides = 2)
		self.layer3 = self._make_layer(NoBottleneck, self.base_n_filter * 4, self.base_n_filter * 8, layers[3], strides = 2)
		self.layer4 = self._make_layer(NoBottleneck, self.base_n_filter * 8, self.base_n_filter * 8, layers[4], strides = 2)

		self.fusionConv = nn.Sequential(
			nn.GroupNorm(4, self.base_n_filter * 8),
			nn.ReLU(inplace = in_place),
			conv_block(self.base_n_filter * 8, self.base_n_filter * 8, kernel = 1, padding = 0)
			)
		self.upsamplex2 = nn.Upsample(scale_factor=2, mode='trilinear')

		self.x8_decode = self.decoder_conv(self.base_n_filter * 8)
		self.x4_decode = self.decoder_conv(self.base_n_filter * 4)
		self.x2_decode = self.decoder_conv(self.base_n_filter * 2)
		self.x1_decode = self.decoder_conv(self.base_n_filter)  

		self.x8_resb = self._make_layer(NoBottleneck, self.base_n_filter * 8, self.base_n_filter * 4, 1, strides=(1, 1, 1))
		self.x4_resb = self._make_layer(NoBottleneck, self.base_n_filter * 4, self.base_n_filter * 2, 1, strides=(1, 1, 1))
		self.x2_resb = self._make_layer(NoBottleneck, self.base_n_filter * 2, self.base_n_filter, 1, strides=(1, 1, 1))
		self.x1_resb = self._make_layer(NoBottleneck, self.base_n_filter, self.base_n_filter, 1, strides=(1, 1, 1))

		self.segconv = conv_block(self.base_n_filter, self.n_classes)

	def _make_layer(self, block, feat_in, feat_out, blocks, strides = 1):
		downsample = None
		if strides != 1 or feat_in != feat_out:
			downsample = nn.Sequential(
				nn.GroupNorm(4, feat_in), 
				nn.ReLU(inplace= in_place), 
				conv_block(feat_in, feat_out, kernel = 1, strides = strides, padding = 0)
				)

		layers = []
		layers.append(block(feat_in, feat_out, strides ,downsample = downsample))

		for i in range(1, blocks):
			layers.append(block(feat_out, feat_out))

		return nn.Sequential(*layers)


	def decoder_conv(self, feat_in, kernel = 2, strides = 2, padding= 0):

		return nn.ConvTranspose3d(feat_in, feat_in, kernel_size= kernel, stride= strides, padding= padding)


	def forward(self, input):
		x = self.conv1(input)
		x = self.layer0(x)
		skip0 = x

		x = self.layer1(x)
		skip1 = x

		x = self.layer2(x)
		skip2 = x

		x = self.layer3(x)
		skip3 = x

		x = self.layer4(x)
		x = self.fusionConv(x)

		x = self.upsamplex2(x)
		x = x + skip3
		x = self.x8_resb(x)

		x = self.upsamplex2(x)
		x = x + skip2
		x = self.x4_resb(x)

		x = self.upsamplex2(x)
		x = x + skip1
		x = self.x2_resb(x)

		x = self.upsamplex2(x)
		x = x + skip0
		x = self.x1_resb(x)

		x = self.segconv(x)

		return x






		
