import torch
import torch.nn as nn
import numpy as np
from models.transformer.PositionalEncoding import modalty_missing

def conv_block(feat_in, feat_out, kernel = 3, padding = 1, strides = 1, bias = False):

	return nn.Conv3d(feat_in, feat_out, kernel_size= kernel, stride= strides, padding= padding, bias= bias)


class decode(nn.Module):
	"""docstring for decode"""
	def __init__(self, feat_in, feat_out):
		super(decode, self).__init__()
		self.feat_in = feat_in
		self.feat_out = feat_out
		self.decode_conv = nn.ConvTranspose3d(self.feat_in, self.feat_out, kernel_size=2, stride=2)

		for i in range(1 , 3):
			conv = nn.Sequential(
										conv_block(self.feat_in, self.feat_out),
										nn.InstanceNorm3d(self.feat_out),
										nn.LeakyReLU(inplace = True)
									  )
			setattr(self, 'conv%d' % i, conv)
			self.feat_in = self.feat_out


	def forward(self, layer, *layers):


		x = self.decode_conv(layer)
		
		for i in range(len(layers)):
			x = torch.cat([x, layers[i]], 1)   #for unet+++

		for i in range(1, 3):
			conv = getattr(self, 'conv%d' % i)
			x = conv(x)

		return x

class encode(nn.Module):
	"""docstring for encode"""
	def __init__(self, feat_in, feat_out , pooling = True):
		super(encode, self).__init__()
		self.feat_in = feat_in
		self.feat_out = feat_out
		self.id_conv = nn.Sequential(
										conv_block(self.feat_in, self.feat_out),
										nn.InstanceNorm3d(self.feat_out)
									  )


		for i in range(1 , 3):
			conv = nn.Sequential(
										conv_block(self.feat_in, self.feat_out, kernel = 1, padding = 0),
										nn.InstanceNorm3d(self.feat_out),
										nn.LeakyReLU(inplace = True)
									  )
			setattr(self, 'conv%d' % i, conv)
			self.feat_in = self.feat_out


		for j in range(1,3):
			conv1 = nn.Sequential(
										conv_block(self.feat_in, self.feat_out),
										nn.InstanceNorm3d(self.feat_out),
										nn.LeakyReLU(inplace = True)
							          )
			setattr(self, 'conv1%d' % j, conv1)

		self.pooling = pooling
		self.pool = nn.MaxPool3d(kernel_size = 2)
		self.relu = nn.LeakyReLU(inplace = True)

	def forward(self, input):

		x = input
		
		identity = input
		identity = self.id_conv(identity)

		#res1 不是很有用
		for i in range(1, 3):
			conv = getattr(self, 'conv%d' % i)
			x = conv(x)
		x = self.relu(x + identity)

		#res2

		identity = x
		for j in range(1,3):
			conv1 = getattr(self, 'conv1%d' % j)
			x = conv1(x)

		context = self.relu(x + identity)

		if self.pooling:

			x = self.relu(x + identity)
			x = self.pool(x)
			return x, context
		else:
			context = x
			return context


class cSE(nn.Module):
    def __init__(self, in_channels, reduction = 2):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.Conv_Squeeze = conv_block(in_channels, in_channels // reduction, kernel=1, padding = 0)
        self.Conv_Excitation = conv_block(in_channels // reduction, in_channels, kernel=1, padding= 0)
        self.norm = nn.Softmax(dim = -1)

    def forward(self, input):
    	x = input
    	N, C, H, W, D = x.size()
    	query = x.view(N, C, -1)
    	key = query.permute(0, 2, 1)
    	energy = torch.bmm(query, key)
    	energy_new = torch.max(energy, -1, keepdim = True)
    	energy_new = energy_new[0].expand_as(energy)
    	energy_new = energy_new - energy
    	attention = self.norm(energy_new)

    	value = x.view(N, C, -1)

    	out = torch.bmm(attention, value)
    	out = out.view(N, C, H, W, D)
    	return out

class sSE(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.query_conv = conv_block(in_dim,
        						     in_dim // 8,
        						     kernel = 1,
        						     padding = 0
        						    )
        self.key_conv = conv_block(in_dim,
        						   in_dim // 8,
        						   kernel = 1,
        						   padding = 0
        						    )
        self.value_conv = conv_block(in_dim,
        						     in_dim,
        						     kernel = 1,
        						     padding = 0
        						    )
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim = -1)
        self.id_conv = conv_block(in_dim, in_dim // 2)


    def forward(self, x):
    	print(x.size())
    	m_batchsize, C, height, width, depth = x.size()
    	proj_query = self.query_conv(x).view(m_batchsize, -1, width * height * depth)
    	proj_query = proj_query.permute(0, 2, 1)
    	proj_key = self.key_conv(x).view(m_batchsize, -1, width * height * depth)
    	energy = torch.bmm(proj_query, proj_key)
    	attention = self.softmax(energy)
    	proj_value = self.value_conv(x).view(m_batchsize, -1, width * height * depth)
    	out = torch.bmm(proj_value, attention.permute(0, 2, 1))
    	out = out.view(m_batchsize, C, height, width, depth)
    	out = self.gamma * out + x
    	out = self.id_conv(out)
    	return out


class Shared_feature(nn.Module):
	"""docstring for Shared_feature"""
	def __init__(self, feat_in, feat_out , reduction, pooling = True):
		super(Shared_feature, self).__init__()
		self.feat_in = feat_in
		self.feat_out = feat_out
		self.reduction = reduction

		for i in range(1 , 3):
			conv = nn.Sequential(
										conv_block(self.feat_in, self.feat_in),
										nn.InstanceNorm3d(self.feat_in),
										nn.LeakyReLU(inplace = True)
									  )
			setattr(self, 'conv%d' % i, conv)
			# self.feat_in = self.feat_out

		self.pooling = pooling
		self.pool = nn.MaxPool3d(kernel_size = 2)
		self.cSE = cSE(self.feat_in, self.reduction)
		# self.sSE = sSE()
		self.relu = nn.LeakyReLU(inplace = True)
		self.gamma = nn.Parameter(torch.zeros(1))
		self.id_conv = conv_block(self.feat_in, self.feat_out)

	def forward(self, input):

		x = input
		attention_map = self.cSE(x)

		for i in range(1, 3):
			conv = getattr(self, 'conv%d' % i)
			x = conv(x)

		x = self.gamma * attention_map + x
		x = self.id_conv(x)
		print(self.gamma)
		# x = self.relu(x + residual)

		# x_mat = self.cSE(x)

		# x = self.relu((1 + x_mat) * x)
		# x = (1 + x_mat) * x

		if self.pooling:

			x = self.pool(x)
			return x
		else:
			return x

class Share_combine2(nn.Module):
	"""docstring for Share_combine2"""
	def __init__(self, feat_in, feat_out, pooling = False):
		super(Share_combine2, self).__init__()
		self.feat_in = feat_in
		self.feat_out = feat_out
		self.id_conv = nn.Sequential(
			                            conv_block(self.feat_in, self.feat_out),
			                            nn.InstanceNorm3d(self.feat_out)
			                        )
		for i in range(1 , 3):
			conv = nn.Sequential(
				                        conv_block(self.feat_out, self.feat_out),
				                        nn.InstanceNorm3d(self.feat_out),
				                        nn.LeakyReLU(inplace = True)
				                )
			setattr(self, 'conv%d' % i, conv)

		self.relu = nn.LeakyReLU(inplace = True)
		self.pooling = pooling
		self.pool = nn.MaxPool3d(kernel_size = 2)
		# self.sSE = sSE()
		self.alpha = nn.Parameter(torch.zeros(1))

	def forward(self, share_up, share_now):

		x = share_now
		x_ori_up = share_up

		######上一层加这一层#########
		x_ori_up = self.id_conv(x_ori_up)
		x_ori_up = self.pool(x_ori_up)

		x = self.relu(x + x_ori_up)
		
		for i in range(1, 3):
			conv = getattr(self, 'conv%d' % i)
			x = conv(x)

		# x = self.relu(residual + x)

		# x_mat = self.sSE(x)

		# x = self.relu((1 + x_mat) * x)

		return x,x_ori_up

class Share_combine3(nn.Module):
	"""docstring for Share_combine3"""
	def __init__(self, feat_in, feat_out, pooling = False):
		super(Share_combine3, self).__init__()
		self.feat_in = feat_in
		self.feat_out = feat_out
		self.id_conv = nn.Sequential(
			                            conv_block(self.feat_in, self.feat_out),
			                            nn.InstanceNorm3d(self.feat_out)
			                        )
		for i in range(1 , 3):
			conv = nn.Sequential(
				                        conv_block(self.feat_out, self.feat_out),
				                        nn.InstanceNorm3d(self.feat_out),
				                        nn.LeakyReLU(inplace = True)
				                )
			setattr(self, 'conv%d' % i, conv)

		self.relu = nn.LeakyReLU(inplace = True)
		self.pooling = pooling
		self.pool = nn.MaxPool3d(kernel_size = 2)
		# self.sSE = sSE()
		self.alpha = nn.Parameter(torch.zeros(1))

	def forward(self, share_up, share_now, share_upp):

		x = share_now
		x_ori_up = share_up
		x_upp = share_upp

		######上一层加这一层#########
		x_ori_up = self.id_conv(x_ori_up)
		x_ori_up = self.pool(x_ori_up)

		x_upp = self.id_conv(x_upp)
		x_ori_upp = self.pool(x_upp)

		x = self.relu(x + x_ori_up + x_ori_upp)
		
		
		for i in range(1, 3):
			conv = getattr(self, 'conv%d' % i)
			x = conv(x)

		# x = self.relu(x + residual)

		# x_mat = self.sSE(x)

		# x = self.relu((1 + x_mat) * x)

		return x,x_ori_up,x_ori_upp

class Share_combine4(nn.Module):
	"""docstring for Share_combine4"""
	def __init__(self, feat_in, feat_out, pooling = False):
		super(Share_combine4, self).__init__()
		self.feat_in = feat_in
		self.feat_out = feat_out
		self.id_conv = nn.Sequential(
			                            conv_block(self.feat_in, self.feat_out),
			                            nn.InstanceNorm3d(self.feat_out)
			                        )
		for i in range(1 , 3):
			conv = nn.Sequential(
				                        conv_block(self.feat_out, self.feat_out),
				                        nn.InstanceNorm3d(self.feat_out),
				                        nn.LeakyReLU(inplace = True)
				                )
			setattr(self, 'conv%d' % i, conv)

		self.relu = nn.LeakyReLU(inplace = True)
		self.pooling = pooling
		self.pool = nn.MaxPool3d(kernel_size = 2)
		# self.sSE = sSE()
		self.alpha = nn.Parameter(torch.zeros(1))

	def forward(self, share_up, share_now, share_upp, share_uppp):

		x = share_now
		x_ori_up = share_up
		x_upp = share_upp
		x_uppp = share_uppp

		######上一层加这一层#########
		x_ori_up = self.id_conv(x_ori_up)
		x_ori_up = self.pool(x_ori_up)

		x_upp = self.id_conv(x_upp)
		x_ori_upp = self.pool(x_upp)

		x_uppp = self.id_conv(x_uppp)
		x_ori_uppp = self.pool(x_uppp)

		x = self.relu(x + x_ori_up + x_ori_upp + x_ori_uppp)


		for i in range(1, 3):
			conv = getattr(self, 'conv%d' % i)
			x = conv(x)

		if self.pooling:

			x = self.pool(x)
			return x
		else:
			return x

class UNet(nn.Module):
	"""docstring for UNet"""
	def __init__(self, in_channels, n_classes, base_n_filter, multi_TR = True):
		super(UNet, self).__init__()
		self.in_channels = in_channels
		self.n_classes = n_classes
		self.multi_TR = multi_TR
		self.filters = [base_n_filter, base_n_filter * 2, base_n_filter * 4, base_n_filter * 8, base_n_filter * 16]

		self.relu = nn.LeakyReLU(inplace = True)

		# self.fuse = Cross_modality(self.in_channels,self.in_channels)

		self.down1_1 = encode(self.in_channels // 4, self.filters[0] // 4)
		self.down1_2 = encode(self.in_channels // 4, self.filters[0] // 4)
		self.down1_3 = encode(self.in_channels // 4, self.filters[0] // 4)
		self.down1_4 = encode(self.in_channels // 4, self.filters[0] // 4)

		self.shared_conv1 = Shared_feature(self.filters[0], self.filters[1], 2)

		self.down2_1 = encode(self.filters[0] // 4, self.filters[1] // 4)
		self.down2_2 = encode(self.filters[0] // 4, self.filters[1] // 4)
		self.down2_3 = encode(self.filters[0] // 4, self.filters[1] // 4)
		self.down2_4 = encode(self.filters[0] // 4, self.filters[1] // 4)
		self.shared_conv2 = Shared_feature(self.filters[1], self.filters[2], 4)
		# self.combine2 = Share_combine2(self.filters[0], self.filters[1])

		self.down3_1 = encode(self.filters[1] // 4, self.filters[2] // 4)
		self.down3_2 = encode(self.filters[1] // 4, self.filters[2] // 4)
		self.down3_3 = encode(self.filters[1] // 4, self.filters[2] // 4)
		self.down3_4 = encode(self.filters[1] // 4, self.filters[2] // 4)
		self.shared_conv3 = Shared_feature(self.filters[2], self.filters[3], 4)
		# self.combine3 = Share_combine3(self.filters[1], self.filters[2])

		self.down4_1 = encode(self.filters[2] // 4, self.filters[3] // 4)
		self.down4_2 = encode(self.filters[2] // 4, self.filters[3] // 4)
		self.down4_3 = encode(self.filters[2] // 4, self.filters[3] // 4)
		self.down4_4 = encode(self.filters[2] // 4, self.filters[3] // 4)
		self.shared_conv4 = Shared_feature(self.filters[3], self.filters[3], 4)
		# self.combine4 = Share_combine4(self.filters[2], self.filters[3] , True)

		self.fusion = sSE(self.filters[3] * 2)
		self.bottom = encode(self.filters[3], self.filters[4], False)
		self.bottom_modal1 = encode(self.filters[3] // 4, self.filters[4] // 4, False)
		self.bottom_modal2 = encode(self.filters[3] // 4, self.filters[4] // 4, False)
		self.bottom_modal3 = encode(self.filters[3] // 4, self.filters[4] // 4, False)
		self.bottom_modal4 = encode(self.filters[3] // 4, self.filters[4] // 4, False)

		self.up4 = decode(self.filters[4], self.filters[3])
		self.up4_modal1 = decode(self.filters[4] // 4, self.filters[3] // 4)
		self.up4_modal2 = decode(self.filters[4] // 4, self.filters[3] // 4)
		self.up4_modal3 = decode(self.filters[4] // 4, self.filters[3] // 4)
		self.up4_modal4 = decode(self.filters[4] // 4, self.filters[3] // 4)

		self.up3 = decode(self.filters[3], self.filters[2])
		self.up3_modal1 = decode(self.filters[3] // 4, self.filters[2] // 4)
		self.up3_modal2 = decode(self.filters[3] // 4, self.filters[2] // 4)
		self.up3_modal3 = decode(self.filters[3] // 4, self.filters[2] // 4)
		self.up3_modal4 = decode(self.filters[3] // 4, self.filters[2] // 4)

		self.up2 = decode(self.filters[2], self.filters[1])
		self.up2_modal1 = decode(self.filters[2] // 4, self.filters[1] // 4)
		self.up2_modal2 = decode(self.filters[2] // 4, self.filters[1] // 4)
		self.up2_modal3 = decode(self.filters[2] // 4, self.filters[1] // 4)
		self.up2_modal4 = decode(self.filters[2] // 4, self.filters[1] // 4)

		self.up1 = decode(self.filters[1], self.filters[0])
		self.up1_modal1 = decode(self.filters[1] // 4, self.filters[0] // 4)
		self.up1_modal2 = decode(self.filters[1] // 4, self.filters[0] // 4)
		self.up1_modal3 = decode(self.filters[1] // 4, self.filters[0] // 4)
		self.up1_modal4 = decode(self.filters[1] // 4, self.filters[0] // 4)

		self.segconv = conv_block(self.filters[0], self.n_classes)
		self.segconv_modal1 = conv_block(self.filters[0] // 4, self.n_classes)
		self.segconv_modal2 = conv_block(self.filters[0] // 4, self.n_classes)
		self.segconv_modal3 = conv_block(self.filters[0] // 4, self.n_classes)
		self.segconv_modal4 = conv_block(self.filters[0] // 4, self.n_classes)

	def forward(self, input):
		index = np.random.randint(0,15,size=1)
		# index = [13]
		print(index)

		x1 = input[:,0,:,:,:]
		x2 = input[:,1,:,:,:]
		x3 = input[:,2,:,:,:]
		x4 = input[:,3,:,:,:]

		layer1_1, context1_1 = self.down1_1(x1.unsqueeze(0))

		layer1_2, context1_2 = self.down1_2(x2.unsqueeze(0))

		layer1_3, context1_3 = self.down1_3(x3.unsqueeze(0))

		layer1_4, context1_4 = self.down1_4(x4.unsqueeze(0))

		context1, context1_ori = modalty_missing(index, context1_1,context1_2,context1_3,context1_4)
		# context1_share_ori = self.shared_conv1(context1_ori)#共享特征
		context1_share = self.shared_conv1(context1)#共享特征
		print(context1_share.shape)

		layer2_1, context2_1 = self.down2_1(layer1_1)
		
		layer2_2, context2_2 = self.down2_2(layer1_2)
		
		layer2_3, context2_3 = self.down2_3(layer1_3)
		
		layer2_4, context2_4 = self.down2_4(layer1_4)

		context2, context2_ori = modalty_missing(index, context2_1,context2_2,context2_3,context2_4)
		context2 = self.relu(context2 + context1_share)
		# context2_ori = self.relu(context2_ori + context1_share_ori)
		# context2_share_ori = self.shared_conv2(context2_ori)
		context2_share = self.shared_conv2(context2)
		# context2_share, context1_new = self.combine2(context1_share, context2_share)
		print(context2_share.shape)

		layer3_1, context3_1 = self.down3_1(layer2_1)

		layer3_2, context3_2 = self.down3_2(layer2_2)
		
		layer3_3, context3_3 = self.down3_3(layer2_3)
		
		layer3_4, context3_4 = self.down3_4(layer2_4)

		context3, context3_ori = modalty_missing(index, context3_1,context3_2,context3_3,context3_4)
		context3 = self.relu(context3 + context2_share)
		# context3_ori = self.relu(context3_ori + context2_share_ori)
		# context3_share_ori = self.shared_conv3(context3_ori)
		context3_share = self.shared_conv3(context3)
		# context3_share, context2_new, context1_new = self.combine3(context2_share, context3_share, context1_new)
		print(context3_share.shape)

		layer4_1, context4_1 = self.down4_1(layer3_1)

		layer4_2, context4_2 = self.down4_2(layer3_2)

		layer4_3, context4_3 = self.down4_3(layer3_3)

		layer4_4, context4_4 = self.down4_4(layer3_4)

		context4, context4_ori = modalty_missing(index, context4_1,context4_2,context4_3,context4_4)
		context4 = self.relu(context4 + context3_share)
		# context4_ori = self.relu(context4_ori + context3_share_ori)
		# context4_share_ori = self.shared_conv4(context4_ori)
		context4_share = self.shared_conv4(context4) # [16,16,16] -- [8,8,8]
		# context4_share = self.combine4(context3_share, context4_share, context2_new, context1_new)
		print(context4_share.shape)



		context4_specific, context4_specific_ori = modalty_missing(layer4_1,layer4_2,layer4_3,layer4_4)
		layer4 = torch.cat((context4_specific, context4_share),1)
		# layer4_ori = torch.cat((context4_specific_ori, context4_share_ori),1)

		# layer1, context1 = self.down1(input)

		# layer2, context2 = self.down2(layer1)

		# layer3, context3 = self.down3(layer2)

		# layer4, context4 = self.down4(layer3)
		layer4 = self.fusion(layer4)
		layer_bottom = self.bottom(layer4)
		if self.multi_TR:
			layer_bottom1 = self.bottom_modal1(layer4_1)
			layer_bottom2 = self.bottom_modal2(layer4_2)
			layer_bottom3 = self.bottom_modal3(layer4_3)
			layer_bottom4 = self.bottom_modal4(layer4_4)

		up_layer4 = self.up4(layer_bottom, context4)

		if self.multi_TR:
			up_layer4_1 = self.up4_modal1(layer_bottom1, context4_1)
			up_layer4_2 = self.up4_modal2(layer_bottom2, context4_2)
			up_layer4_3 = self.up4_modal3(layer_bottom3, context4_3)
			up_layer4_4 = self.up4_modal4(layer_bottom4, context4_4)

		up_layer3 = self.up3(up_layer4, context3)
		if self.multi_TR:
			up_layer3_1 = self.up3_modal1(up_layer4_1, context3_1)
			up_layer3_2 = self.up3_modal2(up_layer4_2, context3_2)
			up_layer3_3 = self.up3_modal3(up_layer4_3, context3_3)
			up_layer3_4 = self.up3_modal4(up_layer4_4, context3_4)


		up_layer2 = self.up2(up_layer3, context2)
		if self.multi_TR:
			up_layer2_1 = self.up2_modal1(up_layer3_1, context2_1)
			up_layer2_2 = self.up2_modal2(up_layer3_2, context2_2)
			up_layer2_3 = self.up2_modal3(up_layer3_3, context2_3)
			up_layer2_4 = self.up2_modal4(up_layer3_4, context2_4)


		up_layer1 = self.up1(up_layer2, context1)
		if self.multi_TR:
			up_layer1_1 = self.up1_modal1(up_layer2_1, context1_1)
			up_layer1_2 = self.up1_modal2(up_layer2_2, context1_2)
			up_layer1_3 = self.up1_modal3(up_layer2_3, context1_3)
			up_layer1_4 = self.up1_modal4(up_layer2_4, context1_4)

		out = self.segconv(up_layer1)
		if self.multi_TR:
			out1 = self.segconv_modal1(up_layer1_1)
			out2 = self.segconv_modal2(up_layer1_2)
			out3 = self.segconv_modal3(up_layer1_3)
			out4 = self.segconv_modal4(up_layer1_4)
			return [out1, out2, out3, out4, out]
		else:
			return out
