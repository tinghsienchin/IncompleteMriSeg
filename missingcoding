import torch
import torch.nn as nn


def conv_block(feat_in, feat_out, kernel = 3, padding = 1, strides = 1, bias = False):

	return nn.Conv3d(feat_in, feat_out, kernel_size= kernel, stride= strides, padding= padding, bias= bias)

class FixedMissingEncoding(nn.Module):
	"""docstring for FixedPositionalEncoding"""
	def __init__(self, embedding_dim, max_length = 15):
		super(FixedPositionalEncoding, self).__init__()

		pe = torch.zeros(max_length, embedding_dim)
		position = torch.arange(0, max_length, dtype = torch.float).unsqueeze(1)
		div_term = torch.exp(
			torch.arange(0, embedding_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embedding_dim)
		)

		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0).transpose(0,1)

		self.register_buffer('pe', pe)

	def forward(self, x, index):
		x = x + self.pe[index, :].unsqueeze(0)
		return x


class LearnedPositionalEncoding(nn.Module):
	def __init__(self, max_position_embeddings, embedding_dim, seq_length):
		super(LearnedPositionalEncoding, self).__init__()

		self.position_embeddings = nn.Parameter(torch.zeros(1, 256, 8, 8, 8)) #16^3 , 512


	def forward(self, x , position_ids = None):

		position_embeddings = self.position_embeddings

		return x + position_embeddings

class modalty_missing(nn.Module):
	def __init__(self, in_dim):
		super(modalty_missing, self).__init__()
		modality_missing_list = [ [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0],
		                          [0, 1, 0, 1], [0, 1, 1, 0], [1, 0, 1, 0], [0, 0, 1, 1],
		                          [1, 0, 0, 1], [1, 1, 0, 0], [1, 1, 1, 0], [1, 0, 1, 1],
		                          [1, 1, 0, 1], [0, 1, 1, 1], [1, 1, 1, 1]
		                        ]
		
		self.in_dim = in_dim
		# self.position_encoding0 = FixedPositionalEncoding(context)
		# self.position_encoding1 = FixedPositionalEncoding(context)
		# self.position_encoding2 = FixedPositionalEncoding(context)
		# self.position_encoding3 = FixedPositionalEncoding(context)
		# self.position_encoding4 = FixedPositionalEncoding(context)
		# self.position_encoding5 = FixedPositionalEncoding(context)
		# self.position_encoding6 = FixedPositionalEncoding(context)
		# self.position_encoding7 = FixedPositionalEncoding(context)
		# self.position_encoding8 = FixedPositionalEncoding(context)
		# self.position_encoding9 = FixedPositionalEncoding(context)
		# self.position_encoding10 = FixedPositionalEncoding(context)
		# self.position_encoding11 = FixedPositionalEncoding(context)
		# self.position_encoding12 = FixedPositionalEncoding(context)
		# self.position_encoding13 = FixedPositionalEncoding(context)
		# self.position_encoding14 = FixedPositionalEncoding(context)

	def forward(self,index, context1, context2, context3, context4):

		self.index = index[0]
		context_ori = torch.cat((context1,context2,context3,context4),1)

		# context = context4
		# self.Conv_modality0 = conv_block(self.in_dim // 4, self.in_dim).cuda()
		# context = self.Conv_modality0(context)
		if self.index == 0:
			context = context4
			context = self.position_encoding0(context)

		elif self.index == 1:
			context = context2
			context = self.position_encoding1(context)

		elif self.index == 2:
			context = context3
			context = self.position_encoding2(context)

		elif self.index == 3:
			context = context1
			context = self.position_encoding3(context)

		elif self.index == 4:
			context = torch.cat((context2,context4),1)
			context = self.position_encoding4(context)

		elif self.index == 5:
			context = torch.cat((context2,context3),1)
			context = self.position_encoding5(context)

		elif self.index == 6:
			context = torch.cat((context1,context3),1)
			context = self.position_encoding6(context)

		elif self.index == 7:
			context = torch.cat((context3,context4),1)
			context = self.position_encoding7(context)

		elif self.index == 8:
			context = torch.cat((context1,context4),1)
			context = self.position_encoding8(context)

		elif self.index == 9:
			context = torch.cat((context1,context2),1)
			context = self.position_encoding9(context)

		elif self.index == 10:
			context = torch.cat((context1,context2,context3),1)
			context = self.position_encoding10(context)

		elif self.index == 11:
			context = torch.cat((context1,context3,context4),1)
      context = self.position_encoding11(context)

		elif self.index == 12:
			context = torch.cat((context1,context2,context4),1)
			context = self.position_encoding12(context)

		elif self.index == 13:
			context = torch.cat((context2,context3,context4),1)
			context = self.position_encoding13(context)
      
		elif self.index == 14:
			context = context = self.position_encoding14(context)
		return context, context_ori
		
