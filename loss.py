import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import math
from common import*

smooth = 0.00001

class BinaryDiceLoss(nn.Module):
	"""docstring for BinaryDiceLoss"""
	def __init__(self, smooth = 1, p = 2, reduction='mean'):
		super(BinaryDiceLoss, self).__init__()
		self.smooth = smooth
		self.p = p
		self.reduction = reduction

	def forward(self, predict, target):
		assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
		predict = predict.contiguous().view(predict.shape[0], -1)
		target = target.contiguous().view(predict.shape[0], -1)

		num = torch.sum(torch.mul(predict, target), dim=1)
		den = torch.sum(predict, dim=1) + torch.sum(target, dim=1) + self.smooth

		dice_score = 2*num / den
		dice_loss = 1 - dice_score

		dice_loss_avg = dice_loss[target[:,0]!=-1].sum()

		return dice_loss_avg

class DiceLoss4MOTS(nn.Module):
	"""docstring for DiceLoss4MOTS"""
	def __init__(self, weight = None, ignore_index = None, num_classes=3, **kwargs):
		super(DiceLoss4MOTS, self).__init__()
		self.kwargs = kwargs
		self.weight = weight
		self.ignore_index = ignore_index
		self.num_classes = num_classes
		self.dice = BinaryDiceLoss(**self.kwargs)

	def forward(self, predict, target):

		total_loss = []
		predict = F.sigmoid(predict)

		for i in range(self.num_classes):
			if i !=self.ignore_index:
				dice_loss = self.dice(predict[:, i], target[:, i])
				if self.weight is not None:
					assert self.weight.shape[0] == self.num_classes, \
					       'Expect weight shape [{}], get[{}]'.format(self.num_classes, self.weight.shape[0])
					dice_loss *= self.weights[i]
				total_loss.append(dice_loss)

		total_loss = torch.stack(total_loss)
		total_loss = total_loss[total_loss==total_loss]

		return total_loss.sum()/total_loss.shape[0]

class CELoss4MOTS(nn.Module):
	"""docstring for CELoss4MOTS"""
	def __init__(self, ignore_index=None, num_classes=3, **kwargs):
		super(CELoss4MOTS, self).__init__()
		self.kwargs = kwargs
		self.num_classes = num_classes
		self.ignore_index = ignore_index
		self.criterion = nn.BCEWithLogitsLoss(reduction='none')

	def weight_function(self, mask):
		weights = torch.ones_like(mask).float()
		voxels_sum = mask.shape[0] * mask.shape[1] * mask.shape[2]
		for i in range(2):
			voxels_i = [mask == i][0].sum().cpu().numpy()
			w_i = np.log(voxels_sum / voxels_i).astype(np.float32)
			weights = torch.where(mask == i, w_i * torch.ones_like(weights).float(), weights)

		return weights

	def forward(self, predict, target):
		assert predict.shape == target.shape, 'predict & target shape do not match'

		total_loss = []
		for i in range(self.num_classes):
			if i != self.ignore_index:
				ce_loss = self.criterion(predict[:,i], target[:,i])
				ce_loss = torch.mean(ce_loss, dim = [1,2,3])

				ce_loss_avg = ce_loss[target[:, i, 0, 0, 0] != -1].sum() / ce_loss[target[:, i, 0, 0, 0] != -1].shape[0]

				total_loss.append(ce_loss_avg)

		total_loss = torch.stack(total_loss)
		total_loss = total_loss[total_loss == total_loss]

		return total_loss.sum()/total_loss.shape[0]

def dice(logits, targets, class_index):
	assert logits.shape[0] == targets.shape[0],"batch size don't match"
	probs = F.sigmoid(logits[:, class_index, :, :, :])

	probs = probs.contiguous().view(probs.shape[0], -1)
	targets = targets.contiguous().view(targets.shape[0], -1)

	num = torch.sum(torch.mul(probs, targets), dim=1)
	den = torch.sum(probs, dim=1) + torch.sum(targets, dim=1) + smooth

	dice_score = 2*num / den
	dice_loss = 1 - dice_score

	dice_loss_avg = dice_loss[targets[:,0]!=-1].sum() / dice_loss[targets[:,0]!=-1].shape[0]
	return dice_loss_avg

def dis(pred, targ):
	dist = abs(pred - targ)
	dist = np.sum(dist, axis = 0)
	dist = dist // 6
	dist_loss = math.log(1 + dist)/math.log(60)
	return dist_loss

def cor_loss(logits, targ_cor, class_index, score):
	probs = F.sigmoid(logits[:, class_index, :, :, :])
	probs = probs.data.cpu().numpy()
	targ_cor = targ_cor.data.cpu().numpy()
	score = score.data.cpu().numpy()
	loss = []
	cor_false1 = np.array([0, 0, 0, 128, 128, 128])
	cor_false2 = [0, 1, 0, 128, 128, 128]
	cor_false3 = [0, 2, 0, 128, 128, 128]
	thres = 0.3

	for i in range (0, probs.shape[0]):
		prob = probs[i, :, :, :]
		if score < 0.7:
			tempt = 0.7 - score
			prob[prob>thres+tempt] = 1
			prob[prob<thres+tempt] = 0
			cor = cal_property(prob)
			if cor.all() == cor_false1.all():
				loss.append(0)
			else:
				print(cor)
				loss.append(dis(cor, targ_cor[i, :]))
		else:
			loss.append(0)
		loss = np.mean(loss)
	return loss


# def similarity_loss(logits, targ):
# 	slimilarity = nn.L1Loss(reduction='elementwise_mean')
# 	loss = slimilarity(logits,targ)
# 	return loss

# def similarity_loss(logits, targ):
# 	logits = logits.view(1,-1)
# 	targ = targ.view(1,-1)
# 	slimilarity = torch.cosine_similarity(logits, targ, dim=0)
# 	loss = 1 - slimilarity
# 	loss = loss + 1e-8
# 	return loss

def similarity_loss(logits, targ):
	slimilarity = nn.MSELoss(reduce=True, size_average=True)
	loss = slimilarity(logits,targ)
	return loss
