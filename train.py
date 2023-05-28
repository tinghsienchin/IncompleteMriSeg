import config
import torch
import torch.nn as nn
import torch.utils
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
# from models.UNet import UNet
# from models.ResUnet import ResUNet
# from models.UNet_ori import UNet
# from models.UNet_resori import UNet
# from models.UNet_cor import UNet
# from models.Unet_cm import UNet
from models.ResUnet_cm import UNet
import MRIDataset2
import logger
import init_util
import timeit
from loss import *
from collections import OrderedDict
from tqdm import tqdm
import os
from visdom import Visdom
from common import feature_map_save
from torchstat import stat


start = timeit.default_timer()

def lr_poly(base_lr, iter, max_iter, power):
	return base_lr*(0.1**(float(iter)//30))
	# return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate(optimizer, i_iter, lr, num_steps, power):
	""""Sets the learning_rate to the initial LR divided by 5 at 60th, 120th and 160th"""
	lr = lr_poly(lr, i_iter, num_steps, power)
	optimizer.param_groups[0]['lr'] = lr
	return lr


def val(model, val_loader):
	model.eval()
	epoch_loss = []
	t1_loss = []
	t2_loss = []
	t1ce_loss = []
	flair_loss = []
	fusion_loss = []
	full_loss = []
	silmilarity_loss = []
	silmilarity_loss1 = []
	silmilarity_loss2 = []
	with torch.no_grad():
		for idx, (data, target) in tqdm(enumerate(val_loader), total = len(val_loader)):
			# data = torch.squeeze(data, dim = 0)
			# target = torch.squeeze(target, dim = 0)
			data, target = data.float(), target.float()
			data, target = data.to(device), target.to(device)

			preds = model(data)
			pred_dice = preds[0:6]
			loss = []
			for pred in pred_dice:
				loss.append(dice(pred, target, 0).cpu())
			loss.append(similarity_loss(preds[6],preds[7]).cpu())
			loss.append(similarity_loss(preds[8],preds[9]).cpu())
			# loss.append(similarity_loss(preds[10],preds[11]).cpu())
			total_loss = np.mean(loss)

			epoch_loss.append(float(total_loss))
			t1_loss.append(float(loss[0]))
			t2_loss.append(float(loss[1]))
			t1ce_loss.append(float(loss[2]))
			flair_loss.append(float(loss[3]))
			fusion_loss.append(float(loss[4]))
			full_loss.append(float(loss[5]))
			silmilarity_loss.append(float(loss[6]))
			silmilarity_loss1.append(float(loss[7]))
			# silmilarity_loss2.append(float(loss[8]))
		
		avg_loss = np.mean(epoch_loss)
		t1_loss = np.mean(t1_loss)
		t2_loss = np.mean(t2_loss)
		t1ce_loss = np.mean(t1ce_loss)
		flair_loss = np.mean(flair_loss)
		fusion_loss = np.mean(fusion_loss)
		full_loss = np.mean(full_loss)
		silmilarity_loss = np.mean(silmilarity_loss)
		silmilarity_loss1 = np.mean(silmilarity_loss1)
		# silmilarity_loss2 = np.mean(silmilarity_loss2)


	
	return OrderedDict({'Val Loss': avg_loss, 't1':t1_loss, 't2':t2_loss, 'flair':t1ce_loss, 't1ce':flair_loss, 'similarity loss':silmilarity_loss,'similarity loss1':silmilarity_loss1, 'fusion Loss':fusion_loss, 'full Loss':full_loss})

def train(model, train_loader):
	print("==============Epoch:{}==================".format(epoch))
	model.train()
	epoch_loss = []

	viz = Visdom()

	for idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
		# data = torch.squeeze(data, dim = 0) ###squeeze检查一下
		# target = torch.squeeze(target, dim = 0)
		data, target = data.float(), target.float()
		data, target = data.to(device), target.to(device)

		optimizer.zero_grad()
		preds = model(data)
		pred_dice = preds[0:6]
		loss = 0
		for pred in pred_dice:
			loss += dice(pred, target, 0)
		loss += similarity_loss(preds[6],preds[7])
		loss += similarity_loss(preds[8],preds[9])
		# loss += similarity_loss(preds[10],preds[11])
		# loss /= (len(preds) - 3)
		loss /= (len(preds) - 2)
		# preds, feature_map = model(data)
		# loss = dice(preds, target, 0)
		# print("feature_map saving")
		# feature_map_save(data, target, feature_map, args.feature_path)

		loss.backward()
		optimizer.step()
		
		epoch_loss.append(float(loss))
	
	epoch_loss = np.mean(epoch_loss)
	print("epoch_loss",epoch_loss)
	
	end = timeit.default_timer()
	print(end - start, 'seconds')

	return OrderedDict({'Train Loss': epoch_loss})

def reload_model(reload_path, reload = False):
	if reload:
		print('loading from checkpoint: {}'.format(reload_path))
		if os.path.exists(reload_path):
			model.load_state_dict(
				torch.load(args.reload_path, map_location=torch.device('cpu'))['net']
				)
		else:
			print('File not exists in the reload path: {}'.format(args.reload_path))

if __name__ == '__main__':
	args = config.args
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	torch.cuda.manual_seed(args.seed)
	#data info
	train_set = MRIDataset2.MRIDataset(args.dataset_path, args.mri_size, mode='train')
	val_set = MRIDataset2.MRIDataset(args.dataset_path, args.mri_size, mode='val')
	train_loader = DataLoader(batch_size = 1, dataset = train_set, shuffle = True, num_workers = args.num_workers)
	val_loader = DataLoader(batch_size = 1, dataset = val_set, shuffle = True, num_workers = args.num_workers)
	model = UNet(in_channels = 4, n_classes=1, base_n_filter=32).to(device)
	stat(model,(128,128,128))
	# model = UNet(in_channels = 1, n_classes=args.num_classes, base_n_filter=16).to(device)
	# model = ResUNet(in_channels = 1, n_classes=args.num_classes, base_n_filter=16, layers = [1, 2, 2, 2, 2]).to(device)

	# model = UNet3D(num_classes=args.num_classes, weight_std=args.weight_std).to(device)
	# reload_model(True, args.reload_path)
	# optimizer = optim.SGD(model.parameters(), args.learning_rate, args.momentum, nesterov=True)
	optimizer = optim.Adam(model.parameters(), args.learning_rate)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5,patience=10,verbose=1)
	init_util.print_network(model)

	log = logger.Logger('./output/{}'.format(args.save))

	best = [0, np.inf] # 初始化最优模型的epoch和performance
	trigger = 0# early stop 计数器

	for epoch in range(1, args.epochs + 1):
		# adjust_learning_rate(optimizer, epoch, args.learning_rate, args.epochs, args.power)
		train_log = train(model, train_loader)
		val_log = val(model, val_loader)
		log.update(epoch, train_log, val_log)
		scheduler.step(val_log['full Loss']) #监督full_loss
		#save check point
		state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
		torch.save(state, os.path.join('./output/{}'.format(args.save), 'latest_model.pth'))
		trigger += 1
		if val_log['full Loss'] < best[1]:
			print('Saving best model')#保存fusion best
			torch.save(state, os.path.join('./output/{}'.format(args.save), 'best_model.pth'))
			best[0] = epoch
			best[1] = val_log['full Loss']
			trigger = 0
		print('Best performance at Epoch: {} | {}'.format(best[0],best[1]))

		if args.early_stop is not None:
			if trigger >= args.early_stop:
				print("=> early stopping")
				break
		torch.cuda.empty_cache() 	
