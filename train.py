import config
import torch
import torch.nn as nn
import torch.utils
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
# from models.UNet import UNet
from models.ResUnet import ResUNet
import MRIDataset
import logger
import init_util
import timeit
from loss import *
from collections import OrderedDict
from tqdm import tqdm
import os
from visdom import Visdom


start = timeit.default_timer()

def lr_poly(base_lr, iter, max_iter, power):
	return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate(optimizer, i_iter, lr, num_steps, power):
	""""Sets the learning_rate to the initial LR divided by 5 at 60th, 120th and 160th"""
	lr = lr_poly(lr, i_iter, num_steps, power)
	optimizer.param_groups[0]['lr'] = lr
	return lr


def val(model, val_loader):
	model.eval()
	epoch_loss = []
	with torch.no_grad():
		for idx, (data, target) in tqdm(enumerate(val_loader), total = len(val_loader)):
			data = torch.squeeze(data, dim = 0)
			target = torch.squeeze(target, dim = 0)
			data, target = data.float(), target.float()
			data, target = data.to(device), target.to(device)

			preds = model(data)
			loss = dice(preds, target, 0)

			epoch_loss.append(float(loss))
		
		epoch_loss = np.mean(epoch_loss)
	
	return OrderedDict({'Val Loss': epoch_loss})

def train(model, train_loader):
	print("==============Epoch:{}==================".format(epoch))
	model.train()
	epoch_loss = []

	viz = Visdom()

	for idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
		data = torch.squeeze(data, dim = 0) ###squeeze检查一下
		print(data.shape)
		target = torch.squeeze(target, dim = 0)
		data, target = data.float(), target.float()
		data, target = data.to(device), target.to(device)

		optimizer.zero_grad()
		preds = model(data)

		loss = dice(preds, target, 0)

		loss.backward()
		optimizer.step()
		
		epoch_loss.append(float(loss))

		viz.image(preds[1,:,10,:,:], win='mri/train1', env='output')
		viz.image(preds[1,:,60,:,:], win='mri/train2', env='output')
		viz.image(preds[1,:,70,:,:], win='mri/train3', env='output')
		viz.image(preds[1,:,90,:,:], win='mri/train4', env='output')

		viz.image(target[1,:,10,:,:], win='mri/train5', env='output')
		viz.image(target[1,:,60,:,:], win='mri/train6', env='output')
		viz.image(target[1,:,70,:,:], win='mri/train7', env='output')
		viz.image(target[1,:,90,:,:], win='mri/train8', env='output')
	
	epoch_loss = np.mean(epoch_loss)
	print("epoch_loss",epoch_loss)
	
	end = timeit.default_timer()
	print(end - start, 'seconds')

	return OrderedDict({'Train Loss': epoch_loss})

def reload_model(reload_path, reload = False):
	if reload:
		print('loading from checkpoint: {}'.format(reload_path))
		if os.path.exists(reload_path):
			model.load_state_dict(torch.load(args.reload_path, map_location=torch.device('cpu')))
		else:
			print('File not exists in the reload path: {}'.format(args.reload_path))

if __name__ == '__main__':
	args = config.args
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	torch.cuda.manual_seed(args.seed)
	#data info
	train_set = MRIDataset.MRIDataset(args.dataset_path, args.batch_size, args.mri_size, mode='train')
	val_set = MRIDataset.MRIDataset(args.dataset_path, args.batch_size, args.mri_size, mode='val')
	train_loader = DataLoader(dataset = train_set, shuffle = True, num_workers = args.num_workers)
	val_loader = DataLoader(dataset = val_set, shuffle = True, num_workers = args.num_workers)
	# model = UNet(in_channels = 1, n_classes=args.num_classes, base_n_filter=16).to(device)
	model = ResUNet(in_channels = 1, n_classes=args.num_classes, base_n_filter=8, layers = [1, 2, 2, 2, 2]).to(device)
	# model = UNet3D(num_classes=args.num_classes, weight_std=args.weight_std).to(device)
	#reload_model(args.reload, args.reload_path)
	optimizer = optim.SGD(model.parameters(), args.learning_rate, args.momentum, nesterov=True)
	init_util.print_network(model)

	log = logger.Logger('./output/{}'.format(args.save))

	best = [0, np.inf] # 初始化最优模型的epoch和performance
	trigger = 0# early stop 计数器

	for epoch in range(1, args.epochs + 1):
		adjust_learning_rate(optimizer, epoch, args.learning_rate, args.epochs, args.power)
		train_log = train(model, train_loader)
		val_log = val(model, val_loader)
		log.update(epoch, train_log, val_log)

		#save check point
		state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
		torch.save(state, os.path.join('./output/{}'.format(args.save), 'latest_model.pth'))
		trigger += 1
		if val_log['Val Loss'] < best[1]:
			print('Saving best model')
			torch.save(state, os.path.join('./output/{}'.format(args.save), 'best_model.pth'))
			best[0] = epoch
			best[1] = val_log['Val Loss']
			trigger = 0
		print('Best performance at Epoch: {} | {}'.format(best[0],best[1]))

		if args.early_stop is not None:
			if trigger >= args.early_stop:
				print("=> early stopping")
				break
		torch.cuda.empty_cache() 	
