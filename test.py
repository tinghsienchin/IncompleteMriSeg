import TestDataset
import config
import torch
import torch.nn.functional as F
import torch.utils
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
# from models.UNet import UNet
from models.ResUnet import ResUNet
import torch
from tqdm import tqdm
from loss import *
import scipy.io as sio
import TestDataset
import os
import numpy as np
from common import *

def val(model, val_loader):
	model.eval()
	filename_list = load_file_name_list(os.path.join(args.dataset_path,'val_name_list_0.txt'))

	image = np.zeros((128,128,128))
	mask = np.zeros((128,128,128))
	all_val_loss = []
	batch_loss = []
	with torch.no_grad():
		for idx, (data, target) in tqdm(enumerate(val_loader), total = len(val_loader)):
			data = torch.squeeze(data, dim = 0)
			target = torch.squeeze(target, dim = 0)
			data, target = data.float(), target.float()
			data, target = data.to(device), target.to(device)
			
			preds = model(data)
			loss = dice(preds, target, 0)
			preds = F.sigmoid(preds)
			
			# preds = torch.argmax(preds,dim=1)

			mri_file = filename_list[idx]

			image = data[0, 0, :, :, :].data.cpu().numpy()
			mask = preds[0, 0, :, :, :].data.cpu().numpy()
			# mask2 = preds[1, :, :, :].data.cpu().numpy()
			sio.savemat(os.path.join(args.matpath + 'image', mri_file), {'image':image})

			mri_file_tempt = mri_file.replace('_t1.mat','_seg.mat')

			sio.savemat(os.path.join(args.matpath + 'mask', mri_file_tempt), {'mask':mask})
			# sio.savemat(os.path.join(args.matpath + 'mask', mri_file), {'mask1':mask1 , 'mask2':mask2, 'task':task_id})

			batch_loss.append(float(loss))

		all_val_loss = batch_loss
		# batch_loss = np.mean(batch_loss)
		# all_val_loss.append(batch_loss)
	return preds, data, idx, all_val_loss

if __name__ == '__main__':
	args = config.args
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	val_set = TestDataset.TestDataset(args.dataset_path, args.test_batch_size, args.mri_size)
	val_loader = DataLoader(dataset = val_set, shuffle = False, num_workers = args.num_workers)
	# model = UNet(in_channels = 1, n_classes=args.num_classes, base_n_filter=16).to(device)
	model = ResUNet(in_channels = 1, n_classes=args.num_classes, base_n_filter=16, layers = [1, 2, 2, 2, 2]).to(device)
	ckpt = torch.load('./output/{}/best_model.pth'.format(args.save))
	model.load_state_dict(ckpt['net'])
	preds, data, idx, all_val_loss = val(model, val_loader)
	sio.savemat(os.path.join(args.matpath + 'mask', 'loss.mat'), {'loss':all_val_loss})