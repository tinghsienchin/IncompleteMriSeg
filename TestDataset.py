import os
import torch
from torch.utils.data import Dataset
import numpy as np
from common import *

class TestDataset(Dataset):
	def __init__(self, dataset_path, batch_size, mri_size):
		self.dataset_path = dataset_path
		self.batch_size = batch_size
		self.mri_size = mri_size
		self.nlabels = 1
		self.filename_list = load_file_name_list(os.path.join(dataset_path,'val_name_list_0.txt'))


	def __getitem__(self, index):
		data, target = self.get_train_batch_by_index(train_batch_size = self.batch_size,
															 mri_size = self.mri_size, index = index)
		data = data.transpose(0, 4, 1, 2, 3) #[batch_size, label_num, W, H, D]
		target = target.transpose(0, 4, 1, 2, 3)
		return torch.from_numpy(data), torch.from_numpy(target)
	def __len__(self):
		return len(self.filename_list)

	def get_train_batch_by_index(self, train_batch_size, mri_size, index):
		train_imgs = np.zeros([train_batch_size, mri_size[0], mri_size[1], mri_size[2], 1]) #[batch_size, W, H, D, label_num]
		train_labels = np.zeros([train_batch_size, mri_size[0], mri_size[1], mri_size[2], self.nlabels])
		print("loading:", self.filename_list[index])
		img, label = self.get_np_data_3d(self.filename_list[index])
		sub_img, sub_label = img, label
		sub_img = sub_img[: , :, :, np.newaxis]
		
		if self.nlabels > 1:
			sub_label_onehot = make_one_hot_3d(sub_label, self.nlabels, t_id)
			train_imgs[0] = sub_img
			train_labels[0] = sub_label_onehot
		else:
			train_imgs[0] = sub_img
			sub_label = sub_label[: , :, :, np.newaxis]
			train_labels[0] = sub_label
		# img, label, t_id = self.get_np_data_3d(self.filename_list[index])
		# for i in range(train_batch_size):
		# 	sub_img, sub_label = img, label
		# 	sub_img = sub_img[: , :, :, np.newaxis]
			
		# 	if self.nlabels > 1:
		# 		sub_label_onehot = make_one_hot_3d(sub_label, self.nlabels, t_id)
		# 		train_imgs[i] = sub_img
		# 		train_labels[i] = sub_label_onehot
		# 	else:
		# 		train_imgs[i] = sub_img
		# 		sub_label = sub_label[: , :, :, np.newaxis]
		# 		train_labels[i] = sub_label

		return train_imgs, train_labels

	def get_np_data_3d(self, filename):
		data_np = sio_read_mat(self.dataset_path + 'image' + '/' + filename, 'image')

		filename_tempt = filename.replace('_t1.mat','_seg.mat')

		label_np = sio_read_mat(self.dataset_path + 'mask' + '/' + filename_tempt, 'mask')

		# task_id = sio_read_mat(self.dataset_path + 'mask' + '/' + filename, 'task')

		return data_np, label_np