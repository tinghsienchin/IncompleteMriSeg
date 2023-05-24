import os
import torch
from torch.utils.data import Dataset
import numpy as np
from common import *

class MRIDataset(Dataset):
	def __init__(self, dataset_path, mri_size ,mode = None):
		self.dataset_path = dataset_path
		self.mri_size = mri_size
		self.nlabels = 4

		if mode == 'train':
			self.filename_list = load_file_name_list(os.path.join(dataset_path,'train_name_list_2.txt'))
		elif mode == 'val':
			self.filename_list = load_file_name_list(os.path.join(dataset_path,'val_name_list_2.txt'))
		else:
			raise TypeError('Dataset mode TypeError!!')

	def __getitem__(self, index):
		data, target = self.get_train_batch_by_index(mri_size = self.mri_size, index = index)
		# data = data[:, :, :, np.newaxis]
		target = target[:, :, :, np.newaxis]
		# data = data.transpose(3, 0, 1, 2) #[batch_size, label_num, D, W, H]
		target = target.transpose(3, 0, 1, 2)
		return torch.from_numpy(data), torch.from_numpy(target)

	def __len__(self):
		return len(self.filename_list)

	def get_train_batch_by_index(self, mri_size, index):
		train_imgs = np.zeros([self.nlabels, mri_size[0], mri_size[1], mri_size[2]]) #[batch_size, D, W, H, label_num]
		train_labels = np.zeros([mri_size[0], mri_size[1], mri_size[2]])

		sub_img_t1, sub_img_t1ce, sub_img_t2, sub_img_flair, train_labels = self.get_np_data_3d(self.filename_list[index])
		train_imgs[0] = sub_img_t1
		train_imgs[1] = sub_img_t1ce
		train_imgs[2] = sub_img_t2
		train_imgs[3] = sub_img_flair
		# img, label, t_id = self.get_np_data_3d(self.filename_list[index])
		# for i in range(train_batch_size):
		# 	img, label = self.get_np_data_3d(self.filename_list[index])
		# 	print(self.filename_list[index])
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

		#################ti######################
		data_t1np = sio_read_mat(self.dataset_path + 'image' + '/' + filename, 'image')


		############t1ce#################
		filename_tempt = filename.replace('_t1.mat','_t1ce.mat')

		data_t1cenp = sio_read_mat(self.dataset_path + 'image' + '/' + filename_tempt, 'image')



        #############t2###################
		filename_tempt = filename_tempt.replace('_t1ce.mat','_t2.mat')

		data_t2np = sio_read_mat(self.dataset_path + 'image' + '/' + filename_tempt, 'image')




        #############flair################
		filename_tempt = filename_tempt.replace('_t2.mat','_flair.mat')

		data_flairnp = sio_read_mat(self.dataset_path + 'image' + '/' + filename_tempt, 'image')




        ###############mask###############
		filename_tempt = filename_tempt.replace('_flair.mat','_seg.mat')

		label_np = sio_read_mat(self.dataset_path + 'mask' + '/' + filename_tempt, 'mask')

		# task_id = sio_read_mat(self.dataset_path + 'mask' + '/' + filename, 'task')

		return data_t1np, data_t1cenp, data_t2np, data_flairnp, label_np
