import numpy as  np
import os
import SimpleITK as sitk
import random
import scipy.io as sio
import shutil
import config
from sklearn.model_selection import KFold
from common import norm_img


class DtasetList(object):
	"""funcrion for  creating DtasetList"""
	def __init__(self, raw_dataset_path, fixed_dataset_path):
		self.raw_path = raw_dataset_path
		self.fixed_path = fixed_dataset_path
		self.KF = KFold(n_splits = 5, shuffle = True, random_state = 66) #5折交叉验证

		if os.path.exists(self.fixed_path + 'image'):#创建保存目录
			shutil.rmtree(self.fixed_path + 'image')
		else:
			os.makedirs(self.fixed_path + 'image')

		if os.path.exists(self.fixed_path + 'mask'):
			shutil.rmtree(self.fixed_path + 'mask')
		else:
			os.makedirs(self.fixed_path + 'mask')


		self.fix_data()
		self.write_KFold_train_val_name_list()
	def fix_data(self):
		###将slice小于128的扩充,若数据slice超过128则不会被使用####
		extend_slice = 128
		image_folder = os.listdir(self.raw_path + 'image')###要加/
		print('the amount of raw dataset is :', len(image_folder))

		for mri_file in image_folder:
			print(mri_file)
			mri_segfile = mri_file.replace('_t1.mat','_seg.mat')
			print(mri_segfile)
			#将mri 读入
			image = sio.loadmat(os.path.join(self.raw_path + 'image', mri_file))['image'].astype(np.float) #join不用+/
			image = norm_img(image)
			mask = sio.loadmat(os.path.join(self.raw_path + 'mask', mri_segfile))['mask'].astype(np.int8) #join不用+/
			# task_id = sio.loadmat(os.path.join(self.raw_path + 'mask', mri_file))['task'].astype(np.int8) #join不用+/
			
			x, y, z = image.shape
			if z < extend_slice:
				z1 = int((extend_slice - z)/2)
				z2 = extend_slice - z1 -z
				patch1 = np.zeros(shape=[x,y,z1], dtype = np.int8)
				patch2 = np.zeros(shape=[x,y,z2], dtype = np.int8)
				image = np.concatenate([patch1, image, patch2], axis = 2)
				mask = np.concatenate([patch1, mask, patch2], axis = 2)
				print("PATH:", os.path.join(self.fixed_path + 'image', mri_file))
				sio.savemat(os.path.join(self.fixed_path + 'image', mri_file), {'image':image})
				sio.savemat(os.path.join(self.fixed_path + 'mask', mri_file), {'mask':mask , 'task':task_id})

			elif z == extend_slice:
				mask[mask>0] = 1
				print("PATH:", os.path.join(self.fixed_path + 'image', mri_file))
				sio.savemat(os.path.join(self.fixed_path + 'image', mri_file), {'image':image})
				sio.savemat(os.path.join(self.fixed_path + 'mask', mri_segfile), {'mask':mask})

			else:
				print("invalid_data", mri_file, "slice", z)

	def write_train_val_name_list(self):###交叉验证可在这做生成多个文件
		data_name_list = os.listdir(self.fixed_path + 'image')
		data_num = len(data_name_list)
		print('the amount of fixed samples is', data_num)
		random.shuffle(data_name_list)

		train_rate = 0.5
		val_rate = 0.5

		assert val_rate + train_rate == 1.0
		train_name_list = data_name_list[0:int(data_num*train_rate)]
		val_name_list = data_name_list[int(data_num*train_rate):int(data_num*(train_rate + val_rate))]

		self.write_name_list(train_name_list, "train_name_list.txt")
		self.write_name_list(val_name_list, "val_name_list.txt")


	def write_KFold_train_val_name_list(self):
		data_name_list = os.listdir(self.fixed_path + 'image')
		data_num = len(data_name_list)
		print('the amount of fixed samples is', data_num)
		for train_index, val_index in self.KF.split(data_name_list):
			train_name_list, val_name_list = np.array(data_name_list)[train_index], np.array(data_name_list)[val_index]
			# print("TRAIN", train_name_list, "TEST", val_name_list)
			self.write_name_list(train_name_list, "train_name_list.txt")
			self.write_name_list(val_name_list, "val_name_list.txt")


	def write_name_list(self, name_list, file_name):
		count = 0
		file_name = self.check_name(file_name, count)
		f = open(self.fixed_path + file_name, 'w')
		for i in range(len(name_list)):
			f.write(str(name_list[i]) + '\n')
		f.close()

	def check_name(self, file_name, count):
		file_name_new = file_name
		if os.path.isfile(self.fixed_path + file_name):
			file_name_new = file_name[:file_name.rfind('.')]+'_'+str(count)+file_name[file_name.rfind('.'):]
			count = count + 1
		if os.path.isfile(self.fixed_path + file_name_new):
			file_name_new = self.check_name(file_name, count)
		return file_name_new

def main():
    raw_dataset_path = '../Bratsdata/trainingdata/'
    fixed_dataset_path = '../fixed_data/'
    
    DtasetList(raw_dataset_path,fixed_dataset_path)

if __name__ == '__main__':
    main()