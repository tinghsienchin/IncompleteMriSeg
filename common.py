import os
import torch
import scipy.io as sio
import numpy as np

total_mean = 0;
total_std = 1;

def norm_img(image):
    volume = image
    volume[volume<0] = 0
    pixels = volume
    mean = pixels.mean()
    std  = pixels.std()
    out = (volume - mean)/std
    out[out<0] = 0
    return out

def sio_read_mat(path, name  = 'image'):
	if name == 'image':
		data = sio.loadmat(path)[name].astype(np.float)

	elif name == 'mask':
		data = sio.loadmat(path)[name].astype(np.int8)

	elif name == 'task':
		data = sio.loadmat(path)[name].astype(np.int8)

	return data

def make_one_hot_3d(x, n, id): # 对输入的volume数据x，对每个像素值进行one-hot编码
    one_hot = np.zeros([x.shape[0], x.shape[1], x.shape[2], n]) # 创建one-hot编码后shape的zero张量
    if id == 1:
    	organ = (x >= 1)
    	tumor = (x == 2)
    elif id == 0:
    	organ = (x == 1)
    	tumor = None
    else:
    	print("Error, No such task!")
    	return None
    
    if organ is None:
    	one_hot[:, :, :, 0] = one_hot[:, :, :, 0] - 1
    else:
    	one_hot[:, :, :, 0] = np.where(organ, 1, 0)
    
    if tumor is None:
    	one_hot[:, :, :, 1] = one_hot[:, :, :, 1] - 1
    else:
    	one_hot[:, :, :, 1] = np.where(tumor, 1, 0)

    return one_hot
    """
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for v in range(x.shape[2]):
                one_hot[i, j, v, int(x[i, j, v])] = 1 # 给相应类别的位置置位1，模型预测结果也应该是这个shape
    return one_hot
    """

def load_file_name_list(file_path):
    file_name_list = []
    with open(file_path, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline().strip()  # 整行读取数据
            if not lines:
                break
                pass
            file_name_list.append(lines)
            pass
    return file_name_list