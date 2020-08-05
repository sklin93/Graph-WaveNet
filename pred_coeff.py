from Utils.CRASH_loader import *
import Utils.util as util

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from torch.utils.checkpoint import checkpoint
from kymatio.numpy import Scattering1D
import sys

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import ipdb

np.random.seed(0)
torch.manual_seed(999)

# load eeg
comn_ids = get_comn_ids()
eeg = get_eeg(comn_ids)
eeg_len = 189282

eeg_mat = []
for subj in comn_ids:
	for k in eeg[subj]:
		cur_eeg = eeg[subj][k].transpose(1,0)[:eeg_len]
		if len(cur_eeg) == eeg_len:
			eeg_mat.append(cur_eeg)

del eeg
eeg_mat = np.stack(eeg_mat).transpose(0,2,1)
eeg_mat = eeg_mat.reshape(-1, eeg_mat.shape[-1])

# prep inputs & normalize
num_train = 10000
num_val = 2000
seq_len = 2912
eeg_mat = eeg_mat[:num_train+num_val, :seq_len][:,None,:]
eeg_mat = eeg_mat / np.max(np.abs(eeg_mat))
x_train = eeg_mat[:num_train]
x_val = eeg_mat[num_train:]

# scattering consts (for y)
J = 6 # or smaller
Q = 2 # or smaller
scattering = Scattering1D(J, seq_len, Q)
meta = scattering.meta()
order0 = np.where(meta['order'] == 0) #1*45
order1 = np.where(meta['order'] == 1) #13*45
order2 = np.where(meta['order'] == 2) #28*45
y = scattering(eeg_mat)
y[:,:,order1] *= 10
y[:,:,order2] *= 100
print(y.min(), y.max())
del eeg_mat
y_train = y[:num_train]
y_val = y[num_train:]
del y

# model
class Net(nn.Module):
	"""ConvNet -> Max_Pool -> RELU -> ConvNet -> Max_Pool -> RELU -> FC -> RELU -> FC -> SOFTMAX"""
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv1d(1, 20, 10)
		self.conv2 = nn.Conv1d(20, 400, 10)
		self.conv3 = nn.Conv1d(400, 2025, 10)

		self.b1_1 = nn.Conv2d(310, 20, (6,1))
		self.b1_2 = nn.Conv2d(20, 1, (3,1))
		self.b1_3 = nn.Conv2d(1, 1, (3,1))

		self.b2_1 = nn.Conv2d(310, 20, (6,1))
		self.b2_2 = nn.Conv2d(20, 1, (5,1))
		self.b2_3 = nn.Conv2d(1, 1, (4,1))

		self.b3_1 = nn.Conv2d(310, 20, (7,1))
		self.b3_2 = nn.Conv2d(20, 1, (7,1))
		self.b3_3 = nn.Conv2d(1, 1, (6,1))

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.max_pool1d(x,3)
		x = F.relu(self.conv2(x))
		x = F.max_pool1d(x,3)
		x = F.relu(self.conv3(x))

		x = x.transpose(1,2)
		x = x.view(-1, x.shape[1], 45, 45)

		x1 = F.relu(self.b1_1(x))
		x1 = F.max_pool2d(x1,(4,1))
		x1 = F.relu(self.b1_2(x1))
		x1 = F.max_pool2d(x1,(2,1))
		x1 = F.relu(self.b1_3(x1))
		x1 = F.max_pool2d(x1,(2,1))

		x2 = F.relu(self.b2_1(x))
		x2 = F.max_pool2d(x2,(2,1))
		x2 = F.relu(self.b2_2(x2))
		x2 = F.relu(self.b2_3(x2))

		x3 = F.relu(self.b3_1(x))
		x3 = F.relu(self.b3_2(x3))
		x3 = F.relu(self.b3_3(x3))

		return x1, x2, x3 #[32, 1, 1/13/28, 45]

# main
num_epoch = 20
batch_size = 32
device = torch.device('cuda:0')
model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.0001)

for e in range(num_epoch):
	# # shuffle training set
	# permutation = np.random.permutation(num_train)
	# x_train, y_train = x_train[permutation], y_train[permutation]
	# training
	model.train()
	for batch_i in range(int(num_train//batch_size)):
		data = x_train[batch_i * batch_size: (batch_i + 1) * batch_size]
		target = y_train[batch_i * batch_size: (batch_i + 1) * batch_size]
		data, target = torch.Tensor(data).to(device), torch.Tensor(target).to(device)

		optimizer.zero_grad()
		pred = model(data)
		pred = torch.cat(pred, -2)
		# loss = util.masked_mae(pred, target, 0.0)
		loss = nn.MSELoss()(pred, target)
		loss.backward()
		optimizer.step()
		if batch_i % 20 == 0:
			print('batch', batch_i, loss.item(), 
				util.masked_mae(pred,target,0).item(),
				util.masked_mape(pred,target,0).item(),
				util.masked_rmse(pred,target,0).item())
	ipdb.set_trace()

	plt.figure()
	plt.plot(pred[0].flatten().squeeze().cpu().detach().numpy(), label='pred')
	plt.plot(target[0].flatten().squeeze().cpu().numpy(), label='real')
	plt.legend()
	plt.show()

	ipdb.set_trace()
	# validation
	test_loss = []
	model.eval()
	with torch.no_grad():
		for batch_i in range(int(num_val//batch_size)):
			data = x_val[batch_i * batch_size: (batch_i + 1) * batch_size]
			target = y_val[batch_i * batch_size: (batch_i + 1) * batch_size]

			data, target = torch.Tensor(data).to(device), torch.Tensor(target).to(device)
			pred = model(data)
			pred = torch.cat(pred, -2)
			loss = util.masked_mae(pred, target, 0.0)
			test_loss.append(loss)
	ipdb.set_trace() # len(test_loss), test_loss.mean()
