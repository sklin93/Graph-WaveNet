import torch
import numpy as np
import os
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform
import ipdb
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

model_dict = torch.load('garage/syn_exp1_best_0.1.pth') #garage/CRASH_avgE_best_20.pth

ipdb.set_trace()

wts = model_dict['end_mlp_e.1.weight'].squeeze().cpu().numpy()
# wts2 = model_dict['end_mlp_e2.1.weight'].squeeze().cpu().numpy()
# wts3 = model_dict['end_mlp_e3.1.weight'].squeeze().cpu().numpy()
# wts = wts + wts2 + wts3

e2r = {}
for i in range(64):
    e2r[i] = list(np.argsort(abs(wts[i]))[::-1][:20])

base_d = '/home/sikun/Documents/data/MRI_EEG/'
sc_d = os.path.join(base_d, 'sc')

# Group assignment mapping: need to assign brain regions to closest electrodes
coor_mri = np.loadtxt(os.path.join(sc_d, 'Parcellations/MNI', 'Schaefer2018_200Parcels_17Networks_order_FSLMNI152_2mm.txt'),
                      usecols=(3,4,5,6))
coor_eeg = np.loadtxt(os.path.join(base_d, 'utils/eeg_coor_conv/ny_x_z'), usecols=(1,2,3))
eeg_permute = [1, 0, 2]
coor_eeg = coor_eeg[:, eeg_permute] #(64,3)

# # distance matrix of eeg electrodes
# dist_mat = squareform(pdist(coor_eeg))
# # closest idx
# dist_idx = np.zeros_like(dist_mat)
# for i in range(64):
# 	dist_idx[i] = np.argsort(dist_mat[i])
# np.savetxt('dist_idx', dist_idx, fmt='% d')

_coor_mri = []
# for each mri region, assign it to the closest eeg electrode
for i in tqdm(range(200)):
    _coor_mri.append(coor_mri[coor_mri[:, -1] == (i+1)][:,:3].mean(0))
   
coor_mri = np.stack(_coor_mri) #(200,3)

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_trisurf(coor_mri[:, 0], coor_mri[:, 1], coor_mri[:, 2])
# ax.plot_trisurf(coor_eeg[:, 0], coor_eeg[:, 1], coor_eeg[:, 2], alpha=0.5)
# plt.show()

check_idxs = [0, 10, 30, 35, 40, 45] # 0-63

# for check_idx in check_idxs:
for check_idx in range(64):
	fig = plt.figure()
	ax = fig.add_subplot(311)
	plt.scatter(coor_eeg[:,0], coor_eeg[:,2], color='gray')
	plt.scatter(coor_mri[:,0], coor_mri[:,2], color='green')
	plt.scatter(coor_eeg[check_idx,0], coor_eeg[check_idx,2], color='red')
	for mri_idx in list(e2r[check_idx]):
		plt.scatter(coor_mri[mri_idx,0], coor_mri[mri_idx,2], color='blue')
	ax.set_aspect('equal')

	ax = fig.add_subplot(312)
	plt.scatter(coor_eeg[:,0], coor_eeg[:,1], color='gray')
	plt.scatter(coor_mri[:,0], coor_mri[:,1], color='green')
	plt.scatter(coor_eeg[check_idx,0], coor_eeg[check_idx,1], color='red')
	for mri_idx in list(e2r[check_idx]):
		plt.scatter(coor_mri[mri_idx,0], coor_mri[mri_idx,1], color='blue')
	ax.set_aspect('equal')

	ax = fig.add_subplot(313)
	plt.scatter(coor_eeg[:,1], coor_eeg[:,2], color='gray')
	plt.scatter(coor_mri[:,1], coor_mri[:,2], color='green')
	plt.scatter(coor_eeg[check_idx,1], coor_eeg[check_idx,2], color='red')
	for mri_idx in list(e2r[check_idx]):
		plt.scatter(coor_mri[mri_idx,1], coor_mri[mri_idx,2], color='blue')
	ax.set_aspect('equal')

	plt.savefig('wts_viz/'+str(check_idx)+'.png', dpi=900)

# plt.show()
