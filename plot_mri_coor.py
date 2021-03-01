import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import ipdb

''' 
For checking brain region's location (whether small difference in parcellation number means the region proximity)
'''

coor_mri = np.loadtxt('/host/data/MRI_EEG/utils/eeg_coor_conv/Schaefer2018_200Parcels_17Networks_order_FSLMNI152_2mm.txt', usecols=(3,4,5,6))

coor_mean = []

for i in range(1,201):
	cur_coor = list(coor_mri[coor_mri[:,-1] == i][:,:3].mean(0))
	cur_coor.append(i)
	coor_mean.append(cur_coor)

coor_mean = np.stack(coor_mean).astype(np.float16)

'''plot region number in 3d space'''
fig = plt.figure()
ax = fig.gca(projection='3d')
for i in range(len(coor_mean)):
	ax.text(coor_mean[i,0], coor_mean[i,1], coor_mean[i,2], str(int(coor_mean[i,3])))

ax.set_xlim(coor_mean[:,0].min()-1, coor_mean[:,0].max()+1)
ax.set_ylim(coor_mean[:,1].min()-1, coor_mean[:,1].max()+1)
ax.set_zlim(coor_mean[:,2].min()-1, coor_mean[:,2].max()+1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

'''project to 2d'''
# fig = plt.figure()
# ax = fig.gca()
# for i in range(200):
# 	ax.text(coor_mean[i,0], coor_mean[i,1], str(int(coor_mean[i,3])))

# ax.set_xlim(coor_mean[:,0].min()-1, coor_mean[:,0].max()+1)
# ax.set_ylim(coor_mean[:,1].min()-1, coor_mean[:,1].max()+1)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# plt.show()

'''check whether larger number discrepancy means larger Euclidean distance'''
# # build distance matrix
# dist_mx = np.zeros((200,200))
# for i in range(1, 201):
# 	for j in range(i+1,201):
# 		dist_mx[i-1,j-1] = dist_mx[j-1,i-1] = ((coor_mean[coor_mean[:,-1]==i].squeeze()[:3] - coor_mean[coor_mean[:,-1]==j].squeeze()[:3])**2).sum()
# print(dist_mx)

'''
Conclusion: no, larger number discrepancy doesn't mean larger Euclidean distance
To properly cluster, need to transfer the index back to coordinate space first
'''