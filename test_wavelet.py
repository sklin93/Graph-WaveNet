import torch
from torch.autograd import backward
import numpy as np
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle
from kymatio.torch import Scattering1D
import pywt
import ipdb

CRASH_fname = 'CRASH_FE_filtered_subsampled_single.pkl'

with open(CRASH_fname, 'rb') as handle:
    F_t, adj_mx, adj_mx_idx, _input, _gt, coeffs, \
    inv_mapping, region_assignment, nTrain, nValid, \
    nTest, scaler_in, scaler_out = pickle.load(handle)


x = _gt[0,:,0][:500]
plt.plot(x)
plt.show()

J = 3
Q = 9
scattering = Scattering1D(J, len(x), Q)
ipdb.set_trace()
t1, t2 = pywt.dwt(x, 'db12')
rec = pywt.idwt(t1, t2,'db12')
'''
x = torch.from_numpy(x).contiguous()
Sx = scattering(x)
print(Sx.shape)

for i in range(len(Sx)):
	plt.plot(Sx[i].cpu().numpy())
plt.show()

# learning_rate = 100
# bold_driver_accelerator = 1.1
# bold_driver_brake = 0.55
# n_iterations = 200

# ###############################################################################
# # Reconstruct the scattering transform back to original signal.

# # Random guess to initialize.
# torch.manual_seed(0)
# y = torch.randn((len(x),), requires_grad=True)
# Sy = scattering(y)

# history = []
# signal_update = torch.zeros_like(x)

# # Iterate to recontsruct random guess to be close to target.
# for k in range(n_iterations):
#     # Backpropagation.
#     err = torch.norm(Sx - Sy)

#     if k % 10 == 0:
#         print('Iteration %3d, loss %.2f' % (k, err.detach().numpy()))

#     # Measure the new loss.
#     history.append(err)

#     backward(err)

#     delta_y = y.grad

#     # Gradient descent
#     with torch.no_grad():
#         signal_update = - learning_rate * delta_y
#         new_y = y + signal_update
#     new_y.requires_grad = True

#     # New forward propagation.
#     Sy = scattering(new_y)

#     if history[k] > history[k - 1]:
#         learning_rate *= bold_driver_brake
#     else:
#         learning_rate *= bold_driver_accelerator
#         y = new_y

# plt.figure(figsize=(8, 2))
# plt.plot(history)
# plt.title("MSE error vs. iterations")

# plt.figure(figsize=(8, 2))
# plt.plot(y.detach().numpy())
# plt.title("Reconstructed signal")

# plt.figure(figsize=(8, 8))
# plt.specgram(y.detach().numpy(), Fs=1024)
# plt.title("Spectrogram of reconstructed signal")

plt.show()
'''