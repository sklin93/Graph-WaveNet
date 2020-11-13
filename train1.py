import torch
import numpy as np
import argparse
import time
import Utils.util as util
from engine import trainer
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle
import os
import glob
from sklearn.preprocessing import RobustScaler
from sklearn.utils import shuffle
from kymatio.numpy import Scattering1D
import networkx as nx
import ipdb

# python train.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --num_nodes 207 --seq_length 12 --save ./garage/metr
# python train.py --gcn_bool --adjtype normlap --addaptadj --num_nodes 80 --data syn --blocks 2 --layers 2 --in_dim=1 --batch_size 32 --learning_rate 0.00005 --weight_decay 0.0001 --epochs 200
# python train.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --data CRASH --num_nodes 200 --seq_length 2912 --in_dim 1 --blocks 2 --layers 2 --batch_size 16 --learning_rate 0.0005 --weight_decay 0.011 --save ./garage/CRASH
# python train.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --data CRASH --num_nodes 200 --seq_length 2912 --in_dim 1 --blocks 2 --layers 2 --batch_size 8 --learning_rate 3e-4 --dropout 0.2 --save ./garage/CRASH --epochs 20 --nhid 128
# nohup python -u train.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --data CRASH --num_nodes 200 --seq_length 2912 --in_dim 1 --blocks 2 --layers 2 --batch_size 16 --learning_rate 0.0005 --save ./garage/CRASH > log_CRASH 2>&1 &
# (Notice the CRASH can handle batch size 32 on server)
##### single sample overfit
# python train.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --data CRASH --num_nodes 200 --seq_length 2912 --in_dim 1 --blocks 2 --layers 2 --batch_size 1 --learning_rate 0.001 --weight_decay 0 --dropout 0 --save ./garage/CRASH --epochs 8000
##### if using wavelet
# python train.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --data CRASH --num_nodes 200 --seq_length 2912 --in_dim 1 --blocks 2 --layers 2 --batch_size 8 --learning_rate 0.00001 --weight_decay 0.0001 --save ./garage/CRASH_wavelet
## on server
# python train.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --data CRASH --num_nodes 200 --seq_length 2912 --in_dim 1 --blocks 2 --layers 2 --batch_size 8 --learning_rate 0.0001 --weight_decay 0.0001 --device 'cuda:2' --save ./garage/CRASH_wavelet_mae
##### F only
# python train.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --data CRASH --num_nodes 200 --seq_length 8736 --in_dim 1 --blocks 2 --layers 2 --batch_size 8 --learning_rate 3e-4 --dropout 0.2 --save ./garage/CRASH --epochs 5 --kernel_size 3 --nhid 16
parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--data',type=str,default='data/METR-LA',help='data path')
parser.add_argument('--adjdata',type=str,default='data/sensor_graph/adj_mx.pkl',help='adj data path')
parser.add_argument('--adjtype',type=str,default='doubletransition',help='adj type')
parser.add_argument('--gcn_bool',action='store_true',help='whether to add graph convolution layer')
parser.add_argument('--aptonly',action='store_true',help='whether only adaptive adj')
parser.add_argument('--addaptadj',action='store_true',help='whether add adaptive adj')
parser.add_argument('--randomadj',action='store_true',help='whether random initialize adaptive adj')
parser.add_argument('--seq_length',type=int,default=120,help='')
parser.add_argument('--nhid',type=int,default=32,help='')
parser.add_argument('--in_dim',type=int,default=2,help='inputs dimension')
parser.add_argument('--num_nodes',type=int,default=80,help='number of nodes')
parser.add_argument('--kernel_size',type=int,default=2,help='kernel_size of dilated convolution')
parser.add_argument('--layers',type=int,default=2,help='number of layers per gwnet block')
parser.add_argument('--blocks',type=int,default=4,help='number of blocks in gwnet model')

parser.add_argument('--batch_size',type=int,default=32,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=0.01,help='weight decay rate')
parser.add_argument('--epochs',type=int,default=40,help='')
parser.add_argument('--print_every',type=int,default=50,help='')
# parser.add_argument('--seed',type=int,default=0,help='random seed')
parser.add_argument('--save',type=str,default='./garage/syn',help='save path')
parser.add_argument('--expid',type=int,default=1,help='experiment id')

args = parser.parse_args()

np.random.seed(0)
torch.manual_seed(999)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(999)
    torch.cuda.empty_cache()

def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)
    
def main(model_name=None, finetune=False, syn_file='syn_diffG.pkl', 
         scatter=False, subsample=False, map=False, F_only=False):
    '''directly loading trained model/ generated syn data
       scatter: whether use wavelet scattering
       subsample: whether subsample EEG signals to the same temporal resolution as fMRI's
       map: whether map to current EEG signal or (else) predict future EEG signal
    '''
    #load data
    same_G = False
    device = torch.device(args.device)

    if args.data == 'CRASH':
        adj_mx, fmri_mat, eeg_mat, region_assignment, F_t = util.load_dataset_CRASH(args.adjtype)
        adj_mx, fmri_mat, eeg_mat = shuffle(adj_mx, fmri_mat, eeg_mat,random_state = 0)
        # if not scatter:
        if True: #TODO: compare preprocessing performance difference
            # Standardize data
            num_subj, t_f, n_f = fmri_mat.shape
            _, t_e, n_e = eeg_mat.shape
            # ######## per channel standardization
            r_mean = fmri_mat.reshape(-1, fmri_mat.shape[-1]).mean(0) # region means (fMRI)
            r_std = fmri_mat.reshape(-1, fmri_mat.shape[-1]).std(0)
            e_mean = eeg_mat.reshape(-1, eeg_mat.shape[-1]).mean(0) # electrode means (EEG)
            e_std = eeg_mat.reshape(-1, eeg_mat.shape[-1]).std(0)

            # pkl_data = {}
            # pkl_data['r_mean'] = r_mean
            # pkl_data['r_std'] = r_std
            # pkl_data['e_mean'] = e_mean
            # pkl_data['e_std'] = e_std
            # with open('re_stat.pkl', 'wb') as handle:
            #     pickle.dump(pkl_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            # with open('re_stat.pkl', 'rb') as handle:
            #     pkl_data = pickle.load(handle)
            #     r_mean = pkl_data['r_mean']
            #     r_std = pkl_data['r_std']
            #     e_mean = pkl_data['e_mean']
            #     e_std = pkl_data['e_std']
            fmri_mat -= r_mean[None,None,:]
            fmri_mat /= r_std[None,None,:]
            eeg_mat -= e_mean[None,None,:]
            eeg_mat /= e_std[None,None,:]

            # # per channel min-max normalization
            # r_max = fmri_mat.reshape(-1, fmri_mat.shape[-1]).max(0)
            # r_min = fmri_mat.reshape(-1, fmri_mat.shape[-1]).min(0)
            # fmri_mat = (fmri_mat - r_min[None,None,:]) / (r_max[None,None,:] - r_min[None,None,:])

            # e_max = eeg_mat.reshape(-1, eeg_mat.shape[-1]).max(0)
            # e_min = eeg_mat.reshape(-1, eeg_mat.shape[-1]).min(0)
            # # eeg_mat = (eeg_mat - e_min[None,None,:]) / (e_max[None,None,:] - e_min[None,None,:])
            # eeg_mat -= e_min[None,None,:]
            # e_max -= e_min
            # eeg_mat /= e_max[None,None,:]
            
            # ################################

            # fmri_mat = fmri_mat.reshape(num_subj, -1)
            # _mean = fmri_mat.mean(0)
            # _std = fmri_mat.std(0)
            # ipdb.set_trace()
            # fmri_mat -= _mean
            # fmri_mat /= _std
            # fmri_mat = fmri_mat.reshape(num_subj, t_f, n_f)
            
            # eeg_mat = eeg_mat.reshape(num_subj, -1)
            # _mean = eeg_mat.mean(0)
            # _std = eeg_mat.std(0)
            # eeg_mat -= _mean
            # eeg_mat /= _std
            # eeg_mat = eeg_mat.reshape(num_subj, t_e, n_e)

            # Min-max normalization
            fmri_mat = (fmri_mat - fmri_mat.min()) / (fmri_mat.max() - fmri_mat.min())
            eeg_mat = (eeg_mat - eeg_mat.min()) / (eeg_mat.max() - eeg_mat.min())

            cutoff = (1/0.91)/(2*3)
            # band pass filter fMRI
            for i in range(fmri_mat.shape[0]):
                for j in range(fmri_mat.shape[1]):
                    fmri_mat[i,j] = util.butter_lowpass_filter(fmri_mat[i,j], cutoff, 0.91)
        else:
            fmri_mat = fmri_mat / np.max(np.abs(fmri_mat))
            eeg_mat = eeg_mat / np.max(np.abs(eeg_mat))

        '''
        print('eeg_mat min max:', eeg_mat.min(), eeg_mat.max())
        # LPF of EEG data
        filtered_eeg_fname = 'filtered_eeg_mat.npy'
        if os.path.isfile(filtered_eeg_fname):
            eeg_mat = np.load(filtered_eeg_fname)
            print('filtered eeg loaded')
        else:            
            for sp in range(eeg_mat.shape[0]): # for every sample
                print('\n', sp, end='')
                for nd in range(eeg_mat.shape[-1]): # for every node
                    print('.', end='')
                    d = eeg_mat[sp, :, nd]
                    power = np.abs(np.fft.fft(d))
                    freq = np.fft.fftfreq(eeg_mat.shape[1], 1/640)

                    # plt.figure()
                    # plt.plot(freq, power)
                    # plt.show()

                    # want to subsample to 1/0.91 Hz (same as fMRI)
                    # need to filter out frequencies above nyquist rate (half that)
                    cutoff = (1/0.91)/2
                    filtered_sig = util.butter_lowpass_filter(d, cutoff, 640)

                    # plt.figure()
                    # plt.plot(d)
                    # plt.plot(filtered_sig)
                    # plt.show()

                    # replace original signal with filtered signal
                    eeg_mat[sp, :, nd] = filtered_sig
            np.save(filtered_eeg_fname, eeg_mat)
        print('filtered eeg min max:', eeg_mat.min(), eeg_mat.max())
        '''
        
        # region_assignment: {EEG_electrodes: brain region}
        inv_mapping = {} #{brain region: EEG_electrodes}
        for k, v in region_assignment.items():
            for _v in v:
                if _v not in inv_mapping:
                    inv_mapping[_v] = []
                inv_mapping[_v] = sorted(list(set(inv_mapping[_v]+[k])))
        
        basic_len = 2912 # hard-coded fot F_t 582.4, used as the sliding window stride to avoid float
        assert int(args.seq_length % basic_len) == 0

        if scatter:
            # wavelet consts J=3 & Q=8
            J = 3 # or smaller
            Q = 9 # or smaller
            print(1+J*Q+J*(J-1)*Q/2, 1000/(2**J))
            scattering = Scattering1D(J, args.seq_length, Q) 
            # scattering = Scattering1D(J, 1000, Q) # predict shorter
            meta = scattering.meta()
            order0 = np.where(meta['order'] == 0) #1*45
            order1 = np.where(meta['order'] == 1) #13*45
            order2 = np.where(meta['order'] == 2) #28*45
            # load wavelet coefficient scalers
            with open('coeffs_scaler.pkl', 'rb') as handle:
                coeffs_scaler = pickle.load(handle)
            # for transform: make it into same shape as y_E
            mean = np.tile(coeffs_scaler['means'][None, None, ...], (args.batch_size, 64, 1, 1))
            std = np.tile(coeffs_scaler['stds'][None, None, ...], (args.batch_size, 64, 1, 1))

    elif args.data == 'syn':
        if  os.path.isfile(syn_file):
            with open(syn_file, 'rb') as handle:
                pkl_data = pickle.load(handle)
            nTrain, nValid, nTest, num_timestep = pkl_data['nTrain'], pkl_data['nValid'],\
                                                  pkl_data['nTest'], pkl_data['num_timestep']
            dataloader, adj_mx, F_t, G = pkl_data['dataloader'], pkl_data['adj_mx'],\
                                         pkl_data['F_t'], pkl_data['G']
            print('synthetic data loaded')
        else:
            nTrain = 200 #10 # Number of training samples
            nValid = int(0.2 * nTrain) # Number of validation samples
            nTest = int(0.2 * nTrain) # Number of testing samples
            num_timestep = 2400
            dataloader, adj_mx, F_t, G = util.load_dataset_syn(args.adjtype, args.num_nodes,
                                                               nTrain, nValid, nTest, num_timestep,
                                                               args.seq_length, args.batch_size, 
                                                               args.batch_size, args.batch_size, 
                                                               same_G=same_G)
            pkl_data = {'nTrain': nTrain, 'nValid': nValid, 'nTest': nTest,
                        'num_timestep': num_timestep, 'dataloader': dataloader,
                        'adj_mx': adj_mx, 'F_t': F_t, 'G':G}
            with open(syn_file, 'wb') as handle:
                pickle.dump(pkl_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if scatter:
            # wavelet consts
            J = 3
            Q = 2
            scattering = Scattering1D(J, args.seq_length, Q)
            meta = scattering.meta()
            order0 = np.where(meta['order'] == 0) #1*45
            order1 = np.where(meta['order'] == 1) #13*45
            order2 = np.where(meta['order'] == 2) #28*45

            sname = 'syn_coeffs_scaler.pkl'
            if os.path.isfile(sname):
                # load wavelet coefficient scalers
                with open(sname, 'rb') as handle:
                    coeffs_scaler = pickle.load(handle)
            else:
                # calculate & save scaler coeff
                coeffs = []
                for iter, (_, y, _) in enumerate(dataloader['train_loader'].get_iterator()):
                    coeffs.append(scattering(y.transpose(0,3,2,1)).squeeze())
                coeffs_scaler = {}
                coeffs_scaler['means'] = np.concatenate(coeffs).mean(0).mean(-1)
                coeffs_scaler['stds'] = np.concatenate(coeffs).std(0).mean(-1)
                with open(sname, 'wb') as handle:
                    pickle.dump(coeffs_scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)
                del coeffs
            # for transform: make it into same shape as y_E
            mean = np.tile(coeffs_scaler['means'][None, None, ..., None], (args.batch_size, 1, 1, 1, 30))
            std = np.tile(coeffs_scaler['stds'][None, None, ..., None], (args.batch_size, 1,1, 1, 30))
            # TODO: now just train for order0
            mean = mean[...,order0[0],:]
            std = std[...,order0[0],:]

    else:
        sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata,args.adjtype)
        dataloader = util.load_dataset_metr(args.data, args.batch_size, args.batch_size, 
                                            args.batch_size)
        F_t = None
        
    if args.data == 'CRASH':
        nTrain = round(0.7 * len(adj_mx))
        nValid = round(0.15 * len(adj_mx))
        nTest = len(adj_mx) - nTrain - nValid

        print('Train, Val, Test numbers:', nTrain, nValid, nTest)

        ## randomize SC entries to see the sensitivity to SC
        # use completely random SC w/ same level of sparsity
        n = args.num_nodes # number of nodes
        p = np.count_nonzero(adj_mx[0][0]) / (n*(n-1)/2) # probability for edge creation
        # G = nx.gnp_random_graph(n, p*2)
        # G = nx.newman_watts_strogatz_graph(n,5,p)
        G = nx.gnm_random_graph(n, np.count_nonzero(adj_mx[0][0]))
        for adj in adj_mx:
            # G = nx.gnp_random_graph(n, p)
            adj = util.mod_adj(nx.to_numpy_matrix(G), args.adjtype)

        # separate adj matrices into train-val-test samples
        adj_train = [[] for i in range(len(adj_mx[0]))]
        for a in adj_mx[:nTrain]:
            for i in range(len(adj_train)):
                adj_train[i].append(a[i])
        adj_train = [np.stack(np.asarray(i)) for i in adj_train]

        adj_val = [[] for i in range(len(adj_mx[0]))]
        for a in adj_mx[nTrain:-nTest]:
            for i in range(len(adj_val)):
                adj_val[i].append(a[i])
        adj_val = [np.stack(np.asarray(i)) for i in adj_val]

        adj_test = [[] for i in range(len(adj_mx[0]))]
        for a in adj_mx[-nTest:]:
            for i in range(len(adj_test)):
                adj_test[i].append(a[i])
        adj_test = [np.stack(np.asarray(i)) for i in adj_test]
        
        scaler_F = util.StandardScaler(mean=fmri_mat[:nTrain].mean(), 
                                        std=fmri_mat[:nTrain].std())
        scaler_E = util.StandardScaler(mean=eeg_mat[:nTrain].mean(), 
                                        std=eeg_mat[:nTrain].std())
        print(args)
        supports = {}
        supports['train'] = adj_train
        supports['val'] = adj_val
        supports['test'] = adj_test

        adjinit = {}
        if args.randomadj:
            adjinit['train'] = adjinit['val'] = adjinit['test'] = None
        else:
            adjinit['train'] = supports['train'][0]
            adjinit['val'] = supports['val'][0]
            adjinit['test'] = supports['test'][0]

        if args.aptonly:
            supports['train'] = supports['val'] = supports['test'] = None

        offset = args.seq_length // basic_len
        E_idxer = np.arange(args.seq_length)[None, :] + np.arange(0, 
                            eeg_mat.shape[1] - args.seq_length + 1, basic_len)[:, None]

        K = int(args.seq_length / F_t)
        F_idxer = np.arange(K)[None, :] + np.arange(0, fmri_mat.shape[1] - K + 1, 
                                                int(basic_len/F_t))[:, None]
        # x = F_idxer[:-offset]
        # y = F_idxer[offset:]

        assert len(F_idxer) == len(E_idxer)
        sample_per_suj = len(F_idxer) - offset
        batch_per_sub = sample_per_suj // args.batch_size
        print('Batch per subject:', batch_per_sub)

        if scatter:
            engine = trainer(coeffs_scaler, args.in_dim, args.seq_length, args.num_nodes, 
            # engine = trainer(coeffs_scaler, args.in_dim, 1000, args.num_nodes, # predict shorter
                             args.nhid, args.dropout, args.learning_rate, args.weight_decay, device, 
                             supports, args.gcn_bool, args.addaptadj, adjinit, args.kernel_size,
                             args.blocks, args.layers, out_nodes=eeg_mat.shape[-1], F_t=F_t,
                             meta=[order0[0],order1[0],order2[0]],scatter=True, F_only=F_only, 
                             batch_size=args.batch_size)
        else:
            engine = trainer([scaler_F,scaler_E], args.in_dim, args.seq_length, args.num_nodes, 
            # engine = trainer([scaler_F,scaler_E], args.in_dim, 1000, args.num_nodes, # predict shorter
                             args.nhid, args.dropout, args.learning_rate, args.weight_decay, device, 
                             supports, args.gcn_bool, args.addaptadj, adjinit, args.kernel_size,
                             args.blocks, args.layers, out_nodes=eeg_mat.shape[-1], F_t=F_t,
                             subsample=subsample, F_only=F_only, batch_size=args.batch_size)

        if model_name is None or finetune is True:
            if finetune is True:
                pretrained_dict = torch.load(model_name)
                # del pretrained_dict['end_module_add.1.weight']
                # del pretrained_dict['end_module_add.1.bias']
                # del pretrained_dict['end_mlp_e.1.weight'] 
                # del pretrained_dict['end_mlp_e.1.bias']
                # del pretrained_dict['end_mlp_f.1.weight']
                # del pretrained_dict['end_mlp_f.1.bias']
                
                model_dict = engine.model.state_dict()
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict) 
                engine.model.load_state_dict(model_dict)

            print("start training...",flush=True)

            his_loss =[]
            val_time = []
            train_time = []
            min_loss = float('Inf')
            grads = []

            sample_idx = np.arange(nTrain*sample_per_suj)
            subj_idx = np.arange(nTrain).repeat(sample_per_suj)
            within_subj_idx = np.arange(sample_per_suj)[None,...].repeat(nTrain,0).flatten()

            for i in range(1,args.epochs+1):
                if i % 3 == 0:
                    # lr = max(0.000002,args.learning_rate * (0.1 ** (i // 10)))
                    for g in engine.optimizer.param_groups:
                        g['lr'] *= 0.9
                train_loss = []
                train_mae = []
                train_mape = []
                train_rmse = []
                t1 = time.time()
                engine.set_state('train')

                iter = 0

                tmp = list(zip(sample_idx, subj_idx, within_subj_idx))
                np.random.shuffle(tmp)
                sample_idx, subj_idx, within_subj_idx = zip(*tmp)
                
                for batch_i in range(len(sample_idx)//args.batch_size):
                    subj_id = subj_idx[batch_i * args.batch_size: (batch_i + 1) * args.batch_size]
                    within_subj_id = within_subj_idx[batch_i * args.batch_size: (batch_i + 1) * args.batch_size]
                    choice = list(zip(subj_id, within_subj_id))

                    # input current F
                    x_F = np.concatenate([fmri_mat[:,F_idxer,:][:,:-offset][c][None,...] for c in choice])
                    x_F = torch.Tensor(x_F[...,None]).to(device).transpose(1, 3)
                    # pred future F
                    y_F = np.concatenate([fmri_mat[:,F_idxer,:][:,offset:][c][None,...] for c in choice])
                    y_F = torch.Tensor(y_F[...,None])#.transpose(1, 3)
                    
                    if F_only:
                        y_E = None
                    else:
                        if map:
                            # map to current E
                            y_E = np.concatenate([eeg_mat[:,E_idxer,:][:,:-offset][c][None,...] for c in choice])
                            
                            # ### test random performance (within same subject)
                            # y_E = eeg_mat[np.random.randint(eeg_mat.shape[0]), E_idxer, :][:-offset][
                            #             batch_i * args.batch_size: (batch_i + 1) * args.batch_size]
                            # np.random.shuffle(y_E)
                        else:
                            # pred future E
                            y_E = np.concatenate([eeg_mat[:,E_idxer,:][:,offset:][c][None,...] for c in choice])
                                     
                        if scatter:
                            trainy = scattering(y_E.transpose(0,2,1))#[...,order1[0],:] # order0 only
                            # trainy = scattering(y_E.transpose(0,2,1)[...,:1000])#[...,order0[0],:] # predict shorter
                            # # scale 
                            # trainy = ((trainy - mean)/std)
                            # plt.figure()
                            # plt.plot(trainy[0,0,0])
                            trainy = torch.Tensor(trainy).to(device)

                            sigy = y_E.transpose(0, 2, 1)
                            # sigy = y_E.transpose(0, 2, 1)[...,:1000] # predict shorter
                            # plt.figure()
                            # plt.plot(sigy[0,0])
                            # plt.show()
                            # ipdb.set_trace()
                            sigy = torch.Tensor(sigy).to(device)
                            y_E = [sigy, trainy]             
                            # y_E = scattering(y_E.transpose(0,2,1))  #(16, 64, 42, 45)
                            # # y_E[:,:,order0] *= 1000
                            # # y_E[:,:,order1] *= 10000
                            # # y_E[:,:,order2] *= 100000
                            # # y_E = y_E.reshape(*y_E.shape[:-2],-1) #(16, 1, 64, 1890)
                            # # y_E = torch.Tensor(y_E[:,None,...])

                            # # standardize coeff
                            # y_E = ((y_E - mean)/std)[:,:,order0[0]] #TODO:order0 for now
                            # y_E = torch.Tensor(y_E)

                        else:
                            y_E = torch.Tensor(y_E[...,None]).transpose(1, 3)
                            # y_E = torch.Tensor(y_E[...,None]).transpose(1, 3)[...,:1000] # predict shorter
                            
                            if subsample:
                                _y_E = []
                                # use averaged E
                                for y_i in range(int(y_E.shape[-1]/F_t)):
                                    _y_E.append(y_E[:,:,:,round(y_i*F_t): round((y_i+1)*F_t)].mean(-1))
                                # use subsampled E (use the mid point of each period)
                                # for y_i in range(int(y_E.shape[-1]/F_t)):
                                #     _y_E.append(y_E[:,:,:,int((round(y_i*F_t)+round((y_i+1)*F_t))//2)])
                                y_E = torch.stack(_y_E, -1)

                    metrics = engine.train_CRASH(x_F, y_F, y_E, region_assignment, np.asarray(subj_id))

                    train_loss.append(metrics[0])
                    train_mae.append(metrics[1])
                    train_mape.append(metrics[2])
                    train_rmse.append(metrics[3])
                    # grads.append(metrics[4])
                    if iter % args.print_every == 0 :
                        log = 'Iter: {:03d}, Train Loss: {:.6f}, Train MAE: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                        print(log.format(iter, train_loss[-1], train_mae[-1], train_mape[-1], train_rmse[-1]),flush=True)
                    iter += 1
                    # break # overfit single batch

                t2 = time.time()
                train_time.append(t2-t1)

                #validation
                valid_loss = []
                valid_mae = []
                valid_mape = []
                valid_rmse = []
                valid_cc = []

                s1 = time.time()
                engine.set_state('val')
                # engine.set_state('train') # overfit single batch
                for subj_id in range(nValid):
                    # for F&E: nTrain + subj_id; for adj_idx: subj_id (supports[state] counts from 0)
                    # subj_F = scaler_F.transform(fmri_mat[nTrain + subj_id, F_idxer, :])
                    subj_F = fmri_mat[nTrain + subj_id, F_idxer, :]
                    # E is only for outputs
                    # subj_E =  scaler_E.transform(eeg_mat[nTrain + subj_id, E_idxer, :][offset:]) 
                    if map:
                        subj_E = eeg_mat[nTrain + subj_id, E_idxer, :][:-offset]
                    else:
                        subj_E = eeg_mat[nTrain + subj_id, E_idxer, :][offset:]
                    
                    # ####### overfit single batch
                    # subj_F = fmri_mat[subj_id, F_idxer, :] 
                    # if map:
                    #     subj_E = eeg_mat[subj_id, E_idxer, :][:-offset]
                    # else:
                    #     subj_E = eeg_mat[subj_id, E_idxer, :][offset:]
                    # #######

                    for batch_i in range(batch_per_sub):
                        x_F = subj_F[:-offset][batch_i * args.batch_size: (batch_i + 1) * args.batch_size]
                        x_F = torch.Tensor(x_F[...,None]).to(device).transpose(1, 3)
                        
                        y_F = subj_F[offset:][batch_i * args.batch_size: (batch_i + 1) * args.batch_size]
                        y_F = torch.Tensor(y_F[...,None])#.transpose(1, 3)
                        
                        if F_only:
                            y_E = None
                        else:
                            y_E = subj_E[batch_i * args.batch_size: (batch_i + 1) * args.batch_size]
      
                            if scatter:
                                testy = scattering(y_E.transpose(0,2,1))#[...,order1[0],:] # order0 only
                                # testy = scattering(y_E.transpose(0,2,1)[...,:1000])#[...,order0[0],:] # predict shorter
                                # # scale 
                                # testy = ((testy - mean)/std)
                                testy = torch.Tensor(testy).to(device)

                                sigy = torch.Tensor(y_E).to(device)
                                sigy = sigy.transpose(1, 2)
                                # sigy = sigy.transpose(1, 2)[...,:1000] # predict shorter
                                y_E = [sigy, testy]
                            
                            else:
                                y_E = torch.Tensor(y_E[...,None]).transpose(1, 3)
                                # y_E = torch.Tensor(y_E[...,None]).transpose(1, 3)[...,:1000] # predict shorter

                                if subsample:
                                    _y_E = []
                                    # use averaged E
                                    for y_i in range(int(y_E.shape[-1]/F_t)):
                                        _y_E.append(y_E[:,:,:,round(y_i*F_t): round((y_i+1)*F_t)].mean(-1))
                                    # use subsampled E (use the mid point of each period)
                                    # for y_i in range(int(y_E.shape[-1]/F_t)):
                                    #     _y_E.append(y_E[:,:,:,int((round(y_i*F_t)+round((y_i+1)*F_t))//2)])
                                    y_E = torch.stack(_y_E, -1)

                        if subj_id == 0 and batch_i == 0: # only viz the first one
                            metrics = engine.eval_CRASH(x_F, y_F, y_E, region_assignment,
                                                        [subj_id]*args.batch_size, viz=False)
                        else:
                            metrics = engine.eval_CRASH(x_F, y_F, y_E, region_assignment,
                                                        [subj_id]*args.batch_size)                            

                        valid_loss.append(metrics[0])
                        valid_mae.append(metrics[1])
                        valid_mape.append(metrics[2])
                        valid_rmse.append(metrics[3])
                        valid_cc.append(metrics[4])

                    # plt.plot(metrics[-1][...,order0,:][0,0,0,0].cpu().numpy())
                    # plt.show()
                    #     break # overfit single batch
                    # break

                s2 = time.time()
                log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
                print(log.format(i,(s2-s1)))
                val_time.append(s2-s1)

                mtrain_loss = np.mean(train_loss)
                mtrain_mae = np.mean(train_mae)
                mtrain_mape = np.mean(train_mape)
                mtrain_rmse = np.mean(train_rmse)

                mvalid_loss = np.mean(valid_loss)
                mvalid_mae = np.mean(valid_mae)
                mvalid_mape = np.mean(valid_mape)
                mvalid_rmse = np.mean(valid_rmse)
                mvalid_cc = np.mean(valid_cc)
                his_loss.append(mvalid_loss)

                log = 'Epoch: {:03d}, Train Loss: {:.6f}, Train MAE: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAE: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Valid CC: {:.4f}, Training Time: {:.4f}/epoch'
                print(log.format(i, mtrain_loss, mtrain_mae, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mae, mvalid_mape, mvalid_rmse, mvalid_cc, (t2 - t1)),flush=True)
                # only save the best
                if mvalid_loss < min_loss:
                    min_loss = mvalid_loss
                    # # remove previous
                    # fname = glob.glob(args.save+'_epoch_*.pth')
                    # for f in fname:
                    #     os.remove(f)
                    # save new
                    torch.save(engine.model.state_dict(), args.save+"_epoch_"+str(i)+"_"+str(round(mvalid_loss,2))+".pth")
                # torch.save(engine.model.state_dict(), args.save+"_epoch_"+str(i)+"_"+str(round(mvalid_loss,2))+".pth")
            print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
            print("Average Inference Time: {:.4f} secs".format(np.mean(val_time))) 
            # ipdb.set_trace() #tmp1 = [i[0] for i in grads]

    elif args.data == 'syn' and not same_G: # different graph structure for each sample
        assert len(adj_mx) == nTrain + nValid + nTest

        # separate adj matrices into train-val-test samples
        num_adj = len(adj_mx[0]) # number of adj mx per sample
        adj_train = [[] for i in range(num_adj)]
        for a in adj_mx[:nTrain]:
            for adj_i in range(num_adj):
                adj_train[adj_i].append(a[adj_i])
        adj_train = [np.stack(np.asarray(i)) for i in adj_train]

        adj_val = [[] for i in range(num_adj)]
        for a in adj_mx[nTrain:-nTest]:
            for adj_i in range(num_adj):
                adj_val[adj_i].append(a[adj_i])
        adj_val = [np.stack(np.asarray(i)) for i in adj_val]

        adj_test = [[] for i in range(num_adj)]
        for a in adj_mx[-nTest:]:
            for adj_i in range(num_adj):
                adj_test[adj_i].append(a[adj_i])
        adj_test = [np.stack(np.asarray(i)) for i in adj_test]
        
        print(args)
        supports = {}
        supports['train'] = adj_train
        supports['val'] = adj_val
        supports['test'] = adj_test

        adjinit = {}
        if args.randomadj:
            adjinit['train'] = adjinit['val'] = adjinit['test'] = None
        else:
            adjinit['train'] = supports['train'][0]
            adjinit['val'] = supports['val'][0]
            adjinit['test'] = supports['test'][0]

        if args.aptonly:
            supports['train'] = supports['val'] = supports['test'] = None

        if scatter:
            engine = trainer(coeffs_scaler, args.in_dim, args.seq_length, args.num_nodes, 
                             args.nhid, args.dropout, args.learning_rate, args.weight_decay, device, 
                             supports, args.gcn_bool, args.addaptadj, adjinit, args.kernel_size,
                             args.blocks, args.layers, out_nodes=5, F_t=F_t,
                             meta=[order0[0],order1[0],order2[0]], scatter=True)
        else:
            engine = trainer(dataloader['scaler'], args.in_dim, args.seq_length, args.num_nodes, 
                             args.nhid, args.dropout, args.learning_rate, args.weight_decay, device, 
                             supports, args.gcn_bool, args.addaptadj, adjinit, args.kernel_size,
                             args.blocks, args.layers, out_nodes=5, F_t=F_t)
        #TODO: out_node should be the graphOptions['nCommunities'] value in Utils/util.py for syn

        if model_name is None:
            print("start training...",flush=True)

            his_loss =[]
            val_time = []
            train_time = []
            min_loss = float('Inf')
            lr = args.learning_rate

            # grad_start = []
            # grad_end = []
            for i in range(1,args.epochs+1):
                if i % 10 == 0:
                    # lr = max(0.000002,args.learning_rate * (0.1 ** (i // 10)))
                    lr = max(0.000002, lr * 0.9)
                    for g in engine.optimizer.param_groups:
                        g['lr'] = lr
                train_loss = []
                train_mae = []
                train_mape = []
                train_rmse = []

                t1 = time.time()
                dataloader['train_loader'].shuffle() # (turned off for) overfit single batch
                engine.set_state('train')
                for iter, (x, y, adj_idx) in enumerate(dataloader['train_loader'].get_iterator()):
                    # # overfit single batch
                    # if iter > 0:
                    #     break
                    # ipdb.set_trace() #x:(32, 5, 80, 1) y:(32, 120, 5, 1)
                    trainx = torch.Tensor(x).to(device) # [batch_size,5,80,1]
                    trainx = trainx.transpose(1, 3) # [batch_size, 1, 80, 5]
                    if scatter:
                        trainy = scattering(y.transpose(0,3,2,1))#[...,order1[0],:] # order0 only
                        # trainy = scattering(y.transpose(0,3,2,1)[...,:10])#[...,order0[0],:] # predict shorter
                        # # scale 
                        # trainy = ((trainy - mean)/std)
                        trainy = torch.Tensor(trainy).to(device)

                        sigy = torch.Tensor(y).to(device)
                        sigy = sigy.transpose(1, 3)
                        # sigy = sigy.transpose(1, 3)[...,:10] # predict shorter
                        metrics = engine.train_syn(trainx, [sigy, trainy], G['train'], adj_idx)
                    else:
                        trainy = torch.Tensor(y).to(device)
                        trainy = trainy.transpose(1, 3)
                        metrics = engine.train_syn(trainx, trainy, G['train'], adj_idx)

                    train_loss.append(metrics[0])
                    train_mae.append(metrics[1])
                    train_mape.append(metrics[2])
                    train_rmse.append(metrics[3])

                    # grad_start.append(metrics[-2].cpu().numpy())
                    # grad_end.append(metrics[-1].cpu().numpy())

                    if iter % args.print_every == 0 :
                        log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAE: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                        print(log.format(iter, train_loss[-1], train_mae[-1], train_mape[-1], train_rmse[-1]),flush=True)
                t2 = time.time()
                train_time.append(t2-t1)

                #validation
                valid_loss = []
                valid_mae = []
                valid_mape = []
                valid_rmse = []
                valid_cc = []

                s1 = time.time()
                engine.set_state('val')
                # engine.set_state('train') # overfit single batch
                for iter, (x, y, adj_idx) in enumerate(dataloader['val_loader'].get_iterator()):
                # # overfit single batch
                # for iter, (x, y, adj_idx) in enumerate(dataloader['train_loader'].get_iterator()):                    
                #     if iter > 0:
                #         break
                    testx = torch.Tensor(x).to(device)
                    testx = testx.transpose(1, 3)
                    if scatter:
                        testy = scattering(y.transpose(0,3,2,1))#[...,order1[0],:] # order0 only
                        # # scale 
                        # testy = ((testy - mean)/std)
                        testy = torch.Tensor(testy).to(device)

                        sigy = torch.Tensor(y).to(device)
                        sigy = sigy.transpose(1, 3)
                        metrics = engine.eval_syn(testx, [sigy, testy], G['val'], adj_idx)
                    else:
                        testy = torch.Tensor(y).to(device)
                        testy = testy.transpose(1, 3)
                        if (i % 30 == 0) and iter == 0:
                            metrics = engine.eval_syn(testx, testy, G['val'], adj_idx, viz=True)
                        else:
                            metrics = engine.eval_syn(testx, testy, G['val'], adj_idx)

                    valid_loss.append(metrics[0])
                    valid_mae.append(metrics[1])
                    valid_mape.append(metrics[2])
                    valid_rmse.append(metrics[3])
                    valid_cc.append(metrics[4])

                s2 = time.time()
                log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
                print(log.format(i,(s2-s1)))
                val_time.append(s2-s1)
                mtrain_loss = np.mean(train_loss)
                mtrain_mae = np.mean(train_mae)
                mtrain_mape = np.mean(train_mape)
                mtrain_rmse = np.mean(train_rmse)

                mvalid_loss = np.mean(valid_loss)
                mvalid_mae = np.mean(valid_mae)
                mvalid_mape = np.mean(valid_mape)
                mvalid_rmse = np.mean(valid_rmse)
                mvalid_cc = np.mean(valid_cc)
                his_loss.append(mvalid_loss)

                log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAE: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAE: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Valid CC: {:.4f}, Training Time: {:.4f}/epoch'
                print(log.format(i, mtrain_loss, mtrain_mae, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mae, mvalid_mape, mvalid_rmse, mvalid_cc, (t2 - t1)),flush=True)
                if mvalid_loss < min_loss:
                    min_loss = mvalid_loss
                    # # remove previous
                    # fname = glob.glob(args.save+'_epoch_*.pth')
                    # for f in fname:
                    #     os.remove(f)
                    # save new
                    # torch.save(engine.model.state_dict(), args.save+"_epoch_"+str(i)+"_"+str(round(mvalid_loss,2))+".pth")
                torch.save(engine.model.state_dict(), args.save+"_epoch_"+str(i)+"_"+str(round(mvalid_loss,2))+".pth")
            print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
            print("Average Inference Time: {:.4f} secs".format(np.mean(val_time))) 
            # ipdb.set_trace()

    else:
        scaler = dataloader['scaler']
        supports = [torch.tensor(i).to(device) for i in adj_mx]
        print(args)

        if args.randomadj:
            adjinit = None
        else:
            adjinit = supports[0]

        if args.aptonly:
            supports = None

        engine = trainer(scaler, args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                        args.learning_rate, args.weight_decay, device, supports, args.gcn_bool, 
                        args.addaptadj, adjinit, args.kernel_size, args.blocks, args.layers, F_t=F_t)

        if model_name is None:
            print("start training...",flush=True)
            his_loss =[]
            val_time = []
            train_time = []
            min_loss = float('Inf')
            for i in range(1,args.epochs+1):
                #if i % 10 == 0:
                    #lr = max(0.000002,args.learning_rate * (0.1 ** (i // 10)))
                    #for g in engine.optimizer.param_groups:
                        #g['lr'] = lr
                train_loss = []
                train_mae = []
                train_mape = []
                train_rmse = []
                t1 = time.time()
                dataloader['train_loader'].shuffle()
                for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
                    trainx = torch.Tensor(x).to(device) # torch.Size([64, 12, 207, 2])
                    trainx= trainx.transpose(1, 3) # torch.Size([64, 2, 207, 12])
                    trainy = torch.Tensor(y).to(device)
                    trainy = trainy.transpose(1, 3)
                    if args.data == 'syn':
                        metrics = engine.train_syn(trainx, trainy, G)
                    else:
                        metrics = engine.train(trainx, trainy[:,0,:,:])
                    train_loss.append(metrics[0])
                    train_mae.append(metrics[1])
                    train_mape.append(metrics[2])
                    train_rmse.append(metrics[3])
                    if iter % args.print_every == 0 :
                        log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAE: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                        print(log.format(iter, train_loss[-1], train_mae[-1], train_mape[-1], train_rmse[-1]),flush=True)
            
                t2 = time.time()
                train_time.append(t2-t1)
                #validation
                valid_loss = []
                valid_mae = []
                valid_mape = []
                valid_rmse = []

                s1 = time.time()
                for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
                    testx = torch.Tensor(x).to(device)
                    testx = testx.transpose(1, 3)
                    testy = torch.Tensor(y).to(device)
                    testy = testy.transpose(1, 3)
                    if args.data == 'syn':
                        metrics = engine.eval_syn(testx, testy, G)
                    else:
                        metrics = engine.eval(testx, testy[:,0,:,:])
                    valid_loss.append(metrics[0])
                    valid_mae.append(metrics[1])
                    valid_mape.append(metrics[2])
                    valid_rmse.append(metrics[3])

                s2 = time.time()
                log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
                print(log.format(i,(s2-s1)))
                val_time.append(s2-s1)
                mtrain_loss = np.mean(train_loss)
                mtrain_mae = np.mean(train_mae)
                mtrain_mape = np.mean(train_mape)
                mtrain_rmse = np.mean(train_rmse)

                mvalid_loss = np.mean(valid_loss)
                mvalid_mae = np.mean(valid_mae)
                mvalid_mape = np.mean(valid_mape)
                mvalid_rmse = np.mean(valid_rmse)
                his_loss.append(mvalid_loss)

                log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAE: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAE: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
                print(log.format(i, mtrain_loss, mtrain_mae, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mae, mvalid_mape, mvalid_rmse, (t2 - t1)),flush=True)
                if mvalid_loss < min_loss:
                    min_loss = mvalid_loss
                    # # remove previous
                    # fname = glob.glob(args.save+'_epoch_*.pth')
                    # for f in fname:
                    #     os.remove(f)
                    # save new                    
                    torch.save(engine.model.state_dict(), args.save+"_epoch_"+str(i)+"_"+str(round(mvalid_loss,2))+".pth")
                # torch.save(engine.model.state_dict(), args.save+"_epoch_"+str(i)+"_"+str(round(mvalid_loss,2))+".pth")
            print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
            print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))


    ################################ TESTING ################################
    if model_name is None:
        bestid = np.argmin(his_loss)
        print(bestid)

        print("Training finished")
        print("The valid loss on best model is", str(round(his_loss[bestid],4)))

        engine.model.load_state_dict(torch.load(args.save+"_epoch_"+str(bestid+1)+"_"+str(
                                                        round(his_loss[bestid],2))+".pth"))
    else:
        pretrained_dict = torch.load(model_name)
        engine.model.load_state_dict(pretrained_dict)
    amae = []
    amape = []
    armse = []
    a_cc = []

    if args.data == 'CRASH':
        engine.set_state('test')
        # engine.set_state('train') # overfit single batch
        real_Fs = []
        real_Es = []
        pred_Fs = []
        pred_Es = []
        pred_coeffs = []
        for subj_id in range(nTest):
            # for F&E: nTrain + nValid + subj_id; for adj_idx: subj_id (supports[state] counts from 0)
            # subj_F = scaler_F.transform(fmri_mat[nTrain + nValid + subj_id, F_idxer, :])
            subj_F = fmri_mat[nTrain + nValid + subj_id, F_idxer, :]
            # E is only for outputs
            # subj_E =  scaler_E.transform(eeg_mat[nTrain + nValid + subj_id, E_idxer, :][offset:])
            if map:
                subj_E =  eeg_mat[nTrain + nValid + subj_id, E_idxer, :][:-offset]
            else:
                subj_E =  eeg_mat[nTrain + nValid + subj_id, E_idxer, :][offset:]

            # ####### overfit single batch
            # subj_F = fmri_mat[subj_id, F_idxer, :] 
            # if map:
            #     subj_E = eeg_mat[subj_id, E_idxer, :][:-offset]
            # else:
            #     subj_E = eeg_mat[subj_id, E_idxer, :][offset:]
            # ######

            for batch_i in range(batch_per_sub):
                x_F = subj_F[:-offset][batch_i * args.batch_size: (batch_i + 1) * args.batch_size]
                x_F = torch.Tensor(x_F[...,None]).to(device).transpose(1, 3)
                
                y_F = subj_F[offset:][batch_i * args.batch_size: (batch_i + 1) * args.batch_size]
                y_F = torch.Tensor(y_F[...,None])#.transpose(1, 3)
                y_E = subj_E[batch_i * args.batch_size: (batch_i + 1) * args.batch_size]

                if scatter:                            
                    testy = scattering(y_E.transpose(0,2,1))#[...,order1[0],:] # order0 only
                    # testy = scattering(y_E.transpose(0,2,1)[...,:1000]) # predict shorter
                    # # scale 
                    # testy = ((testy - mean)/std)
                    testy = torch.Tensor(testy).to(device)

                    sigy = torch.Tensor(y_E).to(device)
                    sigy = sigy.transpose(1, 2)
                    # sigy = sigy.transpose(1, 2)[...,:1000] # predict shorter
                    y_E = [sigy, testy]

                    # sig = y_E.transpose(0,2,1)
                    # y_E = scattering(sig)
                    # # y_E[:,:,order0] *= 1000
                    # # y_E[:,:,order1] *= 10000
                    # # y_E[:,:,order2] *= 100000

                    # # y_E = y_E.reshape(*y_E.shape[:-2],-1)
                    # # y_E = torch.Tensor(y_E[:,None,...])
                    
                    # # standardize coeff
                    # y_E = ((y_E - mean)/std)[:,:,order0[0]] #TODO:order0 for now
                    # y_E = torch.Tensor(y_E)
                else:
                    y_E = torch.Tensor(y_E[...,None]).transpose(1, 3)
                    # y_E = torch.Tensor(y_E[...,None]).transpose(1, 3)[...,:1000] # predict shorter
                    if subsample:
                        _y_E = []
                        # use averaged E
                        for y_i in range(int(y_E.shape[-1]/F_t)):
                            _y_E.append(y_E[:,:,:,round(y_i*F_t): round((y_i+1)*F_t)].mean(-1))
                        # use subsampled E (use the mid point of each period)
                        # for y_i in range(int(y_E.shape[-1]/F_t)):
                        #     _y_E.append(y_E[:,:,:,int((round(y_i*F_t)+round((y_i+1)*F_t))//2)])
                        y_E = torch.stack(_y_E, -1)

                metrics = engine.eval_CRASH(x_F, y_F, y_E, region_assignment, 
                                            [subj_id]*args.batch_size)

                amae.append(metrics[1])
                amape.append(metrics[2])
                armse.append(metrics[3])

                real_Fs.append(y_F)
                real_Es.append(y_E)
                pred_Fs.append(metrics[-3])
                pred_Es.append(metrics[-2])
                pred_coeffs.append(metrics[-1])

                if batch_i == 0:
                    if scatter:
                        # for in-network scatter checking
                        # ipdb.set_trace()
                        plt.figure('sig')
                        plt.plot(real_Es[0][0].squeeze().cpu().numpy()[1,1], label='real')
                        plt.plot(pred_Es[0].squeeze().cpu().numpy()[1,1], label='pred')
                        plt.legend()
                        # plt.savefig('sig.png')

                        plt.figure('coeff')
                        # plt.plot(real_Es[0][1].squeeze().cpu().numpy()[0,0], label='real')
                        # plt.plot(pred_coeffs[0].squeeze().cpu().numpy()[0,0], label='pred')
                        plt.plot(real_Es[0][1].squeeze().cpu().numpy()[1,1,0], label='real')
                        plt.plot(pred_coeffs[0].squeeze().cpu().numpy()[1,1,0], label='pred')
                        plt.legend()
                        # plt.savefig('coeff.png')
                        plt.show()
                        ipdb.set_trace()   

                    else:
                        if F_only:
                            plt.figure(0)
                            plt.plot(real_Fs[0].squeeze().cpu().numpy()[0,0], label='real Fs')
                            plt.plot(pred_Fs[0].squeeze().cpu().numpy()[0,0], label='pred Fs')
                            plt.legend()
                            plt.figure(2)
                            plt.plot(real_Fs[0].squeeze().cpu().numpy()[0,2], label='real Fs')
                            plt.plot(pred_Fs[0].squeeze().cpu().numpy()[0,2], label='pred Fs')
                            plt.legend()
                            plt.show()
                        else:
                            plt.figure()
                            plt.plot(real_Es[0].squeeze().cpu().numpy()[0,0], label='real Es')
                            plt.plot(pred_Es[0].squeeze().cpu().numpy()[0,0], label='pred Es')
                            plt.legend()
                            plt.show()                            
                        ipdb.set_trace()

        real_Fs = torch.stack(real_Fs).cpu().numpy()
        real_Fs = real_Fs.reshape(-1, *real_Fs.shape[2:])
        real_Es = torch.stack(real_Es).cpu().numpy()
        real_Es = real_Es.reshape(-1, *real_Es.shape[2:])

        pred_Fs = [pred_F.cpu().numpy() for pred_F in pred_Fs]
        pred_Fs = np.stack(pred_Fs)
        pred_Fs = pred_Fs.reshape(-1, *pred_Fs.shape[2:]).squeeze()

        pred_Es = [pred_E.cpu().numpy() for pred_E in pred_Es]
        pred_Es = np.stack(pred_Es)
        pred_Es = pred_Es.reshape(-1, *pred_Es.shape[2:]).squeeze()

        # reals shape: (1984, 2, 80, 15); pred_F/Es shape:(1984, 80, 15)

        # reverse slideing window --> results: (num_nodes, total_timesteps)
        # ret = util.inverse_sliding_window([reals[:, 0, :, :].squeeze(), pred_Es])
        # concatenate
        real_Fs = real_Fs.squeeze().transpose(1,0,2)
        real_Fs = real_Fs.reshape((len(real_Fs), -1))
        real_Es = real_Es.squeeze().transpose(1,0,2)
        real_Es = real_Es.reshape((len(real_Es), -1))

        pred_Fs = pred_Fs.transpose(1,0,2)
        pred_Fs = pred_Fs.reshape((len(pred_Fs),-1))
        pred_Es = pred_Es.transpose(1,0,2)
        pred_Es = pred_Es.reshape((len(pred_Es),-1))

        for viz_node_idx in range(64):
            viz_num = 5000 # time length of visualization
            # plt.figure()
            # plt.plot(real_Fs[viz_node_idx, :viz_num], label='real F')
            # plt.plot(pred_Fs[viz_node_idx, :viz_num], label='pred F')
            # plt.legend()
            # plt.savefig('nodes_rst/'+str(viz_node_idx)+'_F.png')

            plt.figure()
            if subsample:
                plt.plot(real_Es[viz_node_idx, :int(viz_num*F_t)], label='real E')
                plt.plot(pred_Es[viz_node_idx, :int(viz_num*F_t)], label='pred E')
            else:
                plt.plot(real_Es[viz_node_idx, :viz_num], label='real E')
                plt.plot(pred_Es[viz_node_idx, :viz_num], label='pred E')
            plt.legend()
            plt.savefig('nodes_rst/'+str(viz_node_idx)+'_E.png')

        ipdb.set_trace()

        viz_node_idx = 0
        viz_num = 5000 # time length of visualization
        plt.figure()
        plt.plot(real_Fs[viz_node_idx, :viz_num], label='real F')
        plt.plot(pred_Fs[viz_node_idx, :viz_num], label='pred F')
        plt.legend()

        plt.figure()
        if subsample:
            plt.plot(real_Es[viz_node_idx, :int(viz_num*F_t)], label='real E')
            plt.plot(pred_Es[viz_node_idx, :int(viz_num*F_t)], label='pred E')
        else:
            plt.plot(real_Es[viz_node_idx, :viz_num], label='real E')
            plt.plot(pred_Es[viz_node_idx, :viz_num], label='pred E')
        plt.legend()
        plt.show()

        log = 'On average over seq_length horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(np.mean(amae),np.mean(amape),np.mean(armse)))
        
        if model_name is None:
            torch.save(engine.model.state_dict(), args.save+"_exp"+str(args.expid)+"_best_"+str(round(his_loss[bestid],2))+".pth")
    
    elif args.data == 'syn':
        if same_G:
            for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
                testx = torch.Tensor(x).to(device)
                testx = testx.transpose(1, 3)
                testy = torch.Tensor(y).to(device)
                testy = testy.transpose(1, 3)
                # [64, 2, 80, 15]
                metrics = engine.eval_syn(testx, testy, G)
                amae.append(metrics[1])
                amape.append(metrics[2])
                armse.append(metrics[3])

        else:
            engine.set_state('test')
            # engine.set_state('train') # overfit single batch
            in_Fs = []
            reals = []
            # pred_Fs = []
            if scatter:
                sigs_pred = []
                sigs_real = []
            preds = []
            for iter, (x, y, adj_idx) in enumerate(dataloader['test_loader'].get_iterator()):
            # overfit single batch
            # for iter, (x, y, adj_idx) in enumerate(dataloader['train_loader'].get_iterator()):                    
            #     if iter > 0:
            #         break    
                testx = torch.Tensor(x).to(device)
                testx = testx.transpose(1, 3)
                if scatter:
                    testy = scattering(y.transpose(0,3,2,1))#[...,order1[0],:] # order0 only
                    # # scale 
                    # testy = ((testy - mean)/std)
                    testy = torch.Tensor(testy).to(device)

                    sigy = torch.Tensor(y).to(device)
                    sigy = sigy.transpose(1, 3)
                    metrics = engine.eval_syn(testx, [sigy, testy], G['val'], adj_idx)

                    sigs_real.append(y)
                    sigs_pred.append(metrics[-2])

                else:
                    testy = torch.Tensor(y).to(device)
                    testy = testy.transpose(1, 3)
                    metrics = engine.eval_syn(testx, testy, G['val'], adj_idx)

                amae.append(metrics[1])
                amape.append(metrics[2])
                armse.append(metrics[3])
                a_cc.append(metrics[4])

                in_Fs.append(testx)
                reals.append(testy)
                preds.append(metrics[-1])
                
                if iter == 0:
                    if scatter:
                        # for in-network scatter checking
                        plt.figure()
                        # plt.plot(sigs_real[0].transpose(0,3,2,1).squeeze()[1,1][...,:10], label='real')
                        plt.plot(sigs_real[0].transpose(0,3,2,1).squeeze()[0,0], label='real')
                        plt.plot(sigs_pred[0].cpu().numpy().squeeze()[0,0], label='pred')
                        plt.legend()
                        plt.savefig('sigs.png')
                        # coeffs
                        plt.figure()
                        plt.plot(reals[0].cpu().numpy().squeeze()[0,0].flatten(), label='real')
                        plt.plot(preds[0].cpu().numpy()[0,0].flatten(), label='pred')
                        plt.legend()
                        plt.savefig('coeff.png')
                        ipdb.set_trace()
                    else:
                        # checking the first prediction
                        plt.figure()
                        plt.plot(reals[0].cpu().numpy().squeeze()[1,1], label='real')
                        plt.plot(preds[0].cpu().numpy()[1,1], label='pred')
                        plt.legend()
                        plt.show()


            if scatter:
                in_Fs = torch.stack(in_Fs).cpu().numpy()
                in_Fs = in_Fs.reshape(-1, *in_Fs.shape[2:]).squeeze()

                sigs_real = np.stack(sigs_real).squeeze().transpose(0,1,3,2)
                sigs_real = sigs_real.reshape(-1, *sigs_real.shape[2:]).squeeze()
                sigs_pred = torch.stack(sigs_pred).cpu().numpy()
                sigs_pred = sigs_pred.reshape(-1, *sigs_pred.shape[2:]).squeeze()

                # reverse slideing window --> results: (num_nodes, total_timesteps)
                ret = util.inverse_sliding_window([in_Fs, sigs_real, sigs_pred], [1]+[F_t]*2)
                viz_node_idx = 1
                plt.figure()
                plt.plot(ret[0][viz_node_idx, :].repeat(F_t)[:600], label='in F')
                plt.plot(ret[1][viz_node_idx, :][:600], label='real E')
                plt.plot(ret[2][viz_node_idx, :][:600], label='pred E')
                plt.legend()
                plt.show()
                ipdb.set_trace()           
            else:
                in_Fs = torch.stack(in_Fs).cpu().numpy()
                in_Fs = in_Fs.reshape(-1, *in_Fs.shape[2:]).squeeze()
                reals = torch.stack(reals).cpu().numpy()
                reals = reals.reshape(-1, *reals.shape[2:]).squeeze()
                preds = torch.stack(preds).cpu().numpy()
                preds = preds.reshape(-1, *preds.shape[2:]).squeeze()

                # reverse slideing window --> results: (num_nodes, total_timesteps)
                ret = util.inverse_sliding_window([in_Fs, reals, preds], [1]+[F_t]*2)
                viz_node_idx = 1
                plt.figure()
                plt.plot(ret[0][viz_node_idx, :].repeat(F_t)[:3000], label='in F')
                plt.plot(ret[1][viz_node_idx, :][:3000], label='real E')
                plt.plot(ret[2][viz_node_idx, :][:3000], label='pred E')
                plt.legend()
                plt.show()
                plt.figure()
                plt.plot(ret[0][viz_node_idx, :].repeat(F_t)[:120], label='in F')
                plt.plot(ret[1][viz_node_idx, :][:120], label='real E')
                plt.plot(ret[2][viz_node_idx, :][:120], label='pred E')
                plt.legend()
                plt.show()
                ipdb.set_trace()

        if model_name is None:
            log = 'On average over seq_length horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}, Test CC: {:.4f}'
            print(log.format(np.mean(amae),np.mean(amape),np.mean(armse),np.mean(a_cc)))
            torch.save(engine.model.state_dict(), args.save+"_exp"+str(args.expid)+"_best_"+str(round(his_loss[bestid],2))+".pth")
    
    else:
        outputs = []
        realy = torch.Tensor(dataloader['y_test']).to(device)
        realy = realy.transpose(1,3)[:,0,:,:]

        for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1,3)
            with torch.no_grad():
                preds = engine.model(testx).transpose(1,3)
            outputs.append(preds.squeeze())

        yhat = torch.cat(outputs,dim=0)
        yhat = yhat[:realy.size(0),...]

        for i in range(args.seq_length):
            pred = scaler.inverse_transform(yhat[:,:,i])
            real = realy[:,:,i]
            metrics = util.metric(pred,real)
            log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
            print(log.format(i+1, metrics[0], metrics[1], metrics[2]))
            amae.append(metrics[2])
            amape.append(metrics[3])
            armse.append(metrics[4])

        log = 'On average over seq_length horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(np.mean(amae),np.mean(amape),np.mean(armse)))
        torch.save(engine.model.state_dict(), args.save+"_exp"+str(args.expid)+"_best_"+str(round(his_loss[bestid],2))+".pth")


if __name__ == "__main__":
    t1 = time.time()
    # main('garage/syn_epoch_95_0.1.pth', syn_file='syn_batch32_diffG.pkl', scatter=True)
    # main(syn_file='syn_batch32_diffG_map_dt.pkl', scatter=False)
    main(scatter=False, map=True, F_only=True)
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2-t1))