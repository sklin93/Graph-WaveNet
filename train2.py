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
# from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.utils import shuffle
from kymatio.numpy import Scattering1D
import ipdb
from tqdm import tqdm
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

def proc_helper(a, b, len_adj, no_scaler=False):
    assert len(a) == len(b), 'input length do not match'
    assert len(a) % len_adj == 0

    # deal with adj_mx repetitions (using idx)
    adj_mx_idx = np.arange(len_adj).repeat(len(a)//len_adj)
    # shuffle for mixing different subjects' image
    adj_mx_idx, a, b = shuffle(adj_mx_idx, a, b, random_state = 0)
    # train-val-test split
    num_total = len(a)
    nTrain = round(0.7 * num_total)
    nValid = round(0.15 * num_total)
    nTest = num_total - nTrain - nValid
    print('Train, Val, Test numbers:', nTrain, nValid, nTest)

    if no_scaler:
        return nTrain, nValid, nTest, a, b, None, None, adj_mx_idx
    
    # # overall scaler
    # scaler_a = util.StandardScaler(mean=a[:nTrain].mean(), std=a[:nTrain].std())
    # scaler_b = util.StandardScaler(mean=b[:nTrain].mean(), std=b[:nTrain].std())
    # a = scaler_a.transform(a)
    # b = scaler_b.transform(b) ### ??? different modality has some problem here..when both needs an individual scaler
    
    # # per-feature standardization using sklearn
    # scaler_a = StandardScaler()
    # scaler_a.fit(a[:nTrain].reshape(-1, a.shape[-1]))
    # scaler_b = StandardScaler()
    # scaler_b.fit(b[:nTrain].reshape(-1, b.shape[-1]))

    # per-feature standardization
    scaler_a = util.StandardScaler(a.mean((0,1)), a.std((0,1)))
    scaler_b = util.StandardScaler(b.mean((0,1)), b.std((0,1)))
    
    a = scaler_a.transform(a.reshape(-1, a.shape[-1])).reshape(a.shape)
    b = scaler_b.transform(b.reshape(-1, b.shape[-1])).reshape(b.shape)
    return nTrain, nValid, nTest, a, b, scaler_a, scaler_b, adj_mx_idx
  
def main(model_name=None, finetune=False, syn_file='syn_diffG.pkl', 
         scatter=False, subsample=False, _map=False, F_only=False):
    '''directly loading trained model/ generated syn data
       scatter: whether use wavelet scattering
       subsample: whether subsample EEG signals to the same temporal resolution as fMRI's
       map: whether map to current EEG signal or (else) predict future EEG signal
    '''
    #load data
    same_G = False
    device = torch.device(args.device)

    if args.data == 'CRASH':
        if subsample:
            basic_len = 1456 #hard-coded for 1/6 subsample
        else:
            basic_len = 2912
        CRASH_fname = 'CRASH_Fonly.pkl'
        try:
            with open(CRASH_fname, 'rb') as handle:
                F_t, adj_mx, adj_mx_idx, _input, _gt, coeffs, \
                inv_mapping, region_assignment, nTrain, nValid, \
                nTest, scaler_in, scaler_out = pickle.load(handle)

        except:
            scs, adj_mx, fmri_mat, eeg_mat, region_assignment, F_t = util.load_dataset_CRASH(args.adjtype)
            
            if subsample:
                F_t /= subsample
            K = int(args.seq_length / F_t)
            print('fMRI signal length in input-output pairs:', K)

            if False:
                import networkx as nx
                ## randomize SC entries to see the sensitivity to SC
                # use completely random SC w/ same level of sparsity
                n = args.num_nodes # number of nodes
                p = np.count_nonzero(adj_mx[0][0]) / (n*(n-1)/2) # probability for edge creation
                _G = nx.gnp_random_graph(n, p)
                # _G = nx.newman_watts_strogatz_graph(n,5,p)
                # _G = nx.gnm_random_graph(n, np.count_nonzero(adj_mx[0][0]))
                for i in range(len(adj_mx)):
                    # _G = nx.gnp_random_graph(n, p)
                    adj_mx[i] = util.mod_adj(nx.to_numpy_matrix(_G), args.adjtype)
            
            ''' plot fmri signal in freq domain '''
            # xf = np.fft.rfftfreq(fmri_mat.shape[1], d=0.91) # up to nyquist freq: 1/2*(1/0.91)
            # yf = np.zeros_like(xf)
            # for i in range(len(fmri_mat)):
            #     for j in range(fmri_mat.shape[-1]):
            #         tmp = fmri_mat[i, :, j]
            #         yf += np.abs(np.fft.rfft(tmp) / len(tmp))
                    
            # yf /= (fmri_mat.shape[0]*fmri_mat.shape[-1])
            # plt.plot(xf, yf)
            # plt.show()
            # highest_f_component = xf[np.where(yf == max(yf))[0][0]]
            # print('most fMRI f:', highest_f_component, 'Hz, aka 1/', 1/highest_f_component, 's')

            ''' low pass filter fMRI with 0.2 hz threshold '''
            cutoff = 0.2 #(1/0.91)/(2*3)
            for i in range(fmri_mat.shape[0]): #fmri_mat: (n, 320, 200)
                for j in range(fmri_mat.shape[-1]):
                    fmri_mat[i,:,j] = util.butter_lowpass_filter(fmri_mat[i,:,j], cutoff, 1/0.91)

            F_idxer = np.arange(K)[None, :] + np.arange(0, fmri_mat.shape[1] - K + 1, 
                                                        int(basic_len/F_t))[:, None]
            fmri_mat = fmri_mat[:, F_idxer,:]

            if _map: # for signal mapping
                fmri_mat = fmri_mat.reshape(-1, *fmri_mat.shape[2:])

                ''' low pass filter eeg with 50 hz threshold'''
                cutoff = 50
                for i in range(len(eeg_mat)):
                    for j in range(eeg_mat.shape[-1]):
                        eeg_mat[i,:,j] = util.butter_lowpass_filter(eeg_mat[i,:,j], cutoff, 640)
                ''' subsample eeg
                since useful info is < 50Hz, as long as sample f > 100Hz we can keep all the info
                640/6 > 100, so doing 1/6 subsample
                '''
                if subsample:
                    eeg_mat = eeg_mat[:,::subsample,:]

                E_idxer = np.arange(args.seq_length)[None, :] + np.arange(0, 
                                eeg_mat.shape[1] - args.seq_length + 1, basic_len)[:, None]
                assert len(F_idxer) == len(E_idxer)

                eeg_mat = eeg_mat[:, E_idxer, :]
                eeg_mat = eeg_mat.reshape(-1, *eeg_mat.shape[2:])

                nTrain, nValid, nTest, _input, _gt, scaler_in, scaler_out, adj_mx_idx = \
                                                proc_helper(fmri_mat, eeg_mat, len(adj_mx))
                del fmri_mat, eeg_mat, F_idxer, E_idxer
                
                # min-max normalization
                _input = (_input - _input.min()) / (_input.max() - _input.min())
                _gt = (_gt - _gt.min()) / (_gt.max() - _gt.min())        
                # _input = _input / np.max(np.abs(_input))
                # _gt = _gt / np.max(np.abs(_gt))

            else: # for fmri signal predictions
                basic_len = 2912 # hard-coded fot F_t 582.4, used as the sliding window stride to avoid float
                assert int(args.seq_length % basic_len) == 0
                offset = args.seq_length // basic_len

                sample_per_suj = len(F_idxer) - offset
                print('sample per subj', sample_per_suj)

                fmri_mat_x = fmri_mat[:,:-offset]
                fmri_mat_y = fmri_mat[:,offset:]

                fmri_mat_x = fmri_mat_x.reshape(-1, *fmri_mat_x.shape[2:])
                fmri_mat_y = fmri_mat_y.reshape(-1, *fmri_mat_y.shape[2:])
                
                nTrain, nValid, nTest, _input, _gt, scaler_in, scaler_out, adj_mx_idx = \
                                            proc_helper(fmri_mat_x, fmri_mat_y, len(adj_mx))
                ipdb.set_trace() #F_t, scs, adj_mx, adj_mx_idx, _input, _gt, nTrain, nValid, nTest, scaler_in, scaler_out
                del fmri_mat_x, fmri_mat_y

        # region_assignment: {EEG_electrodes: brain region}
        inv_mapping = {} #{brain region: EEG_electrodes}
        for k, v in region_assignment.items():
            for _v in v:
                if _v not in inv_mapping:
                    inv_mapping[_v] = []
                inv_mapping[_v] = sorted(list(set(inv_mapping[_v]+[k])))

        if scatter:
            # wavelet consts J=3 & Q=8
            J = 3
            Q = 9
            print(1+J*Q+J*(J-1)*Q/2, 1000/(2**J))
            scattering = Scattering1D(J, args.seq_length, Q)
            # scattering = Scattering1D(J, 1000, Q) # predict shorter
            meta = scattering.meta()
            order0 = np.where(meta['order'] == 0) #1*45
            order1 = np.where(meta['order'] == 1) #13*45
            order2 = np.where(meta['order'] == 2) #28*45

            if 'coeffs' not in locals():
                coeffs = []
                for i in tqdm(range(len(_gt))):
                    coeffs.append(scattering(_gt[i].transpose(1,0)))
                coeffs = np.stack(coeffs)
                
                pkl_data = (F_t, scs, adj_mx, adj_mx_idx, _input, _gt, coeffs, 
                            inv_mapping, region_assignment, nTrain, nValid, 
                            nTest, scaler_in, scaler_out)
                with open(CRASH_fname, 'wb') as handle:
                    pickle.dump(pkl_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    ipdb.set_trace()
            # # load wavelet coefficient scalers
            # with open('coeffs_scaler.pkl', 'rb') as handle:
            #     coeffs_scaler = pickle.load(handle)
            
            # # for transform: make it into same shape as y_E
            # mean = np.tile(coeffs_scaler['means'][None, None, ...], (args.batch_size, 61, 1, 1))
            # std = np.tile(coeffs_scaler['stds'][None, None, ...], (args.batch_size, 61, 1, 1))

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

        if False:
            import networkx as nx
            ## randomize SC entries to see the sensitivity to SC
            # use completely random SC w/ same level of sparsity
            n = args.num_nodes # number of nodes
            p = np.count_nonzero(adj_mx[0][0]) / (n*(n-1)/2) # probability for edge creation
            _G = nx.gnp_random_graph(n, p)
            # _G = nx.newman_watts_strogatz_graph(n,5,p)
            # _G = nx.gnm_random_graph(n, np.count_nonzero(adj_mx[0][0]))
            for i in range(len(adj_mx)):
                # _G = nx.gnp_random_graph(n, p)
                adj_mx[i] = util.mod_adj(nx.to_numpy_matrix(_G), args.adjtype)

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
        # separate adj matrices into train-val-test samples

        _adj = [[] for i in range(len(adj_mx[0]))]
        for a in adj_mx:
            for i in range(len(_adj)):
                _adj[i].append(a[i])
        adj_mx = [np.stack(np.asarray(i)) for i in _adj]
        
        print(args)
        supports = {}
        supports['train'] = supports['val'] = supports['test'] = adj_mx

        adjinit = None
        if args.addaptadj:
            adjinit = {}
            if args.randomadj:
                adjinit['train'] = adjinit['val'] = adjinit['test'] = None
            else:
                adjinit['train'] = np.concatenate([adj_mx[0][c][None,...] for c in adj_mx_idx[:nTrain]]) 
                ipdb.set_trace()# TODO: adjinit['train'] need to be shuffled with train set
                adjinit['val'] = np.concatenate([adj_mx[0][c][None,...] for c in adj_mx_idx[nTrain:-nTest]])
                adjinit['test'] = np.concatenate([adj_mx[0][c][None,...] for c in adj_mx_idx[-nTest:]])

            if args.aptonly:
                supports = None

        if scatter:
            engine = trainer([scaler_in,scaler_out], args.in_dim, args.seq_length, args.num_nodes, 
            # engine = trainer(coeffs_scaler, args.in_dim, args.seq_length, args.num_nodes, 
            # engine = trainer(coeffs_scaler, args.in_dim, 1000, args.num_nodes, # predict shorter
                             args.nhid, args.dropout, args.learning_rate, args.weight_decay, device, 
                             supports, args.gcn_bool, args.addaptadj, adjinit, args.kernel_size,
                             args.blocks, args.layers, out_nodes=_gt.shape[-1], F_t=F_t,
                             meta=[order0[0],order1[0],order2[0]],scatter=True, F_only=F_only, 
                             batch_size=args.batch_size)
        else:
            engine = trainer([scaler_in,scaler_out], args.in_dim, args.seq_length, args.num_nodes,
                             args.nhid, args.dropout, args.learning_rate, args.weight_decay, device, 
                             supports, args.gcn_bool, args.addaptadj, adjinit, args.kernel_size,
                             args.blocks, args.layers, out_nodes=_gt.shape[-1], F_t=F_t,
                             subsample=subsample, F_only=F_only, batch_size=args.batch_size)

        if model_name is None or finetune is True:
            if finetune is True:
                pretrained_dict = torch.load(model_name)
                
                model_dict = engine.model.state_dict()
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict) 
                engine.model.load_state_dict(model_dict)

            print("start training...",flush=True)

            his_loss_train = []
            his_loss_val =[]
            val_time = []
            train_time = []
            min_loss = float('Inf')
            grads = []

            for i in range(1,args.epochs+1):
                # if i % 50 == 0:
                #     # lr = max(0.000002,args.learning_rate * (0.1 ** (i // 10)))
                #     for g in engine.optimizer.param_groups:
                #         g['lr'] *= 0.9

                train_loss = []
                train_mae = []
                train_mape = []
                train_rmse = []
                t1 = time.time()
                engine.set_state('train')
                x = _input[:nTrain]
                y = _gt[:nTrain]
                adj_idx = adj_mx_idx[:nTrain]
                if scatter:
                    _coeffs = coeffs[:nTrain]

                # # for overfitting
                # x = _input
                # y = _gt
                # adj_idx = adj_mx_idx
                # if scatter:
                #     _coeffs = coeffs

                iter = 0
                # shuffle in-out-adj_idx
                if scatter:
                    x, y, adj_idx, _coeffs = shuffle(x, y, adj_idx, _coeffs)
                else:
                    x, y, adj_idx = shuffle(x, y, adj_idx)

                # shuffle y, to test if model really learns anything (~random output)
                # y = shuffle(y)

                for batch_i in range(nTrain//args.batch_size):
                    _adj_idx = adj_idx[batch_i * args.batch_size: (batch_i + 1) * args.batch_size]
                    _x = torch.Tensor(x[batch_i * args.batch_size: (batch_i + 1) * args.batch_size][...,None]).to(device).transpose(1, 3)
                    if scatter:
                        # _y = y[batch_i * args.batch_size: (batch_i + 1) * args.batch_size]
                        # coeff_y = torch.Tensor(scattering(_y.transpose(0,2,1))).to(device)
                        # _y = torch.Tensor(_y).to(device)
                        _y = torch.Tensor(y[batch_i * args.batch_size: (batch_i + 1) * args.batch_size]).to(device)
                        coeff_y  = torch.Tensor(_coeffs[batch_i * args.batch_size: (batch_i + 1) * args.batch_size]).to(device)

                        _y = [_y.transpose(1,2), coeff_y]
                    else:
                        _y = torch.Tensor(y[batch_i * args.batch_size: (batch_i + 1) * args.batch_size]).to(device)

                    if F_only:
                        metrics = engine.train_CRASH(_x, _y, None, region_assignment, _adj_idx)
                    else:
                        metrics = engine.train_CRASH(_x, _x, _y, region_assignment, _adj_idx)

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
                x = _input[nTrain:nTrain+nValid]
                y = _gt[nTrain:nTrain+nValid]
                adj_idx = adj_mx_idx[nTrain:nTrain+nValid]
                if scatter:
                    _coeffs = coeffs[nTrain:nTrain+nValid]

                # # for overfitting
                # x = _input
                # y = _gt
                # adj_idx = adj_mx_idx
                # if scatter:
                #     _coeffs = coeffs

                for batch_i in range(nValid//args.batch_size):
                    _adj_idx = adj_idx[batch_i * args.batch_size: (batch_i + 1) * args.batch_size]
                    _x = torch.Tensor(x[batch_i * args.batch_size: (batch_i + 1) * args.batch_size][...,None]).to(device).transpose(1, 3)
                    if scatter:
                        # _y = y[batch_i * args.batch_size: (batch_i + 1) * args.batch_size]
                        # coeff_y = torch.Tensor(scattering(_y.transpose(0,2,1))).to(device)
                        # _y = torch.Tensor(_y).to(device)
                        _y = torch.Tensor(y[batch_i * args.batch_size: (batch_i + 1) * args.batch_size]).to(device)
                        coeff_y  = torch.Tensor(_coeffs[batch_i * args.batch_size: (batch_i + 1) * args.batch_size]).to(device)

                        _y = [_y.transpose(1,2), coeff_y]
                    else:
                        _y = torch.Tensor(y[batch_i * args.batch_size: (batch_i + 1) * args.batch_size]).to(device)

                    if batch_i == 0: # only viz the first one
                        if F_only:
                            metrics = engine.eval_CRASH(_x, _y, None, region_assignment, _adj_idx, viz=False)
                        else:
                            metrics = engine.eval_CRASH(_x, _x, _y, region_assignment, _adj_idx, viz=False)
                    else:
                        if F_only:
                            metrics = engine.eval_CRASH(_x, _y, None, region_assignment, _adj_idx)
                        else:
                            metrics = engine.eval_CRASH(_x, _x, _y, region_assignment, _adj_idx)
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
                his_loss_train.append(mtrain_loss)
                his_loss_val.append(mvalid_loss)

                log = 'Epoch: {:03d}, Train Loss: {:.6f}, Train MAE: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.6f}, Valid MAE: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Valid CC: {:.4f}, Training Time: {:.4f}/epoch'
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
            plt.plot(his_loss_train, label='train loss')
            plt.plot(his_loss_val, label='val loss')
            plt.legend()
            plt.show()
            # ipdb.set_trace() #tmp1 = [i[0] for i in grads] # plot his_loss_train and his_loss_val
            his_loss = his_loss_val

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
        real_Fs = []
        real_Es = []
        pred_Fs = []
        pred_Es = []
        pred_coeffs = []

        engine.set_state('test')
        x = _input[-nTest:]
        y = _gt[-nTest:]
        adj_idx = adj_mx_idx[-nTest:]
        if scatter:
            _coeffs = coeffs[-nTest:]

        # # for overfitting
        # x = _input
        # y = _gt
        # adj_idx = adj_mx_idx
        # if scatter:
        #     _coeffs = coeffs

        for batch_i in range(nTest//args.batch_size):
            _adj_idx = adj_idx[batch_i * args.batch_size: (batch_i + 1) * args.batch_size]
            _x = torch.Tensor(x[batch_i * args.batch_size: (batch_i + 1) * args.batch_size][...,None]).to(device).transpose(1, 3)
            if scatter:
                # _y = y[batch_i * args.batch_size: (batch_i + 1) * args.batch_size]
                # coeff_y = torch.Tensor(scattering(_y.transpose(0,2,1))).to(device)
                # _y = torch.Tensor(_y).to(device)
                _y = torch.Tensor(y[batch_i * args.batch_size: (batch_i + 1) * args.batch_size]).to(device)
                coeff_y  = torch.Tensor(_coeffs[batch_i * args.batch_size: (batch_i + 1) * args.batch_size]).to(device)
                
                _y = [_y.transpose(1,2), coeff_y]
            else:
                _y = torch.Tensor(y[batch_i * args.batch_size: (batch_i + 1) * args.batch_size]).to(device)

            if batch_i == 0: # only viz the first one
                if F_only:
                    metrics = engine.eval_CRASH(_x, _y, None, region_assignment, _adj_idx, viz=True)
                else:
                    metrics = engine.eval_CRASH(_x, _x, _y, region_assignment, _adj_idx, viz=True)
            else:
                if F_only:
                    metrics = engine.eval_CRASH(_x, _y, None, region_assignment, _adj_idx)
                else:
                    metrics = engine.eval_CRASH(_x, _x, _y, region_assignment, _adj_idx)

            amae.append(metrics[1])
            amape.append(metrics[2])
            armse.append(metrics[3])

            if F_only:
                real_Fs.append(_y)
            else:
                real_Es.append(_y)
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
                        plt.figure(0) # timestep 0
                        plt.plot(real_Fs[0].squeeze().cpu().numpy()[3,0], label='real Fs')
                        plt.plot(pred_Fs[0].squeeze().cpu().numpy()[3,0], label='pred Fs')
                        plt.legend()
                        plt.show()
                        plt.figure(1) # timestep 3
                        plt.plot(real_Fs[0].squeeze().cpu().numpy()[3,3], label='real Fs')
                        plt.plot(pred_Fs[0].squeeze().cpu().numpy()[3,3], label='pred Fs')
                        plt.legend()
                        plt.show()
                    else:
                        ipdb.set_trace()
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

        for viz_node_idx in range(61):
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

    main(scatter=False, _map=False, F_only=True) # F prediction
    # main(scatter=False, _map=True, F_only=False, subsample=6)
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2-t1))