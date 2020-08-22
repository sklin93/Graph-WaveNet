import pickle
import numpy as np
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
from scipy.signal import butter,filtfilt, zpk2sos, sosfilt
from Utils import graphTools
from Utils import dataTools
from Utils.CRASH_loader import *

import ipdb
from tqdm import tqdm

class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample 
         to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()

class DataLoader_syn(object):
    def __init__(self, xs, ys, adj_idx, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample 
         to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            adj_idx_padding = np.repeat(adj_idx[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
            adj_idx = np.concatenate([adj_idx, adj_idx_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys
        self.adj_idx = adj_idx

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys, adj_idx = self.xs[permutation], self.ys[permutation], \
                          self.adj_idx[permutation]
        self.xs = xs
        self.ys = ys
        self.adj_idx = adj_idx

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                adj_i = self.adj_idx[start_ind: end_ind, ...]
                yield (x_i, y_i, adj_i)
                self.current_ind += 1

        return _wrapper()

class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean



def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()

def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def mod_adj(adj_mx, adjtype):
    if adjtype == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx)]
    elif adjtype == "normlap":
        adj = [calculate_normalized_laplacian(adj_mx).astype(np.float32).todense()]
    elif adjtype == "symnadj":
        adj = [sym_adj(adj_mx)]
    elif adjtype == "transition":
        adj = [asym_adj(adj_mx)]
    elif adjtype == "doubletransition":
        adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
    elif adjtype == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    else:
        error = 0
        assert error, "adj type not defined"
    return adj

def load_adj(pkl_filename, adjtype):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    adj = mod_adj(adj_mx, adjtype)
    return sensor_ids, sensor_id_to_ind, adj


def load_dataset_metr(dataset_dir, batch_size, valid_batch_size= None, test_batch_size=None):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
    # x/y_train/val/test: (N, 12, 207, 2)
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scaler
    return data

def load_dataset_syn(adjtype, nNodes, nTrain, nValid, nTest, num_timestep, K, 
                     batch_size, valid_batch_size= None, test_batch_size=None, 
                     same_G=True, pooltype='avg'): 
    '''
    K: K-step prediction (also K step input)
    same_G: whether all samples have a same graph structure or not
    pooltype: can be 'avg','selectOne','weighted'
    '''
    # graph config
    graphType = 'SBM' # Type of graph
    graphOptions = {}
    graphOptions['nCommunities'] = 5 #64 # Number of communities (EEG node number)
    graphOptions['probIntra'] = 0.8 # Intracommunity probability
    graphOptions['probInter'] = 0.2 # Intercommunity probability
    # sample config
    F_t = 24 # need K%F_t==0 for a cleaner fMRI cut
    # noise parameters
    sigmaSpatial = 0.1
    sigmaTemporal = 0.1
    rhoSpatial = 0
    rhoTemporal = 0

    if same_G:
        # data generation
        G = graphTools.Graph(graphType, nNodes, graphOptions)
        G.computeGFT() # Compute the eigendecomposition of the stored GSO
        _data = dataTools.MultiModalityPrediction(G, K, nTrain, nValid, nTest, num_timestep, 
                                                  F_t=F_t, pooltype=pooltype, 
                                                  sigmaSpatial=sigmaSpatial, 
                                                  sigmaTemporal=sigmaTemporal,
                                                  rhoSpatial=rhoSpatial, 
                                                  rhoTemporal=rhoTemporal)
        data = {}
        for category in ['train', 'val', 'test']:
            data['x_' + category], _, _, data['y_' + category] = _data.getSamples(category)

        scaler_F = StandardScaler(mean=data['x_train'][..., 0].mean(), 
                                std=data['x_train'][..., 0].std())
        scaler_E = StandardScaler(mean=data['x_train'][..., 1].mean(), 
                                std=data['x_train'][..., 1].std())
        # Data format
        for category in ['train', 'val', 'test']:
            data['x_' + category][..., 0] = scaler_F.transform(data['x_' + category][..., 0])
        
        data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
        data['val_loader'] = DataLoader(data['x_val'], data['y_val'], valid_batch_size)
        data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
        data['scaler'] = [scaler_F, scaler_E]
        adj = mod_adj(G.W, adjtype)
        return data, adj, F_t, G
    else:
        nTotal = nTrain + nValid + nTest
        # Gs = []
        # adjs = []
        # F_xs = []
        # # E_xs = []
        # # F_ys = []
        # E_ys = []
        # for i in tqdm(range(nTotal)):
        #     G = graphTools.Graph(graphType, nNodes, graphOptions)
        #     G.computeGFT()
        #     _data = dataTools.MultiModalityPrediction(G, K, 1, 0, 0, num_timestep, 
        #                                               F_t=F_t, pooltype=pooltype, 
        #                                               sigmaSpatial=sigmaSpatial, 
        #                                               sigmaTemporal=sigmaTemporal,
        #                                               rhoSpatial=rhoSpatial, 
        #                                               rhoTemporal=rhoTemporal)
        #     _data = _data.getSamples('train') # [F_x, E_x, F_y, E_y]
            
        #     F_xs.append(_data[0])
        #     # E_xs.append(_data[1])
        #     # F_ys.append(_data[2])
        #     E_ys.append(_data[3])
        #     Gs.append(G)
        #     adjs.append(mod_adj(G.W, adjtype))

        # F_xs = np.concatenate(F_xs) #(2, 95, 5, 80)
        # # E_xs = np.concatenate(E_xs)
        # # F_ys = np.concatenate(F_ys)
        # E_ys = np.concatenate(E_ys) #(2, 95, 120, 5)

        ##### same G
        G = graphTools.Graph(graphType, nNodes, graphOptions)
        G.computeGFT() # Compute the eigendecomposition of the stored GSO
        _data = dataTools.MultiModalityPrediction(G, K, nTrain, nValid, nTest, num_timestep, 
                                                  F_t=F_t, pooltype=pooltype, 
                                                  sigmaSpatial=sigmaSpatial, 
                                                  sigmaTemporal=sigmaTemporal,
                                                  rhoSpatial=rhoSpatial, 
                                                  rhoTemporal=rhoTemporal)
        Gs = [G] * nTotal
        adjs = [mod_adj(G.W, adjtype)] * nTotal
        F_xs = np.concatenate([_data.getSamples('train')[0],_data.getSamples('val')[0],
                                                            _data.getSamples('test')[0]])
        E_ys = np.concatenate([_data.getSamples('train')[3],_data.getSamples('val')[3],
                                                            _data.getSamples('test')[3]])
        #####

        G = {}
        data = {}
        # data['x_train'], data['y_train'], G['train'] = xs[:nTrain], ys[:nTrain], Gs[:nTrain]
        # data['x_val'], data['y_val'], G['val'] = xs[nTrain:-nTest], ys[nTrain:-nTest], Gs[nTrain:-nTest]
        # data['x_test'], data['y_test'], G['test'] = xs[-nTest:], ys[-nTest:], Gs[-nTest:]
        
        # F in & E out
        data['x_train'], data['y_train'], G['train'] = F_xs[:nTrain][...,None], E_ys[:nTrain][...,None], Gs[:nTrain]
        data['x_val'], data['y_val'], G['val'] = F_xs[nTrain:-nTest][...,None], E_ys[nTrain:-nTest][...,None], Gs[nTrain:-nTest]
        data['x_test'], data['y_test'], G['test'] = F_xs[-nTest:][...,None], E_ys[-nTest:][...,None], Gs[-nTest:]
        
        data['train_adj_idx'] = np.arange(nTrain).reshape(-1,1).repeat(
                                                data['x_train'].shape[1],axis=1)
        data['val_adj_idx'] = np.arange(nValid).reshape(-1,1).repeat(
                                                data['x_val'].shape[1],axis=1)
        data['test_adj_idx'] = np.arange(nTest).reshape(-1,1).repeat(
                                                data['x_test'].shape[1],axis=1)
        
        for k, v in data.items():
            # batching 1 : train model on one subject then finetune
            data[k] = v.reshape(-1, *v.shape[2:])
            # # batching 2 : each batch contains different subject
            # v = np.transpose(v, (1,0,2,3,4)).reshape(-1, *v.shape[2:])

        # scaler_F = StandardScaler(mean=data['x_train'][..., 0].mean(), 
        #                         std=data['x_train'][..., 0].std())
        # scaler_E = StandardScaler(mean=data['x_train'][..., 1].mean(), 
        #                         std=data['x_train'][..., 1].std())

        scaler_F = StandardScaler(mean=data['x_train'].mean(), 
                                std=data['x_train'].std())
        # TODO: is it ok to use y's info?
        scaler_E = StandardScaler(mean=data['y_train'].mean(), 
                                std=data['y_train'].std())
        # Data format
        for category in ['train', 'val', 'test']:
            data['x_' + category][..., 0] = scaler_F.transform(data['x_' + category][..., 0])
            # TODO: temporarily scale y 
            data['y_' + category][..., 0] = scaler_F.transform(data['y_' + category][..., 0])
        
        data['train_loader'] = DataLoader_syn(data['x_train'], data['y_train'], 
                                              data['train_adj_idx'], batch_size)
        data['val_loader'] = DataLoader_syn(data['x_val'], data['y_val'], 
                                            data['val_adj_idx'], valid_batch_size)
        data['test_loader'] = DataLoader_syn(data['x_test'], data['y_test'],
                                             data['test_adj_idx'], test_batch_size)
        data['scaler'] = [scaler_F, scaler_E]
        
        return data, adjs, F_t, G

def load_dataset_CRASH(adjtype, pad_seq=False): 
    '''
    If pad_seq is True, the shorter EEG and fMRI sequences will be padded with their 
    values on the last timestep;
    If it is set to False, the shorter (irregular) sequences will be discarded from 
    dataloading process
    '''

    comn_ids = get_comn_ids()
    print(len(comn_ids), 'subjects:', comn_ids)
    num_region = 200 # 200 or 400

    eeg = get_eeg(comn_ids)
    sc = get_sc(comn_ids, num_region)
    fmri = get_fmri(comn_ids, num_region)
    fmri_time_res = fmri['time_res']
    eeg_time_res = eeg['time_res']
    F_t = fmri_time_res / eeg_time_res #582.4

    ''' 
    # choose a common length for EEG based on length of fMRI, and their time resolutions
    clip & pad EEG sequences with different length (to the most common length)
    '''
    # subj = fmri[(list(fmri)[-1])]
    # fmri_len = len(subj[list(subj)[0]])
    # eeg_len = 1 + int((fmri_len - 1) * fmri_time_res / eeg_time_res)
    fmri_len = 326   # hard coded to the most common length
    eeg_len = 189282# hard coded to the most common length

    # fmri_len = 51  # due to memory limit
    # eeg_len = 1 + int((fmri_len - 1) * F_t) # due to memory limit

    # check and keep only common sessions for each subject
    eeg_mat = []
    adjs = []
    fmri_mat = []

    # sub_ses = {}
    for subj in comn_ids:
        # comn_sess = []
        for k in eeg[subj]:
            if k in sc[subj] and k in fmri[subj]:
                cur_fmri = fmri[subj][k][:fmri_len]
                cur_eeg = eeg[subj][k].transpose(1,0)[:eeg_len]
                # OPTION 1: pad shorter sequences
                if pad_seq:
                    # comn_sess.append(k)
                    adjs.append(mod_adj(sc[subj][k], adjtype))

                    if len(cur_fmri) < fmri_len:
                        cur_fmri = np.concatenate((cur_fmri, cur_fmri[-1:].repeat(
                                                    fmri_len - len(cur_fmri),axis=0)))
                    if len(cur_eeg) < eeg_len:
                        cur_eeg = np.concatenate((cur_eeg, cur_eeg[-1:].repeat(
                                                    eeg_len - len(cur_eeg),axis=0)))
                    fmri_mat.append(cur_fmri)
                    eeg_mat.append(cur_eeg)

                # OPTION 2: discard shorter sequences (only keep the legit ones..)
                else:
                    if len(cur_fmri) == fmri_len and len(cur_eeg) == eeg_len:
                        # comn_sess.append(k)
                        adjs.append(mod_adj(sc[subj][k], adjtype))
                        fmri_mat.append(cur_fmri)
                        eeg_mat.append(cur_eeg)
        # sub_ses[subj] = comn_sess
    del fmri
    del eeg
    fmri_mat = np.stack(fmri_mat)
    eeg_mat = np.stack(eeg_mat)

    region_assignment = get_region_assignment(num_region) #{EEG_electrodes: brain region}

    # TODO: normalize/Standardize

    return adjs, fmri_mat, eeg_mat, region_assignment, F_t

    '''
    Several issues: 
    - F_t = fmri_res / eeg_res is not an integer [x]
    - choose K (which needs to be very large --> since F_t~580) [x]
    - region assignment have empty nodes, need to think of how to deal with them [x]
    - cannot feed the whole sequence length into the memory, has to break into chuncks
    '''
    '''
    K = int(F_t * 5)
    
    # MEMORY ISSUE #
    # repeat fmri F_t times for each timestep
    print('fmri temporal extension')
    signals = [] #fmri
    rpt_ts = 0
    for i in tqdm(range(fmri_len-1)):
        rpt_t = round((i+1)*F_t) - round(i*F_t)
        rpt_ts += rpt_t
        signals.append(fmri_mat[:, i:i+1, :].repeat(rpt_t, axis=1))
    signals.append(fmri_mat[:, -1:, :])
    signals = np.concatenate(signals, axis=1)
    del fmri_mat
    
    # expand eeg signals from num_electrods to num_region
    print('eeg spatial extension')
    eeg = np.zeros_like(signals)
    for i in tqdm(range(num_region)):
        eeg[:, :, i] = eeg_mat[:, :, inv_mapping[i]].mean(-1)
    del eeg_mat
    
    signals = np.stack((signals, eeg), axis=-1)
    del eeg

    # TODO: sliding window
    idxer = np.arange(K)[None, :] + np.arange(eeg_len - K + 1)[:, None]
    ipdb.set_trace()
    ###### MEMORY ISSUE ######
    signals[:, idxer, :, :]

    # TODO: put region_assignment mapping under G
    G = {}
    data = {}
    data['x_train'], data['y_train'], G['train'] = xs[:nTrain], ys[:nTrain], Gs[:nTrain]
    data['x_val'], data['y_val'], G['val'] = xs[nTrain:-nTest], ys[nTrain:-nTest], Gs[nTrain:-nTest]
    data['x_test'], data['y_test'], G['test'] = xs[-nTest:], ys[-nTest:], Gs[-nTest:]

    data['train_adj_idx'] = np.arange(nTrain).reshape(-1,1).repeat(
                                            data['x_train'].shape[1],axis=1)
    data['val_adj_idx'] = np.arange(nValid).reshape(-1,1).repeat(
                                            data['x_val'].shape[1],axis=1)
    data['test_adj_idx'] = np.arange(nTest).reshape(-1,1).repeat(
                                            data['x_test'].shape[1],axis=1)
    
    for k, v in data.items():
        # batching 1 : train model on one subject then finetune
        data[k] = v.reshape(-1, *v.shape[2:])
        # # batching 2 : each batch contains different subject
        # v = np.transpose(v, (1,0,2,3,4)).reshape(-1, *v.shape[2:])

    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), 
                            std=data['x_train'][..., 0].std())
    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
    
    data['train_loader'] = DataLoader_syn(data['x_train'], data['y_train'], 
                                          data['train_adj_idx'], batch_size)
    data['val_loader'] = DataLoader_syn(data['x_val'], data['y_val'], 
                                        data['val_adj_idx'], valid_batch_size)
    data['test_loader'] = DataLoader_syn(data['x_test'], data['y_test'],
                                         data['test_adj_idx'], test_batch_size)
    data['scaler'] = scaler
    
    return data, adjs, F_t, G
    '''

def inverse_sliding_window(li, K=None):
    '''
    takes in a list, each with dimension [num_window, num_nodes, window_width]
    with stride (each window's starting index discrepancy) K
    return a list, each with dimension [num_nodes, num_timesteps]
    ***** The overlapped portion are averaged *****
    '''
    def _rev(a, _K):
        assert len(a.shape) == 3
        num_window, num_nodes, width = a.shape
        num_t = width + (num_window - 1) * _K

        a = a.transpose(0, 2, 1)
        idxer = np.arange(width)[None, :] + np.arange(0, num_t-width+1, _K)[:, None]
        rev = np.zeros((num_nodes, num_t))
        for l in range(num_t):
            rev[:, l] = a[idxer == l].mean(0)
        return rev

    ret = []
    if K is None:
        K = [1]*len(li)
    else:
        assert len(li) == len(K)
    for i in range(len(li)):
        ret.append(_rev(li[i], K[i]))
    return ret

def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred,real,0.0).item()
    mape = masked_mape(pred,real,0.0).item()
    rmse = masked_rmse(pred,real,0.0).item()
    return mae,mape,rmse

def butter_lowpass_filter(data, cutoff, fs, order=6):
    nyq = 0.5*fs
    normal_cutoff = cutoff / nyq

    # z,p,k = butter(order, normal_cutoff, output="zpk", btype='low', analog=False)
    # lesos = zpk2sos(z, p, k)
    # y = sosfilt(lesos, data)
    # return y

    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y
