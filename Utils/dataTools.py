# 2018/12/4~
"""
dataTools.py Data management module

Several tools to manage data

SourceLocalization (class): creates the datasets for a source localization 
    problem
Authorship (class): loads and splits the dataset for the authorship attribution
    problem
TwentyNews (class): handles the 20NEWS dataset
"""

import numpy as np
import torch
import pickle as pkl
#import hdf5storage # This is required to import old Matlab(R) files.

import Utils.graphTools as graph
import Utils.miscTools as misc

import ipdb

class MultiModalityPrediction():
    """
    Creates the dataset for a prediction problem supervised by 2 coarsed version of original signal

    Initialization:

    Input:
        G (class): Graph on which to diffuse the process, needs an attribute
            .N with the number of nodes (int) and attribute .W with the
            adjacency matrix (np.array)
        nTrain (int): number of training samples
        nValid (int): number of validation samples
        nTest (int): number of testing samples
        horizon (int): length of the process
        F_t (int): time resolution of generated F signal
        pooltype (string): 'avg' or 'selectOne' or 'weighted'; the method for extracting F and E
        FPoolDecay (float): decay factor between 0 and 1 for weight pool type generating F
        EPoolDecay (float): decay factor between 0 and 1 for weight pool type generating E
        sigmaSpatial (float): spatial variance
        sigmaTemporal (float): temporal variance
        rhoSpatial (float): spatial correlation
        rhoTemporal (float): temporal correlation
        dataType (dtype): datatype for the samples created (default: np.float64)
        device (device): if torch.Tensor datatype is selected, this is on what
            device the data is saved.

    Methods:

    signals, labels = .getSamples(samplesType[, optionalArguments])
        Input:
            samplesType (string): 'train', 'val' or 'test' to determine from
                which dataset to get the samples from
            optionalArguments:
                0 optional arguments: get all the samples from the specified set
                1 optional argument (int): number of samples to get (at random)
                1 optional argument (list): specific indices of samples to get
        Output:
            signals (dtype.array): numberSamples x numberNodes
            labels (dtype.array): numberSamples
            >> Obs.: The 0th dimension matches the corresponding signal to its
                respective label

    .astype(type): change the type of the data matrix arrays.
        Input:
            type (dtype): target type of the variables (e.g. torch.float64,
                numpy.float64, etc.)

    .to(device): if dtype is torch.tensor, move them to the specified device.
        Input:
            device (string): target device to move the variables to (e.g. 'cpu',
                'cuda:0', etc.)

    RMSE = .evaluate(yHat, y, tol = 1e-9)
        Input:
            yHat (dtype.array): estimated labels (1-D binary vector)
            y (dtype.array): correct labels (1-D binary vector)
            >> Obs.: both arrays are of the same length
            tol (float): numerical tolerance to consider two numbers to be equal
        Output:
            RMSE (float): root mean square error between y and yHat

    """

    def __init__(self, G, K, nTrain, nValid, nTest, horizon, F_t = 5, 
                pooltype='weighted', FPoolDecay=0.8, EPoolDecay=0.8,
                sigmaSpatial = 1, sigmaTemporal = 0, rhoSpatial = 0, 
                rhoTemporal = 0, dataType = np.float64, device = 'cpu'):
        # store attributes
        assert K%F_t == 0 # for cleaner F prediction
        self.K = K
        self.dataType = dataType
        self.device = device
        self.nTrain = nTrain
        self.nValid = nValid
        self.nTest = nTest
        self.horizon = horizon
        self.sigmaSpatial = sigmaSpatial
        self.sigmaTemporal = sigmaTemporal
        self.rhoSpatial = rhoSpatial
        self.rhoTemporal = rhoTemporal
        #\\\ Generate the samples
        # Get the largest eigenvalue of the weighted adjacency matrix
        EW, VW = graph.computeGFT(G.W, order = 'totalVariation')
        eMax = np.max(EW)
        # Normalize the matrix so that it doesn't explode
        Wnorm = G.W / eMax
        # total number of samples
        nTotal = nTrain + nValid + nTest
        # x_0
        x_t = np.random.rand(nTotal,G.N);
        x = [x_t]
        # Temporal noise
        tempNoise = np.random.multivariate_normal(np.zeros(self.horizon),
                                                  np.power(self.sigmaTemporal,2)*np.eye(self.horizon) + 
                                                  np.power(self.rhoTemporal,2)*np.ones((self.horizon,self.horizon)),
                                                  (nTotal, G.N))
        tempNoise = np.transpose(tempNoise, (2,0,1)) #(horizon, nTotal, G.N)
        # Create LS
        A = Wnorm # = A x_t + w (Gaussian noise)
        for t in range(self.horizon-1):
            # spatialNoise (for each t): (nTotal, G.N)
            spatialNoise = np.random.multivariate_normal(np.zeros(G.N), 
                                 np.power(self.sigmaSpatial,2)*np.eye(G.N) + 
                                 np.power(self.rhoSpatial,2)*np.ones((G.N,G.N)), nTotal)
            x_tplus1 = np.matmul(x_t,A) + spatialNoise + tempNoise[t, :, :]
            x_t = x_tplus1
            x.append(x_t)

        x = np.stack(x, axis=-1) # (nTotal, G.N, horizon)
        
        # synthetic F (coarse temporal) and E (coarse spacial)
        F = self._gen_F(x, F_t, pooltype, FPoolDecay) #(nTotal, horizen//F_t, G.N)
        E = self._gen_E(x, G, pooltype, EPoolDecay) #(nTotal, horizen, G.nCommunities)
        FE = np.stack((F,E), axis=-1) # combined signal, along feature dimension
        # # signals and labels for F
        # F_idxer = np.arange(K//F_t)[None, :] + np.arange((horizon-K)//F_t+1)[:, None]
        # F_signals = F[:, F_idxer[:-K//F_t], :]
        # F_labels = F[:, F_idxer[K//F_t:], :]
        # # signals and labels for E
        # E_idxer = np.arange(K)[None, :] + np.arange(horizon-K+1)[:, None]
        # E_signals = E[:, E_idxer[:-K], :]
        # E_labels = E[:, E_idxer[K:], :]

        # sliding window indexer
        idxer = np.arange(K)[None, :] + np.arange(horizon-K+1)[:, None]
        signals = FE[:, idxer[:-K], :, :]
        labels = FE[:, idxer[K:], :, :]
        # Split and save them
        self.samples = {}
        self.samples['train'] = {}
        self.samples['train']['x'] = signals[0:nTrain, :]
        self.samples['train']['y'] = labels[0:nTrain, :]
        self.samples['val'] = {} 
        self.samples['val']['x'] = signals[nTrain:nTrain+nValid, :]
        self.samples['val']['y'] = labels[nTrain:nTrain+nValid, :]
        self.samples['test'] = {}
        self.samples['test']['x'] = signals[nTrain+nValid:nTotal, :]
        self.samples['test']['y'] = labels[nTrain+nValid:nTotal, :]

        # Change data to specified type and device
        self.astype(self.dataType)
        self.to(self.device)

    def _gen_F(self, x, F_t, pooltype, alpha=0.8):
        # synthetic F (1 every F_t time step)
        if pooltype == 'selectOne':
            F = x[:, :, np.arange(0, x.shape[-1], F_t)] #(L, N, T//F_t)

        elif pooltype == 'avg':
            F = x.reshape(x.shape[0], x.shape[1], -1, F_t).mean(-1) #(L, N, T//F_t)
            
        elif pooltype == 'weighted':
            alpha = 1
            weight = []
            for i in range(F_t):
                weight.append(alpha ** abs(i - (F_t//2)))
            weight = np.asarray(weight) / sum(weight)
            F = x.reshape(x.shape[0], x.shape[1], -1, F_t)
            F = (F * weight[None, None, None, :]).sum(-1) #(L, N, T//F_t)

        F = F.transpose(0, 2, 1) #(L, T//F_t, N)
        F = F.repeat(F_t, axis=1)
        '''
        # TODO: return F.reshape(-1, *F.shape[2:]) treats every sample as a same one
        # need differentiation when using different graphs
        '''
        return F
        
    def _gen_E(self, x, G, pooltype, beta=0.8):
        # synthetic E (1 signal for each clusters)
        assign_dict = G.assign_dict
        if pooltype == 'selectOne':
            E = []
            for k, v in assign_dict.items():
                E.append(x[:, v[len(v)//2], :])

        if pooltype == 'avg':
            E = []
            for k, v in assign_dict.items():
                E.append(np.average(x[:, v, :], axis=1))
            
        elif pooltype == 'weighted':
            E = []
            for k, v in assign_dict.items():
                # chose one node that contribute the most
                chosen = len(v)//2
                cluster_W = G.W[v][:,v]
                #assumption here: symmetric graph (TODO: Asymmetric ones)
                weight = np.zeros(len(v))
                remained = np.ones(len(v), dtype=int)

                weight[chosen] = 1
                remained[chosen] = 0
                nei_idx = cluster_W[chosen].astype(bool)
                k = 1
                # contribution decays by hop
                while sum(remained) != 0:
                    # k-hop neighbor of the chosen node
                    weight[nei_idx] = beta**k
                    remained = remained - nei_idx
                    nei_idx = (cluster_W[nei_idx].sum(0).astype(bool) \
                                * remained).astype(bool)
                    k += 1
                E.append((x[:, v, :] * weight[None, :, None]).sum(1))
             

        _E = np.stack(E, axis=-1) # (L, T, K)
        E = np.zeros((_E.shape[0], _E.shape[1], G.N))
        for k in range(len(assign_dict)):
            E[:, :, assign_dict[k]] = _E[:, :, k:k+1].repeat(len(assign_dict[k]), axis=-1)
        '''
        # TODO: return E.reshape(-1, *E.shape[2:]) treats every sample as a same one
        # need differentiation when using different graphs
        '''
        return E

    def getSamples(self, samplesType):
        # type: train, valid, test
        # args: 0 args, give back all
        # args: 1 arg: if int, give that number of samples, chosen at random
        # args: 1 arg: if list, give those samples precisely.
        # Check that the type is one of the possible ones
        # return: F signal, F label, E signal, E label
        assert samplesType == 'train' or samplesType == 'val' \
                    or samplesType == 'test'
        
        x = self.samples[samplesType]['x']
        y = self.samples[samplesType]['y']
        '''
        # TODO: return x.reshape(-1, *x.shape[2:]) treats every sample as a same one
        # need differentiation when using different graphs
        '''
        x = x.reshape(-1, *x.shape[2:])
        y = y.reshape(-1, *y.shape[2:])
        return x, y

    def astype(self, dataType):
        if repr(dataType).find('torch') == -1:
            for key in self.samples.keys():
                for secondKey in self.samples[key].keys():
                    self.samples[key][secondKey] \
                                            = dataType(self.samples[key][secondKey])
        else:
            for key in self.samples.keys():
                for secondKey in self.samples[key].keys():
                    self.samples[key][secondKey] \
                         = torch.tensor(self.samples[key][secondKey]).type(dataType)

        if dataType is not self.dataType:
            self.dataType = dataType

    def to(self, device):
        # This can only be done if they are torch tensors
        if repr(self.dataType).find('torch') >= 0:
            for key in self.samples.keys():
                for secondKey in self.samples[key].keys():
                    self.samples[key][secondKey] \
                                       = self.samples[key][secondKey].to(device)

            # If the device changed, save it.
            if device is not self.device:
                self.device = device

    def evaluate(self, yHat, y, tol = 1e-9):
        """
        Return the MSE loss
        """
        lossValue = misc.batchTimeMSELoss(yHat,y.type(torch.float64))
        return lossValue
        