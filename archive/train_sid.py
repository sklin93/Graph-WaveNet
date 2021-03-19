import torch
import statistics
import numpy as np
import argparse
import time
import Utils.util as util
from engine import trainer
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer
import ipdb
from sklearn.utils import shuffle
# import GPUtil
import glob

# python train.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --num_nodes 207 --seq_length 12 --save ./garage/metr
# python train.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --num_nodes 80 --data syn --blocks 2 --layers 2 --in_dim=1
# python train.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --data CRASH --num_nodes 200 --seq_length 2912 --in_dim 1 --blocks 2 --layers 2 --batch_size 16 --learning_rate 0.0005 --save ./garage/CRASH
# nohup python -u train.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --data CRASH --num_nodes 200 --seq_length 2912 --in_dim 1 --blocks 2 --layers 2 --batch_size 16 --learning_rate 0.0005 --save ./garage/CRASH > log_CRASH 2>&1 &
# (Notice the CRASH can handle batch size 32 on server)
parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--data',type=str,default='data/METR-LA',help='data path')
parser.add_argument('--adjdata',type=str,default='data/sensor_graph/adj_mx.pkl',help='adj data path')
parser.add_argument('--adjtype',type=str,default='doubletransition',help='adj type')
parser.add_argument('--gcn_bool',action='store_true',help='whether to add graph convolution layer')
parser.add_argument('--aptonly',action='store_true',help='whether only adaptive adj')
parser.add_argument('--addaptadj',action='store_true',help='whether add adaptive adj')
parser.add_argument('--randomadj',action='store_true',help='whether random initialize adaptive adj')
parser.add_argument('--seq_length',type=int,default=60,help='')
parser.add_argument('--nhid',type=int,default=32,help='')
parser.add_argument('--in_dim',type=int,default=2,help='inputs dimension')
parser.add_argument('--num_nodes',type=int,default=80,help='number of nodes')
parser.add_argument('--kernel_size',type=int,default=3,help='kernel_size of dilated convolution')
parser.add_argument('--layers',type=int,default=2,help='number of layers per gwnet block')
parser.add_argument('--blocks',type=int,default=4,help='number of blocks in gwnet model')

parser.add_argument('--batch_size',type=int,default=32,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--epochs',type=int,default=15,help='')
parser.add_argument('--print_every',type=int,default=50,help='')
# parser.add_argument('--seed',type=int,default=0,help='random seed')
parser.add_argument('--save',type=str,default='./garage/syn',help='save path')
parser.add_argument('--expid',type=int,default=1,help='experiment id')

args = parser.parse_args()

np.random.seed(0)
torch.manual_seed(999)
if torch.cuda.is_available():
    #GPUtil.showUtilization()
    torch.cuda.manual_seed_all(999)
    # with torch.cuda.device('cuda:003'):
    #     torch.cuda.empty_cache()

def main(model_name=None, syn_file='syn_diffG.pkl'): # directly loading trained model/ generated syn data
    #set seed
    #torch.manual_seed(args.seed)
    #np.random.seed(args.seed)
    #load data
    same_G = False
    device = torch.device(args.device)

    if args.data == 'CRASH':
        adj_mx, fmri_mat, eeg_mat, region_assignment, F_t = util.load_dataset_CRASH(args.adjtype)
        adj_mx, fmri_mat, eeg_mat = shuffle(adj_mx, fmri_mat, eeg_mat,random_state = 0)
        
        #for i in range(10):
        #    plt.hist(eeg_mat[i,:,:])
        #    plt.show()
        #for i in range(5):
        #    plt.figure()
        #    plt.plot(fmri_mat[0,:,5*i], label = "eeg")
        #    plt.savefig('./preprocessPics/' + 'origFMRI'+str(i)+'.png')
        #    plt.close()
        #    plt.plot(fmri_mat[i,:,0], label = "fmri")
        #    plt.show()
        #for i in range(1):
        #    plt.plot(fmri_mat[113,40:49,i], label = "fmri")
        #    plt.show()

        #GPUtil.showUtilization() 
        # Standardize data
        #print(adj_mx.shape)
        #print(fmri_mat.shape)
        #print(eeg_mat.shape)
        nTrain = round(0.75 * len(adj_mx))
        nValid = round(0.15 * len(adj_mx))
        nTest = len(adj_mx) - nTrain - nValid
        num_subj, t_f, n_f = fmri_mat.shape
        _, t_e, n_e = eeg_mat.shape
        fmri_mat = fmri_mat.reshape(num_subj, n_f, t_f)
        #fmri_mat = fmri_mat.reshape(num_subj*n_f, -1)
        #axis is 0 since u want to handle each subject differently(large variance)
        _meanSub = fmri_mat[:nTrain].mean()
        #_meanNode = fmri_mat[:nTrain].mean(1, keepdims = True)
        #print(_mean.shape)
        _stdSub = fmri_mat[:nTrain].std((0,1), keepdims = True)
        _stdNode = fmri_mat[:nTrain].std(1, keepdims = True)
        #fmri_mat -= _meanSub
        #fmri_mat -= _meanNode
        #fmri_mat /= _stdSub
        #fmri_mat /= _stdNode
        fmri_mat = fmri_mat.reshape(num_subj, t_f, n_f)
        #
        eeg_mat = eeg_mat.reshape(num_subj,n_e, t_e)
        _meanSub = eeg_mat[:nTrain].mean()
        _std = eeg_mat[:nTrain].std((0,1), keepdims = True)
        #eeg_mat -= _meanSub
        #eeg_mat /= _std
        eeg_mat = eeg_mat.reshape(num_subj, t_e, n_e)
        
        #for i in range(5):
        #    plt.figure()
        #    plt.plot(fmri_mat[0,:,5*i], label = "eeg")
        #    plt.savefig('./preprocessPics/' + 'meanFMRI'+str(i)+'.png')
        #    plt.close()
        #for i in range(10):
        #    plt.plot(fmri_mat[0,:,i], label = "fmri")
        #    plt.show()
        #for i in range(10):
        #    plt.plot(eeg_mat[0,:30000,i], label = "eeg")
        #    plt.show()
        
        #print("fmri min : " + str(fmri_mat[:nTrain].min()))
        #print("fmri max : " + str(fmri_mat[:nTrain].max()))
        # Min-max normalization
        #fmri_mat = (fmri_mat - fmri_mat[:nTrain].min(0,keepdims = True)) / (fmri_mat[:nTrain].max(0, keepdims = True) - fmri_mat[:nTrain].min(0, keepdims = True))
        #eeg_mat = (eeg_mat - eeg_mat[:nTrain].min(0, keepdims = True)) / (eeg_mat[:nTrain].max(0, keepdims = True) - eeg_mat[:nTrain].min(0, keepdims = True))
        
        
        num_subj, t_f, n_f = fmri_mat.shape
        fmri_mat = fmri_mat.reshape(num_subj*n_f, -1)
        scaler_F = MinMaxScaler()
        #print(fmri_mat[:nTrain*n_f].shape)
        ipdb.set_trace()
        scaler_F.fit(fmri_mat[:nTrain*n_f])
        fmri_mat[:nTrain*n_f] = scaler_F.transform(fmri_mat[:n_f*nTrain])
        fmri_mat[n_f*nTrain:-nTest*n_f] = scaler_F.transform(fmri_mat[n_f*nTrain:-nTest*n_f])
        fmri_mat[-nTest*n_f:] = scaler_F.transform(fmri_mat[-nTest*n_f:])
        fmri_mat = fmri_mat.reshape(num_subj, t_f, n_f)
        ipdb.set_trace()
#
        _, t_e, n_e = eeg_mat.shape
        eeg_mat = eeg_mat.reshape(num_subj*n_e, -1)
        scaler_E = MinMaxScaler()
        scaler_E.fit(eeg_mat[:nTrain*n_e])
        eeg_mat[:nTrain*n_e] = scaler_E.transform(eeg_mat[:nTrain*n_e])
        eeg_mat[n_e*nTrain:-nTest*n_e] = scaler_E.transform(eeg_mat[n_e*nTrain:-nTest*n_e])
        eeg_mat[-nTest*n_e:] = scaler_E.transform(eeg_mat[-nTest*n_e:])
        eeg_mat = eeg_mat.reshape(num_subj, t_e, n_e)

        #for i in range(5):
        #    plt.figure()
        #    plt.plot(fmri_mat[0,:,5*i], label = "eeg")
        #    plt.savefig('./preprocessPics/' +'minMaxFMRI' + str(i) + '.png')
        #    plt.close()
        
        #for i in range(10):
        #    plt.plot(eeg_mat[0,:,5*i], label = "eeg")
        #    plt.show()
        #for i in range(10):
        #    plt.plot(fmri_mat[0,:,5*i], label = "fmri")
        #    plt.show()
        
        
        #ipdb.set_trace()
        # region_assignment: {EEG_electrodes: brain region}
        inv_mapping = {} #{brain region: EEG_electrodes}
        for k, v in region_assignment.items():
            for _v in v:
                if _v not in inv_mapping:
                    inv_mapping[_v] = []
                inv_mapping[_v] = sorted(list(set(inv_mapping[_v]+[k])))
        
        basic_len = 2912 # hard-coded fot F_t 582.4, used as the sliding window stride to avoid float
        #adj_mx, fmri_mat, eeg_mat = shuffle(adj_mx, fmri_mat, eeg_mat,random_state = 0)
        assert int(args.seq_length % basic_len) == 0
    
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
            nTrain = 100 # Number of training samples
            nValid = int(0.2 * nTrain) # Number of validation samples
            nTest = int(0.2 * nTrain) # Number of testing samples
            num_timestep = 1200
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
    else:
        sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata,args.adjtype)
        dataloader = util.load_dataset_metr(args.data, args.batch_size, args.batch_size, 
                                            args.batch_size)
        F_t = None
        
    if args.data == 'CRASH':

        # separate adj matrices into train-val-test samples
        #adj_train = [[],[]]
        adj_train = [[]]
        for a in adj_mx[:nTrain]:
            #adj_train.append(a)
            adj_train[0].append(a[0])
            #adj_train[1].append(a[1])
        adj_train = [np.stack(np.asarray(i)) for i in adj_train]

        print(np.asarray(adj_train).shape)
        #adj_val = [[],[]]
        adj_val = [[]]
        for a in adj_mx[nTrain:-nTest]:
            adj_val[0].append(a[0])
            #adj_val[0].append(a[0])
            #adj_val[1].append(a[1])
        adj_val = [np.stack(np.asarray(i)) for i in adj_val]

        #adj_test = [[],[]]
        adj_test = [[]]
        for a in adj_mx[-nTest:]:
            adj_test[0].append(a[0])
            #adj_test[0].append(a[0])
            #adj_test[1].append(a[1])
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

        offset = args.seq_length // basic_len
        E_idxer = np.arange(args.seq_length+500)[None, :] + np.arange(0, 
                            eeg_mat.shape[1] - args.seq_length*2 + 1, basic_len*2)[:, None]

        K = int(args.seq_length / F_t)
        F_idxer = np.arange(10)[None, :] + np.arange(0, fmri_mat.shape[1] - 10 + 1, 
                                                10)[:, None]
        print("Shape of fmri is ", fmri_mat.shape)
        print("Shape of eeg is ", eeg_mat.shape)
        # x = F_idxer[:-offset]
        # y = F_idxer[offset:]

        assert len(F_idxer) == len(E_idxer)
        #sample_per_suj = len(F_idxer) - offset
        #print(len(E_idxer))
        #print(len(F_idxer))
        sample_per_suj = len(F_idxer) - offset
        batch_per_sub = sample_per_suj // args.batch_size
        print('Batch per subject:', batch_per_sub)
        # ipdb.set_trace() # sample_per_suj, batch_per_sub
        
        #GPUtil.showUtilization()
        engine = trainer([],args.in_dim, args.seq_length, args.num_nodes, 
                         args.nhid, args.dropout, args.learning_rate, args.weight_decay, device, 
                         supports, args.gcn_bool, args.addaptadj, adjinit, args.kernel_size,
                         args.blocks, args.layers, out_nodes=eeg_mat.shape[-1], F_t=F_t)

        old_state_dict = {}
        
        for key in engine.model.state_dict():
            old_state_dict[key] = engine.model.state_dict()[key].clone()
        
        if model_name is None:

            print("start training...",flush=True)

            his_loss =[]
            val_time = []
            train_time = []
            min_loss = float('Inf')
            train_loss_arr = []
            valid_loss_arr = []
            for i in range(1,args.epochs+1):
                if i % 500 == 0:
                    # lr = max(0.000002,args.learning_rate * (0.1 ** (i // 10)))
                    for g in engine.optimizer.param_groups:
                        g['lr'] *= 1.00
                train_loss = []
                train_mae = []
                train_mape = []
                train_rmse = []
                train_corr_coeff = []
                t1 = time.time()
                engine.set_state('train')
                
                iter = 0
                # cross-subject shuffle
                train_idx = np.arange(nTrain * batch_per_sub)
                np.random.shuffle(train_idx)
                #print(train_idx.shape)
                for _train_idx in train_idx:
                    subj_id = _train_idx // batch_per_sub
                    batch_i = _train_idx % batch_per_sub
                    
                    #print("TrainSubj_id is " +str(subj_id))
                    #print("TrainBatch_i is " + str(batch_i))
                    # input current F
                    #GPUtil.showUtilization()
                    #if iter % args.print_every == 0 :    
                    #plt.figure()
                    x_F = fmri_mat[subj_id, F_idxer, :][:-offset][
                                batch_i * args.batch_size: (batch_i + 1) * args.batch_size]
                    #print(x_F.shape)
                    x_F = torch.Tensor(x_F[...,None]).transpose(1, 3)
                    #print(x_F.shape)
                    # pred future F
                    y_F = fmri_mat[subj_id, F_idxer, :][offset:][
                                batch_i * args.batch_size: (batch_i + 1) * args.batch_size]
                    #if iter % args.print_every == 0 :
                    #print(y_F.shape)
                    y_F = torch.Tensor(y_F[...,None]).transpose(1, 3)
                    #for k in range(1):
                    #    plt.plot(x_F[k,0,0,:], label = 'input F')
                    #    plt.plot(y_F[k,0,0,:], label = 'output F')
                    #    plt.legend()
                    #    plt.show()
                    #print(y_F.shape)
                    #input current E
                    x_E = eeg_mat[subj_id, E_idxer, :][:-offset][
                                batch_i * args.batch_size: (batch_i + 1) * args.batch_size]
                    x_E = torch.Tensor(x_E[...,None]).to(device).transpose(1, 3)
                    #if iter % args.print_every == 0 :    
                    #for k in range(4):
                    #    plt.plot(x_E[k,:,0], label = 'input E')
                    #   plt.show()
                    # pred future E
                    y_E = eeg_mat[subj_id, E_idxer, :][offset:][
                                batch_i * args.batch_size: (batch_i + 1) * args.batch_size]
                    y_E = torch.Tensor(y_E[...,None]).transpose(1, 3)
                    # map to current E
                    # y_E = eeg_mat[subj_id, E_idxer, :][:-offset][
                    #batch_i * args.batch_size: (batch_i + 1) * args.batch_size]                    
                    #print(x_E.shape)
                    #print(x_E.shape)

                    #plt.figure()
                    #for i in range (1):
                    #    plt.figure()
                        #plt.plot(x_F[0,0,152,:].detach().cpu(), label = 'input F')
                    #    plt.plot(y_F[0,0,152,:].detach().cpu(), label = 'output F')
                    #    plt.legend()
                    #    plt.show()
                    #plt.plot(y_E[0,0,:,:], label = 'output E')
                    #plt.plot(y_F[0,0,0,:].detach().cpu(), label = 'output F')
                    #if iter % args.print_every == 0 :    
                    #plt.legend()
                    #print("plt is showing")
                    #plt.show()
                    #print("plt has shown")

                    #GPUtil.showUtilization()
                    metrics = engine.train_CRASH(x_E, y_F, y_E, region_assignment, 
                                                [subj_id]*args.batch_size)
                        
                    train_loss.append(metrics[0])
                    train_mae.append(metrics[1])
                    train_mape.append(metrics[2])
                    train_rmse.append(metrics[3])
                    train_corr_coeff.append(metrics[5])
                    if iter % args.print_every == 0 :
                        log = 'Iter: {:03d}, Train Loss: {:.6f}, Train MAE: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                        print(log.format(iter, train_loss[-1], train_mae[-1], train_mape[-1], train_rmse[-1]),flush=True)
                        F = metrics[4].detach().cpu()
                        #for k in range(2):
                            #plt.figure()
                            #plt.plot(y_F[0,0,k,:], label = 'real F')
                            #plt.plot(F[0,0,k,:].detach().cpu(), label = 'pred F')
                            #plt.legend()     
                            #plt.show()
                    iter += 1
                    #break # overfit single batch

                if(i%100 == 0):   
                    for i in range(5):   
                        #plt.figure()
                        #y_F = y_F.squeeze()
                        #num_subj,n_f, t_f = y_F.shape
                        #y_F = y_F.reshape(num_subj*n_f, -1)
                        #print(subj_id)
                        #print(batch_i * args.batch_size)
                        #y_F = y_F / scaler_F.scale_[F_idxer][
                        #        batch_i * args.batch_size: (batch_i + 1) * args.batch_size][0]
                        #y_F = y_F.reshape(num_subj, n_f, t_f)
                        #F = F.squeeze()
                        #num_subj,n_f, t_f = F.shape
                        #F = F.reshape(num_subj*n_f, -1)
                        #F = F / scaler_F.scale_[F_idxer][
                        #        batch_i * args.batch_size: (batch_i + 1) * args.batch_size][0]
                        #F= F.reshape(num_subj, n_f, t_f)
                        #for k in range(4):
                        plt.plot(y_F[0,k,:], label = 'real F')
                        plt.plot(F[0,k,:].detach().cpu(), label = 'pred F')
                        plt.legend()
                        plt.show()

                t2 = time.time()
                train_time.append(t2-t1)

                #validation
                valid_loss = []
                valid_mae = []
                valid_mape = []
                valid_rmse = []
                valid_coeff = []
               
                flag = True
                s1 = time.time()
                engine.set_state('val')
                #engine.set_state('train')
                for subj_id in range(nValid):
                    # for F&E: nTrain + subj_id; for adj_idx: subj_id (supports[state] counts from 0)
                    # subj_F = scaler_F.transform(fmri_mat[nTrain + subj_id, F_idxer, :])
                    subj_F = fmri_mat[nTrain + subj_id, F_idxer, :]
                    # E is only for outputs
                    # subj_E =  scaler_E.transform(eeg_mat[nTrain + subj_id, E_idxer, :][offset:]) 
                    subj_E = eeg_mat[nTrain + subj_id, E_idxer, :]
                    # subj_E = eeg_mat[nTrain + subj_id, E_idxer, :][:-offset]
                    
                    ###overfit single batch
                    #subj_F = fmri_mat[subj_id, F_idxer, :]
                    #subj_E = eeg_mat[subj_id, E_idxer, :]
                    #####

                    for batch_i in range(batch_per_sub):
                        #print("ValSubj_id " + str(subj_id))
                        #print("ValBatch_i " + str(batch_i))
                        x_F = subj_F[:-offset][batch_i * args.batch_size: (batch_i + 1) * args.batch_size]
                        x_F = torch.Tensor(x_F[...,None]).transpose(1, 3)
                        
                        x_E = subj_E[:-offset][batch_i * args.batch_size: (batch_i + 1) * args.batch_size]
                        x_E = torch.Tensor(x_E[...,None]).to(device).transpose(1, 3)
                        
                        y_F = subj_F[offset:][batch_i * args.batch_size: (batch_i + 1) * args.batch_size]
                        y_F = torch.Tensor(y_F[...,None]).transpose(1, 3)
                        
                        y_E = subj_E[offset:][batch_i * args.batch_size: (batch_i + 1) * args.batch_size]
                        y_E = torch.Tensor(y_E[...,None]).transpose(1, 3)
                        
                        #plt.figure()
                        #plt.plot(x_E[0,0,0,:].detach().cpu(), label = 'input E')
                        #plt.plot(y_E[0,0,:,:], label = 'output E')
                        #plt.plot(y_F[0,0,0,:].detach().cpu(), label = 'output F')
                        #plt.legend()
                        #plt.show()
                        #plt.figure()
                        #plt.plot(y_F[0,0,0,:].detach().cpu(), label = 'output F')
                        #plt.show()
                        
                        metrics = engine.eval_CRASH(x_E, y_F, y_E, region_assignment, 
                                                    [subj_id]*args.batch_size)
                        
                        #if(flag):
                            #F = metrics[4]
                            #for k in range(5):
                            #    plt.figure()
                            #    plt.plot(y_F[0,0,k+10,:], label = 'real F')
                            #    plt.plot(F[0,0,k+10,:].detach().cpu(), label = 'pred F')
                            #    plt.legend()
                            #    plt.show()
                            #flag = False

                        valid_loss.append(metrics[0])
                        valid_mae.append(metrics[1])
                        valid_mape.append(metrics[2])
                        valid_rmse.append(metrics[3])
                        valid_coeff.append(metrics[5])
                        #break
                    #break

                s2 = time.time()
                log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
                print(log.format(i,(s2-s1)))
                val_time.append(s2-s1)

                mtrain_loss = np.mean(train_loss)
                mtrain_mae = np.mean(train_mae)
                mtrain_mape = np.mean(train_mape)
                mtrain_rmse = np.mean(train_rmse)
                mtrain_coeff = np.mean(train_corr_coeff)

                train_loss_arr.append(mtrain_loss)

                mvalid_loss = np.mean(valid_loss)
                mvalid_mae = np.mean(valid_mae)
                mvalid_mape = np.mean(valid_mape)
                mvalid_rmse = np.mean(valid_rmse)
                mvalid_coeff = np.mean(valid_coeff)
                his_loss.append(mvalid_loss)
                
                valid_loss_arr.append(mvalid_loss)

                log = 'Epoch: {:03d}, Train Loss: {:.6f}, Train MAE: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Train Coeff: {:.4f}, Valid Loss: {:.4f}, Valid MAE: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Valid Coeff: {:.4f}, Training Time: {:.4f}/epoch'
                print(log.format(i, mtrain_loss, mtrain_mae, mtrain_mape, mtrain_rmse, mtrain_coeff,  mvalid_loss, mvalid_mae, mvalid_mape, mvalid_rmse,mvalid_coeff,(t2 - t1)),flush=True)
                if mvalid_loss < min_loss:
                    min_loss = mvalid_loss
                    # remove previous
                    #fname = glob.glob(args.save+'_epoch_*.pth')
                    #for f in fname:
                    #    os.remove(f)
                    # save new
                    torch.save(engine.model.state_dict(), args.save+"_epoch_"+str(i)+"_"+str(round(mvalid_loss,2))+".pth")
                #torch.save(engine.model.state_dict(), args.save+"_epoch_"+str(i)+"_"+str(round(mvalid_loss,2))+".pth")
            print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
            print("Average Inference Time: {:.4f} secs".format(np.mean(val_time))) 
            
            plt.figure()
            plt.plot(train_loss_arr)
            plt.plot(valid_loss_arr)
            plt.show()

    elif args.data == 'syn' and not same_G: # different graph structure for each sample
        assert len(adj_mx) == nTrain + nValid + nTest

        # separate adj matrices into train-val-test samples
        adj_train = [[],[]]
        for a in adj_mx[:nTrain]:
            adj_train[0].append(a[0])
            adj_train[1].append(a[1])
        adj_train = [np.stack(np.asarray(i)) for i in adj_train]

        adj_val = [[],[]]
        for a in adj_mx[nTrain:-nTest]:
            adj_val[0].append(a[0])
            adj_val[1].append(a[1])
        adj_val = [np.stack(np.asarray(i)) for i in adj_val]

        adj_test = [[],[]]
        for a in adj_mx[-nTest:]:
            adj_test[0].append(a[0])
            adj_test[1].append(a[1])
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
                engine.set_state('train')

                for iter, (x, y, adj_idx) in enumerate(dataloader['train_loader'].get_iterator()):
                    trainx = torch.Tensor(x).to(device) # torch.Size([64, 15, 80, 2])
                    trainx= trainx.transpose(1, 3) # torch.Size([64, 2, 80, 15])
                    trainy = torch.Tensor(y).to(device)
                    trainy = trainy.transpose(1, 3)

                    metrics = engine.train_syn(trainx, trainy, G['train'], adj_idx)
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
                engine.set_state('val')
                for iter, (x, y, adj_idx) in enumerate(dataloader['val_loader'].get_iterator()):
                    testx = torch.Tensor(x).to(device)
                    testx = testx.transpose(1, 3)
                    testy = torch.Tensor(y).to(device)
                    testy = testy.transpose(1, 3)
                    # [64, 2, 80, 15]
                    metrics = engine.eval_syn(testx, testy, G['val'], adj_idx)
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
                torch.save(engine.model.state_dict(), args.save+"_epoch_"+str(i)+"_"+str(round(mvalid_loss,2))+".pth")
            print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
            print("Average Inference Time: {:.4f} secs".format(np.mean(val_time))) 

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
                torch.save(engine.model.state_dict(), args.save+"_epoch_"+str(i)+"_"+str(round(mvalid_loss,2))+".pth")
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
        engine.model.load_state_dict(torch.load(model_name))
    amae = []
    amape = []
    armse = []
    test_loss = []
    test_coeff = []
    
    new_state_dict = {}
    for key in engine.model.state_dict():
        new_state_dict[key] = engine.model.state_dict()[key].clone()
    #for key in old_state_dict:
        #if not (old_state_dict[key] == new_state_dict[key]).all():
            #print('Diff in {}'.format(key))
    
    if args.data == 'CRASH':
        engine.set_state('test')
        #engine.set_state('train') #overfit single batch
        real_Fs = []
        real_Es = []
        pred_Fs = []
        pred_Es = []
        for subj_id in range(nTest):
            # for F&E: nTrain + nValid + subj_id; for adj_idx: subj_id (supports[state] counts from 0)
            #subj_F = scaler_F.transform(fmri_mat[nTrain + nValid + subj_id, F_idxer, :])
            #subj_F = fmri_mat[subj_id, F_idxer, :]
            #test train data
            subj_F = fmri_mat[subj_id + nValid + nTrain, F_idxer, :]
            # E is only for outputs
            # subj_E =  scaler_E.transform(eeg_mat[nTrain + nValid + subj_id, E_idxer, :][offset:])
            #subj_E =  eeg_mat[subj_id, E_idxer, :][offset:]
            #test train data
            subj_E =  eeg_mat[subj_id + nValid + nTrain, E_idxer, :] 
        
            
            for batch_i in range(batch_per_sub):
                x_F = subj_F[:-offset][batch_i * args.batch_size: (batch_i + 1) * args.batch_size]
                x_F = torch.Tensor(x_F[...,None]).transpose(1, 3)
                
                x_E = subj_E[:-offset][batch_i * args.batch_size: (batch_i + 1) * args.batch_size]
                x_E = torch.Tensor(x_E[...,None]).to(device).transpose(1, 3)
                
                y_F = subj_F[offset:][batch_i * args.batch_size: (batch_i + 1) * args.batch_size]
                y_F = torch.Tensor(y_F[...,None]).transpose(1, 3)
                
                y_E = subj_E[offset:][batch_i * args.batch_size: (batch_i + 1) * args.batch_size]
                y_E = torch.Tensor(y_E[...,None]).transpose(1, 3)

                metrics = engine.eval_CRASH(x_E, y_F, y_E, region_assignment, 
                                            [subj_id]*args.batch_size)
                test_loss.append(metrics[0])
                amae.append(metrics[1])
                amape.append(metrics[2])
                armse.append(metrics[3])
                test_coeff.append(metrics[5])
                real_Fs.append(y_F)
                #real_Es.append(y_E)
                #print(y_F.shape)
                pred_Fs.append(metrics[-2])
                #print(metrics[-2].shape)
                #pred_Es.append(metrics[-1])

                if batch_i % 10 == 0:
                    for i in range(i):
                        plt.figure()
                        plt.plot(y_F[i,0,:,:].detach().cpu(), label='real Fs')
                        plt.plot(metrics[-2][0,0,i,:].detach().cpu(), label='pred Fs')
                        plt.legend()
                        plt.show()
        
        real_Fs = torch.stack(real_Fs).cpu().numpy()
        #print("Real F shape after stack:")
        print(real_Fs.shape)
        real_Fs = real_Fs.reshape(-1, *real_Fs.shape[2:])
        #real_Es = torch.stack(real_Es).cpu().numpy()
        #real_Es = real_Es.reshape(-1, *real_Es.shape[2:])

        pred_Fs = [pred_F.cpu().numpy() for pred_F in pred_Fs]
        pred_Fs = np.stack(pred_Fs)
        #print("Pred f shape after stack:")
        print(pred_Fs.shape)
        pred_Fs = pred_Fs.reshape(-1, *pred_Fs.shape[2:]).squeeze()
        #pred_Es = [pred_E.cpu().numpy() for pred_E in pred_Es]
        #pred_Es = np.stack(pred_Es)
        #pred_Es = pred_Es.reshape(-1, *pred_Es.shape[2:]).squeeze()
        # reals shape: (1984, 2, 80, 15); pred_F/Es shape:(1984, 80, 15)

        # reverse slideing window --> results: (num_nodes, total_timesteps)
        # ret = util.inverse_sliding_window([reals[:, 0, :, :].squeeze(), pred_Es])
        # concatenate

        print(real_Fs.shape)
        #real_Fs = real_Fs.squeeze()
        #print(real_Fs.shape)
        #real_Fs = real_Fs.reshape((len(real_Fs), -1))
        #real_Fs = scaler_F.inverse_transform(real_Fs.reshape((len(real_Fs), -1)))
        #real_Fs = real_Fs.reshape(576, 200,10)
        #real_Fs = real_Fs.transpose(1,0,2)
        #real_Fs = real_Fs.reshape((len(real_Fs), -1))
        #print(real_Fs.shape)
        real_Fs = real_Fs.squeeze().transpose(1,0,2)
        real_Fs = real_Fs.reshape((len(real_Fs), -1))

        print(pred_Fs.shape)
        #pred_Fs = scaler_F.inverse_transform(pred_Fs.reshape((len(pred_Fs), -1)))
        #pred_Fs = pred_Fs.reshape(576,200,10)
        #pred_Fs = pred_Fs.transpose(1,0,2)
        #pred_Fs = pred_Fs.reshape((len(pred_Fs),-1))
        #print(pred_Fs.shape)
        pred_Fs = pred_Fs.transpose(1,0,2)
        pred_Fs = pred_Fs.reshape((len(pred_Fs),-1))

        #for i in range(10):
        #    viz_node_idx = i+10
        #    viz_num = 5000 # time length of visualization
        #    plt.figure()
        #    plt.plot(real_Fs[viz_node_idx, :viz_num], label='real F')
        #    plt.plot(pred_Fs[viz_node_idx, :viz_num], label='pred F')
        #    plt.legend()
        #plt.figure()
        #plt.plot(real_Es[viz_node_idx, :viz_num], label='real E')
        #plt.plot(pred_Es[viz_node_idx, :viz_num], label='pred E')
        #plt.legend()
        #    plt.show()

        if model_name is None:
            log = 'On average over seq_length horizons, Test Loss: {:.4f}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}, Test Coeff: {:.4f}'
            print(np.mean(test_loss))
            print(np.mean(amae))
            print(np.mean(amape))
            print(np.mean(armse))
            print(np.mean(test_coeff))
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
                amae.append(metrics[2])
                amape.append(metrics[3])
                armse.append(metrics[4])

        else:
            engine.set_state('test')
            in_Fs = []
            reals = []
            # pred_Fs = []
            pred_Es = []
            for iter, (x, y, adj_idx) in enumerate(dataloader['test_loader'].get_iterator()):
                testx = torch.Tensor(x).to(device)
                testx = testx.transpose(1, 3)
                testy = torch.Tensor(y).to(device)
                testy = testy.transpose(1, 3)
                # [64, 2, 80, 15]
                metrics = engine.eval_syn(testx, testy, G['test'], adj_idx)
                amae.append(metrics[2])
                amape.append(metrics[3])
                armse.append(metrics[4])

                in_Fs.append(testx)
                reals.append(testy)
                pred_Es.append(metrics[-1])
            
            in_Fs = torch.stack(in_Fs).cpu().numpy()
            in_Fs = in_Fs.reshape(-1, *in_Fs.shape[2:]).squeeze()[:95]
            reals = torch.stack(reals).cpu().numpy()
            reals = reals.reshape(-1, *reals.shape[2:]).squeeze()[:95]
            pred_Es = torch.stack(pred_Es).cpu().numpy()
            pred_Es = pred_Es.reshape(-1, *pred_Es.shape[2:]).squeeze()[:95]
            # reals shape: (1984, 2, 80, 15); pred_F/Es shape:(1984, 80, 15)

            # reverse slideing window --> results: (num_nodes, total_timesteps)
            ret = util.inverse_sliding_window([in_Fs, reals, pred_Es], [1]+[F_t]*2)
            viz_node_idx = 0
            plt.figure()
            plt.plot(ret[0][viz_node_idx, :].repeat(F_t), label='in F')
            plt.plot(ret[1][viz_node_idx, :], label='real E')
            plt.plot(ret[2][viz_node_idx, :], label='pred E')
            plt.legend()
            plt.show()

        if model_name is None:
            log = 'On average over seq_length horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
            print(log.format(np.mean(amae),np.mean(amape),np.mean(armse)))
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
        print(log.format(torch.mean(torch.stack(amae)),torch.mean(torch.stack(amape)),torch.mean(torch.stack(armse))))
        torch.save(engine.model.state_dict(), args.save+"_exp"+str(args.expid)+"_best_"+str(round(his_loss[bestid],2))+".pth")

if __name__ == "__main__":
    t1 = time.time()
    # main('garage/syn_epoch_24940_0.0.pth', finetune=True)
    # main('garage/syn_epoch_95_0.1.pth', syn_file='syn_batch32_diffG.pkl', scatter=True)
    # main('garage/CRASH_wavelet_mae_epoch_5_4.02.pth',scatter=True, map=True)
    # main('garage/syn_epoch_5988_0.04.pth')
    #main(syn_file='syn_batch32_diffG.pkl', scatter=True)
    # main(syn_file='syn_batch32_diffG.pkl')
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2-t1))
