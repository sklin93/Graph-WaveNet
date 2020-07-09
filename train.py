import torch
import numpy as np
import argparse
import time
import Utils.util as util
from engine import trainer
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle
import os
import ipdb

# python train.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --num_nodes 207 --seq_length 12 --save ./garage/metr
# python train.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --num_nodes 80 --data syn --blocks 2 --layers 3 --in_dim=1
# python train.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --data CRASH --num_nodes 200 --seq_length 2912 --in_dim 1 --blocks 2 --layers 2 --batch_size 16 --save ./garage/CRASH
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
parser.add_argument('--seq_length',type=int,default=48,help='')
parser.add_argument('--nhid',type=int,default=32,help='')
parser.add_argument('--in_dim',type=int,default=2,help='inputs dimension')
parser.add_argument('--num_nodes',type=int,default=80,help='number of nodes')
parser.add_argument('--layers',type=int,default=2,help='number of layers per gwnet block')
parser.add_argument('--blocks',type=int,default=4,help='number of blocks in gwnet model')

parser.add_argument('--batch_size',type=int,default=32,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--epochs',type=int,default=100,help='')
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

def main(model_name=None, syn_file='syn_diffG.pkl'): # directly loading trained model/ generated syn data
    #set seed
    #torch.manual_seed(args.seed)
    #np.random.seed(args.seed)
    #load data
    same_G = False
    device = torch.device(args.device)

    if args.data == 'CRASH':
        adj_mx, fmri_mat, eeg_mat, region_assignment, F_t = util.load_dataset_CRASH(args.adjtype)
        # region_assignment: {EEG_electrodes: brain region}
        inv_mapping = {} #{brain region: EEG_electrodes}
        for k, v in region_assignment.items():
            for _v in v:
                if _v not in inv_mapping:
                    inv_mapping[_v] = []
                inv_mapping[_v] = sorted(list(set(inv_mapping[_v]+[k])))
        
        basic_len = 2912 # hard-coded fot F_t 582.4, used as the sliding window stride to avoid float
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
            nTrain = 50 # Number of training samples
            nValid = int(0.25 * nTrain) # Number of validation samples
            nTest = int(0.05 * nTrain) # Number of testing samples
            num_timestep = 1000 # 1000
            dataloader, adj_mx, F_t, G = util.load_dataset_syn(args.adjtype, args.num_nodes,
                                                               nTrain, nValid, nTest, num_timestep,
                                                               args.seq_length, args.batch_size, 
                                                               args.batch_size, args.batch_size, 
                                                               same_G=same_G)
            # pkl_data = {'nTrain': nTrain, 'nValid': nValid, 'nTest': nTest,
            #             'num_timestep': num_timestep, 'dataloader': dataloader,
            #             'adj_mx': adj_mx, 'F_t': F_t, 'G':G}
            # with open(syn_file, 'wb') as handle:
            #     pickle.dump(pkl_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata,args.adjtype)
        dataloader = util.load_dataset_metr(args.data, args.batch_size, args.batch_size, 
                                            args.batch_size)
        
    if args.data == 'CRASH':
        nTrain = round(0.7 * len(adj_mx))
        nValid = round(0.15 * len(adj_mx))
        nTest = len(adj_mx) - nTrain - nValid

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
        # ipdb.set_trace() # sample_per_suj, batch_per_sub

        engine = trainer([scaler_F,scaler_E], args.in_dim, args.seq_length, args.num_nodes, 
                         args.nhid, args.dropout, args.learning_rate, args.weight_decay, device, 
                         supports, args.gcn_bool, args.addaptadj, adjinit, args.blocks, args.layers)

        if model_name is None:
            print("start training...",flush=True)

            his_loss =[]
            val_time = []
            train_time = []
            for i in range(1,args.epochs+1):
                if i % 5 == 0:
                    # lr = max(0.000002,args.learning_rate * (0.1 ** (i // 10)))
                    for g in engine.optimizer.param_groups:
                        g['lr'] *= 0.9
                train_loss = []
                train_mape = []
                train_rmse = []
                t1 = time.time()
                engine.set_state('train')

                iter = 0
                # TODO: cross-subject shuffle
                # shuffle subject
                train_idx = np.arange(nTrain)
                np.random.shuffle(train_idx)
                for subj_id in train_idx:
                    subj_F = scaler_F.transform(fmri_mat[subj_id, F_idxer, :])
                    # E is only for outputs
                    subj_E =  scaler_E.transform(eeg_mat[subj_id, E_idxer, :][offset:]) 
                    # shuffle batch
                    batch_idx = np.arange(batch_per_sub)
                    np.random.shuffle(batch_idx)
                    for batch_i in batch_idx:
                        x_F = subj_F[:-offset][batch_i * args.batch_size: (batch_i + 1) * args.batch_size]
                        # x = []
                        # for i in range(K):
                        #     rpt_t = round((i+1)*F_t) - round(i*F_t)
                        #     x.append(x_F[:, i:i+1, :].repeat(rpt_t, axis=1))
                        # x = np.concatenate(x, axis=1)
                        # x = torch.Tensor(x[...,None]).to(device).transpose(1, 3)
                        x_F = torch.Tensor(x_F[...,None]).to(device).transpose(1, 3)
                        y_F = subj_F[offset:][batch_i * args.batch_size: (batch_i + 1) * args.batch_size]
                        y_F = torch.Tensor(y_F[...,None]).transpose(1, 3)
                        y_E = subj_E[batch_i * args.batch_size: (batch_i + 1) * args.batch_size]
                        y_E = torch.Tensor(y_E[...,None]).transpose(1, 3)
                        
                        metrics = engine.train_CRASH(x_F, y_F, y_E, F_t, region_assignment, 
                                                    [subj_id]*args.batch_size)
                        train_loss.append(metrics[0])
                        train_mape.append(metrics[1])
                        train_rmse.append(metrics[2])
                        if iter % args.print_every == 0 :
                            log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                            print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]),flush=True)
                        iter += 1

                t2 = time.time()
                train_time.append(t2-t1)

                #validation
                valid_loss = []
                valid_mape = []
                valid_rmse = []

                s1 = time.time()
                engine.set_state('val')
                for subj_id in range(nValid):
                    # for F&E: nTrain + subj_id; for adj_idx: subj_id (supports[state] counts from 0)
                    subj_F = scaler_F.transform(fmri_mat[nTrain + subj_id, F_idxer, :])
                    # E is only for outputs
                    subj_E =  scaler_E.transform(eeg_mat[nTrain + subj_id, E_idxer, :][offset:]) 
                    for batch_i in range(batch_per_sub):
                        x_F = subj_F[:-offset][batch_i * args.batch_size: (batch_i + 1) * args.batch_size]
                        x_F = torch.Tensor(x_F[...,None]).to(device).transpose(1, 3)
                        
                        y_F = subj_F[offset:][batch_i * args.batch_size: (batch_i + 1) * args.batch_size]
                        y_F = torch.Tensor(y_F[...,None]).transpose(1, 3)
                        y_E = subj_E[batch_i * args.batch_size: (batch_i + 1) * args.batch_size]
                        y_E = torch.Tensor(y_E[...,None]).transpose(1, 3)

                        metrics = engine.eval_CRASH(x_F, y_F, y_E, F_t, region_assignment, 
                                                    [subj_id]*args.batch_size)

                        valid_loss.append(metrics[0])
                        valid_mape.append(metrics[1])
                        valid_rmse.append(metrics[2])

                s2 = time.time()
                log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
                print(log.format(i,(s2-s1)))
                val_time.append(s2-s1)

                mtrain_loss = np.mean(train_loss)
                mtrain_mape = np.mean(train_mape)
                mtrain_rmse = np.mean(train_rmse)

                mvalid_loss = np.mean(valid_loss)
                mvalid_mape = np.mean(valid_mape)
                mvalid_rmse = np.mean(valid_rmse)
                his_loss.append(mvalid_loss)

                log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
                print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)),flush=True)
                torch.save(engine.model.state_dict(), args.save+"_epoch_"+str(i)+"_"+str(round(mvalid_loss,2))+".pth")
            print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
            print("Average Inference Time: {:.4f} secs".format(np.mean(val_time))) 

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
        
        scaler = dataloader['scaler']
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


        engine = trainer(scaler, args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                         args.learning_rate, args.weight_decay, device, supports, args.gcn_bool, 
                         args.addaptadj, adjinit, args.blocks, args.layers)

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

                    metrics = engine.train_syn(trainx, trainy, F_t, G['train'], adj_idx)
                    train_loss.append(metrics[0])
                    train_mape.append(metrics[1])
                    train_rmse.append(metrics[2])
                    if iter % args.print_every == 0 :
                        log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                        print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]),flush=True)

                t2 = time.time()
                train_time.append(t2-t1)

                #validation
                valid_loss = []
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
                    metrics = engine.eval_syn(testx, testy, F_t, G['val'], adj_idx)
                    valid_loss.append(metrics[0])
                    valid_mape.append(metrics[1])
                    valid_rmse.append(metrics[2])
                s2 = time.time()
                log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
                print(log.format(i,(s2-s1)))
                val_time.append(s2-s1)
                mtrain_loss = np.mean(train_loss)
                mtrain_mape = np.mean(train_mape)
                mtrain_rmse = np.mean(train_rmse)

                mvalid_loss = np.mean(valid_loss)
                mvalid_mape = np.mean(valid_mape)
                mvalid_rmse = np.mean(valid_rmse)
                his_loss.append(mvalid_loss)

                log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
                print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)),flush=True)
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
                             args.addaptadj, adjinit, args.blocks, args.layers)

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
                        metrics = engine.train_syn(trainx, trainy, F_t, G)
                    else:
                        metrics = engine.train(trainx, trainy[:,0,:,:])
                    train_loss.append(metrics[0])
                    train_mape.append(metrics[1])
                    train_rmse.append(metrics[2])
                    if iter % args.print_every == 0 :
                        log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                        print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]),flush=True)
            
                t2 = time.time()
                train_time.append(t2-t1)
                #validation
                valid_loss = []
                valid_mape = []
                valid_rmse = []

                s1 = time.time()
                for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
                    testx = torch.Tensor(x).to(device)
                    testx = testx.transpose(1, 3)
                    testy = torch.Tensor(y).to(device)
                    testy = testy.transpose(1, 3)
                    if args.data == 'syn':
                        metrics = engine.eval_syn(testx, testy, F_t, G)
                    else:
                        metrics = engine.eval(testx, testy[:,0,:,:])
                    valid_loss.append(metrics[0])
                    valid_mape.append(metrics[1])
                    valid_rmse.append(metrics[2])

                s2 = time.time()
                log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
                print(log.format(i,(s2-s1)))
                val_time.append(s2-s1)
                mtrain_loss = np.mean(train_loss)
                mtrain_mape = np.mean(train_mape)
                mtrain_rmse = np.mean(train_rmse)

                mvalid_loss = np.mean(valid_loss)
                mvalid_mape = np.mean(valid_mape)
                mvalid_rmse = np.mean(valid_rmse)
                his_loss.append(mvalid_loss)

                log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
                print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)),flush=True)
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

    if args.data == 'CRASH':
        engine.set_state('test')
        reals = []
        pred_Es = []
        for subj_id in range(nTest):
            # for F&E: nTrain + nValid + subj_id; for adj_idx: subj_id (supports[state] counts from 0)
            subj_F = scaler_F.transform(fmri_mat[nTrain + nValid + subj_id, F_idxer, :])
            # E is only for outputs
            subj_E =  scaler_E.transform(eeg_mat[nTrain + nValid + subj_id, E_idxer, :][offset:]) 
            for batch_i in range(batch_per_sub):
                x_F = subj_F[:-offset][batch_i * args.batch_size: (batch_i + 1) * args.batch_size]
                x_F = torch.Tensor(x_F[...,None]).to(device).transpose(1, 3)
                
                y_F = subj_F[offset:][batch_i * args.batch_size: (batch_i + 1) * args.batch_size]
                y_F = torch.Tensor(y_F[...,None]).transpose(1, 3)
                y_E = subj_E[batch_i * args.batch_size: (batch_i + 1) * args.batch_size]
                y_E = torch.Tensor(y_E[...,None]).transpose(1, 3)

                metrics = engine.eval_CRASH(x_F, y_F, y_E, F_t, region_assignment, 
                                            [subj_id]*args.batch_size)

                amae.append(metrics[0])
                amape.append(metrics[1])
                armse.append(metrics[2])

                reals.append(y_E)
                pred_Es.append(metrics[-1])
        
        reals = torch.stack(reals).cpu().numpy()
        reals = reals.reshape(-1, *reals.shape[2:])
        pred_Es = torch.stack(pred_Es).cpu().numpy()
        pred_Es = pred_Es.reshape(-1, *pred_Es.shape[2:]).squeeze()
        # ipdb.set_trace()
        # reals shape: (1984, 2, 80, 15); pred_F/Es shape:(1984, 80, 15)

        # reverse slideing window --> results: (num_nodes, total_timesteps)
        # ret = util.inverse_sliding_window([reals[:, 0, :, :].squeeze(), pred_Es])
        # concatenate
        reals = reals.squeeze().transpose(1,0,2)
        reals = reals.reshape((len(reals), -1))
        pred_Es = pred_Es.transpose(1,0,2)
        pred_Es = pred_Es.reshape((len(pred_Es),-1))
        viz_node_idx = 0
        plt.figure()
        plt.plot(reals[viz_node_idx, :], label='real E')
        plt.plot(pred_Es[viz_node_idx, :], label='pred E')
        plt.legend()
        plt.show()

        if model_name is None:
            log = 'On average over seq_length horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
            print(log.format(np.mean(amae),np.mean(amape),np.mean(armse)))
            torch.save(engine.model.state_dict(), args.save+"_exp"+str(args.expid)+"_best_"+str(round(his_loss[bestid],2))+".pth")
    
    elif args.data == 'syn':
        if same_G:
            for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
                testx = torch.Tensor(x).to(device)
                testx = testx.transpose(1, 3)
                testy = torch.Tensor(y).to(device)
                testy = testy.transpose(1, 3)
                # [64, 2, 80, 15]
                metrics = engine.eval_syn(testx, testy, F_t, G)
                amae.append(metrics[0])
                amape.append(metrics[1])
                armse.append(metrics[2])

        else:
            engine.set_state('test')
            reals = []
            pred_Fs = []
            pred_Es = []
            for iter, (x, y, adj_idx) in enumerate(dataloader['test_loader'].get_iterator()):
                testx = torch.Tensor(x).to(device)
                testx = testx.transpose(1, 3)
                testy = torch.Tensor(y).to(device)
                testy = testy.transpose(1, 3)
                # [64, 2, 80, 15]
                metrics = engine.eval_syn(testx, testy, F_t, G['val'], adj_idx)
                amae.append(metrics[0])
                amape.append(metrics[1])
                armse.append(metrics[2])

                reals.append(testy)
                pred_Fs.append(metrics[3])
                pred_Es.append(metrics[4])

            
            reals = torch.stack(reals).cpu().numpy()
            reals = reals.reshape(-1, *reals.shape[2:])
            pred_Fs = torch.stack(pred_Fs).cpu().numpy()
            pred_Fs = pred_Fs.reshape(-1, *pred_Fs.shape[2:]).squeeze()
            pred_Es = torch.stack(pred_Es).cpu().numpy()
            pred_Es = pred_Es.reshape(-1, *pred_Es.shape[2:]).squeeze()
            # reals shape: (1984, 2, 80, 15); pred_F/Es shape:(1984, 80, 15)

            # reverse slideing window --> results: (num_nodes, total_timesteps)
            ret = util.inverse_sliding_window([reals[:, 0, :, :].squeeze(), 
                                         reals[:, 1, :, :].squeeze(),
                                         pred_Fs, pred_Es])

            viz_node_idx = 0
            plt.figure()
            plt.plot(ret[0][viz_node_idx, :], label='real F')
            plt.plot(ret[1][viz_node_idx, :], label='real E')
            plt.plot(ret[2][viz_node_idx, :], label='pred F')
            plt.plot(ret[3][viz_node_idx, :], label='pred E')
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
            amae.append(metrics[0])
            amape.append(metrics[1])
            armse.append(metrics[2])

        log = 'On average over seq_length horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(np.mean(amae),np.mean(amape),np.mean(armse)))
        torch.save(engine.model.state_dict(), args.save+"_exp"+str(args.expid)+"_best_"+str(round(his_loss[bestid],2))+".pth")


if __name__ == "__main__":
    t1 = time.time()
    # main('garage/CRASH_epoch_97_0.61.pth')
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2-t1))
