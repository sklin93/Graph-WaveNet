import torch
import numpy as np
import argparse
import time
import Utils.util as util
import matplotlib.pyplot as plt
from engine import trainer
import ipdb

# python train.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --num_nodes 207 --seq_length 12 --save ./garage/metr
# python train.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --num_nodes 80 --data syn --blocks 5

parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--data',type=str,default='data/METR-LA',help='data path')
parser.add_argument('--adjdata',type=str,default='data/sensor_graph/adj_mx.pkl',help='adj data path')
parser.add_argument('--adjtype',type=str,default='doubletransition',help='adj type')
parser.add_argument('--gcn_bool',action='store_true',help='whether to add graph convolution layer')
parser.add_argument('--aptonly',action='store_true',help='whether only adaptive adj')
parser.add_argument('--addaptadj',action='store_true',help='whether add adaptive adj')
parser.add_argument('--randomadj',action='store_true',help='whether random initialize adaptive adj')
parser.add_argument('--seq_length',type=int,default=15,help='')
parser.add_argument('--nhid',type=int,default=32,help='')
parser.add_argument('--in_dim',type=int,default=2,help='inputs dimension')
parser.add_argument('--num_nodes',type=int,default=80,help='number of nodes')
parser.add_argument('--layers',type=int,default=2,help='number of layers per gwnet block')
parser.add_argument('--blocks',type=int,default=4,help='number of blocks in gwnet model')

parser.add_argument('--batch_size',type=int,default=64,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--epochs',type=int,default=100,help='')
parser.add_argument('--print_every',type=int,default=50,help='')
#parser.add_argument('--seed',type=int,default=99,help='random seed')
parser.add_argument('--save',type=str,default='./garage/syn',help='save path')
parser.add_argument('--expid',type=int,default=1,help='experiment id')

args = parser.parse_args()




def main():
    #set seed
    #torch.manual_seed(args.seed)
    #np.random.seed(args.seed)
    #load data
    same_G = False
    device = torch.device(args.device)

    if args.data == 'syn':
        nTrain = 40 # Number of training samples
        nValid = int(0.25 * nTrain) # Number of validation samples
        nTest = int(0.05 * nTrain) # Number of testing samples
        num_timestep = 1000
        dataloader, adj_mx, F_t, G = util.load_dataset_syn(args.adjtype, args.num_nodes,
                                                           nTrain, nValid, nTest, num_timestep,
                                                           args.seq_length, args.batch_size, 
                                                           args.batch_size, args.batch_size, 
                                                           same_G=same_G)
    else:
        sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata,args.adjtype)
        dataloader = util.load_dataset_metr(args.data, args.batch_size, args.batch_size, 
                                            args.batch_size)
        

    if args.data == 'syn' and not same_G: # different graph structure for each sample
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
        supports['train'] = [torch.tensor(i).to(device) for i in adj_train]
        supports['val'] = [torch.tensor(i).to(device) for i in adj_val]
        supports['test'] = [torch.tensor(i).to(device) for i in adj_test]

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
    bestid = np.argmin(his_loss)

    print("Training finished")
    print("The valid loss on best model is", str(round(his_loss[bestid],4)))

    engine.model.load_state_dict(torch.load(args.save+"_epoch_"+str(bestid+1)+"_"+str(round(his_loss[bestid],2))+".pth"))

    amae = []
    amape = []
    armse = []

    if args.data == 'syn':
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
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2-t1))
