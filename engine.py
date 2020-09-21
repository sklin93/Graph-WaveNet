import torch.optim as optim
import torch.nn as nn
from model import *
import Utils.util as util
from torch.utils.checkpoint import checkpoint
import ipdb

def MSPELoss(yhat,y):
    return torch.mean((yhat-y)**2 / y**2)

# class MSLELoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.mse = nn.MSELoss()
        
#     def forward(self, pred, actual):
#         return self.mse(torch.log(pred + 1), torch.log(actual + 1))

class trainer():
    def __init__(self, scaler, in_dim, seq_length, num_nodes, nhid , 
                dropout, lrate, wdecay, device, supports, gcn_bool,
                addaptadj, aptinit, kernel_size, blocks, layers, 
                out_nodes=None, F_t=None, meta=None, subsample=False, 
                scatter=False):
        '''
        - F_t is the time interval between each F (input signals) using E (output) as the scale
        - subsample controls whether output E is subsampled to the same temporal resolution as input F
        - meta controls whether the model with predict scattering coefficients
        - scatter controls whether the model contains scattering layer 
          (if not, model will use conv layers to learn the coefficients)
        '''
        if type(supports) == dict: # different graph structure for each sample
            assert out_nodes is not None, 'need out_nodes number'
            assert F_t is not None, 'need F_t number'
            for k in supports: 
                supports_len = len(supports[k])
                break
            if gcn_bool and addaptadj:
                supports_len += 1

            out_dim_f = int(seq_length//F_t)
            if meta is None:
                if subsample:
                    out_dim = out_dim_f #subsampled E
                else:
                    out_dim = seq_length #original E
            else: #wavelet scattered E
                if scatter:
                    out_dim = seq_length
                else:
                    # out_dim = 1890 # real data 
                    out_dim = 30 # syn

            self.model = gwnet_diff_G(device, num_nodes, dropout, supports_len,
                               gcn_bool=gcn_bool, addaptadj=addaptadj,
                               in_dim=in_dim, out_dim=out_dim, out_dim_f=out_dim_f,
                               residual_channels=nhid, dilation_channels=nhid, 
                               skip_channels=nhid*8, end_channels=nhid*16,
                               kernel_size=kernel_size, blocks=blocks, layers=layers, 
                               out_nodes=out_nodes, meta=meta,
                               scatter=scatter)
        else:
            self.model = gwnet(device, num_nodes, dropout, supports=supports, 
                               gcn_bool=gcn_bool, addaptadj=addaptadj, aptinit=aptinit, 
                               in_dim=in_dim, out_dim=seq_length, residual_channels=nhid, 
                               dilation_channels=nhid, skip_channels=nhid*8, end_channels=nhid*16,
                               kernel_size=kernel_size, blocks=blocks, layers=layers)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = util.masked_mae #util.masked_mse
        # self.loss = nn.MSELoss() # MSPELoss
        # self.loss = nn.SmoothL1Loss() # HuberLoss
        self.scaler = scaler
        self.clip = 1 # TODO: tune, original 5
        self.supports = supports
        self.aptinit = aptinit
        self.state = None
        self.device = device
        self.F_t = F_t
        self.meta = meta
        self.scatter = scatter

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        input = nn.functional.pad(input,(1,0,0,0))
        output = self.model(input)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)

        loss = self.loss(predict, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(), mae, mape,rmse

    def set_state(self, state):
        assert state in ['train', 'val', 'test']
        self.state = state
        self.state_supports = [torch.tensor(i).to(self.device) for i in self.supports[state]]

    def train_syn(self, input, real, G, adj_idx=None, pooltype='None'):
        '''output p=1 sequence, then deteministically subsample/average to F and E'''

        self.model.train()
        self.optimizer.zero_grad()
        # input = input[:,:1,:,:] #only use F as input
        input = nn.functional.pad(input,(1,0,0,0))

        if adj_idx is not None: # different graph structure for each sample
            assert self.state is not None, 'set train/val/test state first'

            supports = self.state_supports
            supports = [supports[i][adj_idx] for i in range(len(supports))]
            aptinit = self.aptinit[self.state]
            if aptinit is not None:
                aptinit = torch.Tensor(aptinit[adj_idx]).to(self.device)

            output = self.model(input, supports, aptinit)
        else:
            output = self.model(input)  #[batch_size, seq_len, num_nodes, 1]

        if pooltype == 'None':
            if self.meta is None:
                # predict = output[-1].transpose(1,3)
                predict = output[-1].squeeze()
                predict = self.scaler[1].inverse_transform(predict)
            else:
                if self.scatter:
                    pred_sig = output[0].squeeze()
                    predict = output[-1].squeeze()
                    # # scale
                    # # TODO: now only predict order0
                    # # make mean and std the same shape as predict
                    # _mean = torch.Tensor(np.tile(self.scaler['means'][None,...][:,:,self.meta[0]], 
                    #                 (len(predict),1,predict.shape[-1]))).to(self.device)
                    # _std = torch.Tensor(np.tile(self.scaler['stds'][None,...][:,:,self.meta[0]], 
                    #                 (len(predict),1,predict.shape[-1]))).to(self.device)
                    # predict = ((predict - _mean)/_std)
                else:
                    ipdb.set_trace() #TODO
        
        elif pooltype == 'avg':
            predict = output.transpose(1,3)
            # F from predict & expand
            # F = predict.reshape(*predict.shape[:-1], -1, self.F_t).mean(-1)
            # F = F.unsqueeze(-1).repeat(*[1]*len(F.shape), self.F_t)
            # F = F.view(*F.shape[:-2], -1)
            # E from predict & expand
            if not type(G) == list:
                # if all the graphs share a same cluster structure
                assign_dict = G.assign_dict
                for k in range(len(assign_dict)):
                    predict[:,:,assign_dict[k],:] = predict[:,:,assign_dict[k],:].\
                                mean(2, keepdim=True).repeat(1,1,len(assign_dict[k]),1)
                predict = self.scaler[1].inverse_transform(predict)
            else: # different graph structure for each sample
                _predict = torch.zeros_like(real)
                for sample in range(len(predict)):
                    assign_dict = G[adj_idx[sample]].assign_dict
                    for k in range(len(assign_dict)):
                        # predict[sample:sample+1, :,assign_dict[k],:] = \
                        # predict[sample:sample+1,:,assign_dict[k],:].mean(2, 
                        #     keepdim=True).repeat(1,1,len(assign_dict[k]),1)
                        _predict[sample:sample+1, :, k, :] = predict[sample:sample+1, 
                                                        :, assign_dict[k], :].mean(2)
                predict = self.scaler[1].inverse_transform(_predict)

        elif pooltype == 'subsample':
            pass #TODO

        if self.scatter:
            loss = self.loss(pred_sig, real[0].squeeze(), 0.0) + self.loss(predict, real[1].squeeze(), 0.0)\
                    + 0.8 * self.loss(predict[...,self.meta[1],:], real[1].squeeze()[...,self.meta[1],:], 0.0)
            # loss = self.loss(pred_sig, real[0].squeeze()) + self.loss(predict, real[1].squeeze())
        else:
            real = real.squeeze()
            loss = self.loss(predict, real, 0.0)
            # loss = self.loss(torch.cat((F, predict), 1), real, 0.0)

        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        #ipdb.set_trace()
        # print(self.model.end_module_add[1].weight.grad.mean())
        # print(self.model.start_conv.weight.grad.mean())

        if self.scatter:
            mae = util.masked_mae(pred_sig, real[0].squeeze(), 0.0).item()
            mape = util.masked_mape(pred_sig, real[0].squeeze(), 0.0).item()
            rmse = util.masked_rmse(pred_sig, real[0].squeeze(), 0.0).item()
            # cc, _, _ = util.get_cc(pred_sig, real[0].squeeze())
        else:
            mae = util.masked_mae(predict,real,0.0).item()
            mape = util.masked_mape(predict,real,0.0).item()
            rmse = util.masked_rmse(predict,real,0.0).item()

        return loss.item(), mae, mape, rmse#, \
        #self.model.start_conv.weight.grad.mean(),self.model.end_module_add[1].weight.grad.mean()#, cc

    def train_CRASH(self, input, real_F, real_E, assign_dict, adj_idx, pooltype='None'):
        '''output p=1 sequence, then deteministically subsample/average to F and E'''

        # self.model = torch.nn.DataParallel(self.model, device_ids=[0, 1])
        self.model.train()
        self.optimizer.zero_grad()
        input = nn.functional.pad(input,(1,0,0,0))

        assert self.state is not None, 'set train/val/test state first'

        supports = self.state_supports
        supports = [supports[i][adj_idx] for i in range(len(supports))]
        aptinit = self.aptinit[self.state]
        if aptinit is not None:
            aptinit = torch.Tensor(aptinit[adj_idx]).to(self.device)

        output = self.model(input, supports, aptinit)
        
        if pooltype == 'None':
            if self.meta is None:
                F = output[0]#.transpose(1,3)
                # E = output[-1].transpose(1,3)
                # E = torch.cat(output[1:],-1).mean(-1,keepdim=True).transpose(1,3)
                E = output[1].squeeze()
                # E = self.scaler[1].inverse_transform(E)
            else:
                if self.scatter:
                    E = output[0].squeeze()
                    predict = output[1].squeeze()

                    # E = output[-1].squeeze()

                    # E[:,:,self.meta[0]] *= 1000
                    # E[:,:,self.meta[1]] *= 10000
                    # E[:,:,self.meta[2]] *= 100000
                    
                    # # TODO: inverse? transform
                    # mean = torch.Tensor(self.scaler['means'][self.meta[0]][None,...]).repeat(
                    #     E.shape[0],E.shape[1],1).to(self.device) #TODO: train on order0 for now
                    # std = torch.Tensor(self.scaler['stds'][self.meta[0]][None,...]).repeat(
                    #     E.shape[0],E.shape[1],1).to(self.device)
                    # # E = (E * std) + mean
                    # E = ((E - mean)/std)
                else:
                    # output = [tmp.transpose(1,3) for tmp in output]
                    # E = torch.cat(output,-1)
                    E = torch.cat(output,-1).transpose(2,3)

        elif pooltype == 'avg':
            output = output.transpose(1,3)
            # TODO: F from output & expand (F_t not int, cannot directly using reshape&mean)
            # ipdb.set_trace()
            # F = output.reshape(*output.shape[:-1], -1, self.F_t).mean(-1)
            # # F shape
            # F = F.unsqueeze(-1).repeat(*[1]*len(F.shape), self.F_t)
            # F = F.view(*F.shape[:-2], -1)
            # F = self.scaler[0].inverse_transform(F)

            # E from output & expand, assign_dict the same (common EEG-brain region mapping)
            E = []
            for k in range(len(assign_dict)):
                E.append(output[:,:,assign_dict[k],:].mean(2, keepdim=True))
            E = torch.cat(E, dim=2)
            # E = self.scaler[1].inverse_transform(E)
            
        # loss = (self.loss(F.cpu(), real_F, 0.0) + self.loss(output.cpu(), real_E, 0.0)).to(self.device)

        if self.meta is None:
            real_F = real_F.to(self.device)
            real_E = real_E.to(self.device).squeeze()
            loss = self.loss(E, real_E, 0.0) #+ self.loss(F, real_F)
            # loss = self.loss(E.cpu(), real_E).to(self.device)
        else:
            if self.scatter:
                # ipdb.set_trace()
                loss = self.loss(E, real_E[0], 0.0) + self.loss(predict, real_E[1], 0.0)#\
                        # + 0.8 * self.loss(predict[...,self.meta[1],:], real[1].squeeze()[...,self.meta[1],:], 0.0)
                # real_E = real_E.squeeze()
                # loss = self.loss(E, real_E, 0.0)
            else:
                # more penalty on the first order
                loss = 4 * self.loss(E[:,:,self.meta[0]], real_E[:,:,self.meta[0]], 
                    0.0) / 7 + 2 * self.loss(E[:,:,self.meta[1]], real_E[:,:,self.meta[1]], 
                    0.0) / 7 + self.loss(E[:,:,self.meta[2]], real_E[:,:,self.meta[2]], 0.0)
                
                # train on the first order
                # real_E = real_E[:,:,self.meta[0]]

        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()

        if self.scatter:
            mae = util.masked_mae(E,real_E[0],0).item()
            mape = util.masked_mape(E,real_E[0],0).item()
            rmse = util.masked_rmse(E,real_E[0],0).item()
        else:
            mae = util.masked_mae(E,real_E,0).item()
            mape = util.masked_mape(E,real_E,0).item()
            rmse = util.masked_rmse(E,real_E,0).item()            
        return loss.item(), mae, mape, rmse, self.model.skip_convs[0].weight.grad.mean()

    def eval(self, input, real_val):
        self.model.eval()
        input = nn.functional.pad(input,(1,0,0,0))
        output = self.model(input)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(), mae, mape, rmse

    def eval_syn(self, input, real, G, adj_idx=None, pooltype='None'):
        same_G = (type(G) != list)
        self.model.eval()
        # input = input[:,:1,:,:]
        input = nn.functional.pad(input,(1,0,0,0))

        if same_G:
            with torch.no_grad():
                output = self.model(input)

        else:
            assert adj_idx is not None, 'adj index needed.'

            supports = self.state_supports
            supports = [supports[i][adj_idx] for i in range(len(supports))]
            aptinit = self.aptinit[self.state]
            if aptinit is not None:
                aptinit = torch.Tensor(aptinit[adj_idx]).to(self.device)

            with torch.no_grad():
                output = self.model(input, supports, aptinit)

        pred_sig = None
        if pooltype == 'None':
            if self.meta is None:
                # predict = output[-1].transpose(1,3)
                predict = output[-1].squeeze()
                predict = self.scaler[1].inverse_transform(predict)
            else:
                if self.scatter:
                    pred_sig = output[0].squeeze()
                    predict = output[-1].squeeze()
                    # # scale
                    # # TODO: now only predict order0
                    # # make mean and std the same shape as predict
                    # _mean = torch.Tensor(np.tile(self.scaler['means'][None,...][:,:,self.meta[0]], 
                    #                 (len(predict),1,predict.shape[-1]))).to(self.device)
                    # _std = torch.Tensor(np.tile(self.scaler['stds'][None,...][:,:,self.meta[0]], 
                    #                 (len(predict),1,predict.shape[-1]))).to(self.device)
                    # predict = ((predict - _mean)/_std)                  
                else:
                    ipdb.set_trace() #TODO

        elif pooltype == 'avg':
            # F from predict & expand
            # F = predict.reshape(*predict.shape[:-1], -1, self.F_t).mean(-1)
            # F = F.unsqueeze(-1).repeat(*[1]*len(F.shape), self.F_t)
            # F = F.view(*F.shape[:-2], -1)
            # E from predict & expand
            if same_G :
                assign_dict = G.assign_dict
                for k in range(len(assign_dict)):
                    predict[:,:,assign_dict[k],:] = predict[:,:,assign_dict[k],:].\
                                mean(2, keepdim=True).repeat(1,1,len(assign_dict[k]),1)
                predict = self.scaler[1].inverse_transform(predict)
            else:
                _predict = torch.zeros_like(real)
                for sample in range(len(predict)):
                    assign_dict = G[adj_idx[sample]].assign_dict
                    for k in range(len(assign_dict)):
                        # predict[sample:sample+1, :,assign_dict[k],:] = \
                        # predict[sample:sample+1,:,assign_dict[k],:].mean(2, 
                        #     keepdim=True).repeat(1,1,len(assign_dict[k]),1)
                        _predict[sample:sample+1, :, k, :] = predict[sample:sample+1, 
                                                        :, assign_dict[k], :].mean(2)
                predict = self.scaler[1].inverse_transform(_predict)

        elif pooltype == 'subsample':
            pass #TODO

        if self.scatter:
            loss = self.loss(pred_sig, real[0].squeeze(), 0.0) + self.loss(predict, real[1].squeeze(), 0.0)\
                    + 0.8 * self.loss(predict[...,self.meta[1],:], real[1].squeeze()[...,self.meta[1],:], 0.0)
            # loss = self.loss(pred_sig, real[0].squeeze()) + self.loss(predict, real[1].squeeze())
        else:
            real = real.squeeze()
            loss = self.loss(predict, real, 0.0)
            # loss = self.loss(torch.cat((F, predict), 1), real, 0.0)
        
        if self.scatter:
            mae = util.masked_mae(pred_sig, real[0].squeeze(), 0.0).item()
            mape = util.masked_mape(pred_sig, real[0].squeeze(), 0.0).item()
            rmse = util.masked_rmse(pred_sig, real[0].squeeze(), 0.0).item()
            cc, _, _ = util.get_cc(pred_sig, real[0].squeeze())
        else:
            mae = util.masked_mae(predict,real,0.0).item()
            mape = util.masked_mape(predict,real,0.0).item()
            rmse = util.masked_rmse(predict,real,0.0).item()
            cc, _, _ = util.get_cc(pred_sig, real)

        return loss.item(), mae, mape, rmse, cc, pred_sig, predict

    def eval_CRASH(self, input, real_F, real_E, assign_dict, adj_idx, pooltype='None'):
        self.model.eval()
        input = nn.functional.pad(input,(1,0,0,0))

        supports = self.state_supports
        supports = [supports[i][adj_idx] for i in range(len(supports))]
        aptinit = self.aptinit[self.state]
        if aptinit is not None:
            aptinit = torch.Tensor(aptinit[adj_idx]).to(self.device)

        with torch.no_grad():
            output = self.model(input, supports, aptinit)

        F = None
        E = None
        predict = None # E's scattering coefficients

        if pooltype == 'None':
            if self.meta is None:
                F = output[0]#.transpose(1,3)
                # E = output[-1].transpose(1,3)
                # E = torch.cat(output[1:],-1).mean(-1,keepdim=True).transpose(1,3)
                E = output[1].squeeze()
                # E = self.scaler[1].inverse_transform(E)
            else:
                if self.scatter:
                    E = output[0].squeeze()
                    predict = output[1].squeeze()
                    
                    # F = output[0] ### NOT F! just use its place to hold the output signal
                    # E = output[1].squeeze() ### the wavelet coefficients
                    # # E[:,:,self.meta[0]] *= 1000
                    # # E[:,:,self.meta[1]] *= 10000
                    # # E[:,:,self.meta[2]] *= 100000

                    # # # TODO: inverse transform
                    # mean = torch.Tensor(self.scaler['means'][self.meta[0]][None,...]).repeat(
                    #     E.shape[0],E.shape[1],1).to(self.device) #TODO: train on order0 for now
                    # std = torch.Tensor(self.scaler['stds'][self.meta[0]][None,...]).repeat(
                    #     E.shape[0],E.shape[1],1).to(self.device)
                    # # E = (E * std) + mean
                    # E = ((E - mean)/std)
                else:
                    # output = [tmp.transpose(1,3) for tmp in output]
                    # predict = torch.cat(output,-1)
                    predict = torch.cat(output,-1).transpose(2,3)
                    
        if pooltype == 'avg':
            output = output.transpose(1,3)
            # E from output & expand
            E = []
            for k in range(len(assign_dict)):
                E.append(output[:,:,assign_dict[k],:].mean(2, keepdim=True))
            E = torch.cat(E, dim=2)
            # E = self.scaler[1].inverse_transform(E)
        
        if self.meta is None:
            real_F = real_F.to(self.device)
            real_E = real_E.to(self.device).squeeze()
            loss = self.loss(E, real_E, 0.0) #+ self.loss(F, real_F)
            # loss = self.loss(E.cpu(), real_E).to(self.device)
        else:
            if self.scatter:
                loss = self.loss(E, real_E[0], 0.0) + self.loss(predict, real_E[1], 0.0)#\
                        # + 0.8 * self.loss(predict[...,self.meta[1],:], real[1].squeeze()[...,self.meta[1],:], 0.0)
                # real_E = real_E.squeeze()
                # loss = self.loss(E, real_E, 0.0)
            else:
                # more penalty on the first order
                loss = 4 * self.loss(E[:,:,self.meta[0]], real_E[:,:,self.meta[0]], 
                    0.0) / 7 + 2 * self.loss(E[:,:,self.meta[1]], real_E[:,:,self.meta[1]], 
                    0.0) / 7 + self.loss(E[:,:,self.meta[2]], real_E[:,:,self.meta[2]], 0.0)
                # train on the first order
                # real_E = real_E[:,:,self.meta[0]]

        if self.scatter:
            mae = util.masked_mae(E,real_E[0],0).item()
            mape = util.masked_mape(E,real_E[0],0).item()
            rmse = util.masked_rmse(E,real_E[0],0).item()
            cc, _, _ = util.get_cc(E, real_E[0])
        else:
            mae = util.masked_mae(E,real_E,0).item()
            mape = util.masked_mape(E,real_E,0).item()
            rmse = util.masked_rmse(E,real_E,0).item()
            cc, _, _ = util.get_cc(E, real_E)
        return loss.item(), mae, mape, rmse, cc, F, E, predict
