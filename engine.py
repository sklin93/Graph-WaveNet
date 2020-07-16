import torch.optim as optim
from model import *
import Utils.util as util
from torch.utils.checkpoint import checkpoint
import ipdb

class trainer():
    def __init__(self, scaler, in_dim, seq_length, num_nodes, nhid , 
                dropout, lrate, wdecay, device, supports, gcn_bool,
                addaptadj, aptinit, kernel_size, blocks, layers, 
                out_nodes=None, F_t=None):

        if type(supports) == dict: # different graph structure for each sample
            assert out_nodes is not None, 'need out_nodes number'
            assert F_t is not None, 'need F_t number'
            for k in supports: 
                supports_len = len(supports[k])
                break
            if gcn_bool and addaptadj:
                supports_len += 1
            out_dim_f = int(seq_length//F_t)
            self.model = gwnet_diff_G(device, num_nodes, dropout, supports_len,
                               gcn_bool=gcn_bool, addaptadj=addaptadj,
                               in_dim=in_dim, out_dim=seq_length, out_dim_f=out_dim_f, 
                               residual_channels=nhid, dilation_channels=nhid, 
                               skip_channels=nhid*8, end_channels=nhid*16,
                               kernel_size=kernel_size, blocks=blocks, layers=layers, 
                               out_nodes=out_nodes)
        else:
            self.model = gwnet(device, num_nodes, dropout, supports=supports, 
                               gcn_bool=gcn_bool, addaptadj=addaptadj, aptinit=aptinit, 
                               in_dim=in_dim, out_dim=seq_length, residual_channels=nhid, 
                               dilation_channels=nhid, skip_channels=nhid*8, end_channels=nhid*16,
                               kernel_size=kernel_size, blocks=blocks, layers=layers)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = util.masked_mse #util.masked_mae
        self.scaler = scaler
        self.clip = 1 # TODO: tune, original 5
        self.supports = supports
        self.aptinit = aptinit
        self.state = None
        self.device = device
        self.F_t = F_t

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
                aptinit = aptinit[adj_idx]

            output = self.model(input, supports, aptinit)
        else:
            output = self.model(input)  #[batch_size, seq_len, num_nodes, 1]

        if pooltype == 'None':
            predict = output[-1].permute(0,3,1,2)
            predict = self.scaler[1].inverse_transform(predict)
        
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

        # loss = self.loss(torch.cat((F, predict), 1), real, 0.0)
        loss = self.loss(predict, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(), mae, mape, rmse

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
            aptinit = aptinit[adj_idx]

        predict = self.model(input, supports, aptinit)
        
        if pooltype == 'None':
            F = predict[0].transpose(1,3)
            E = predict[-1].permute(0,3,1,2)
            # E = self.scaler[1].inverse_transform(E)

        elif pooltype == 'avg':
            predict = predict.transpose(1,3)
            # TODO: F from predict & expand (F_t not int, cannot directly using reshape&mean)
            # ipdb.set_trace()
            # F = predict.reshape(*predict.shape[:-1], -1, self.F_t).mean(-1)
            # # F shape
            # F = F.unsqueeze(-1).repeat(*[1]*len(F.shape), self.F_t)
            # F = F.view(*F.shape[:-2], -1)
            # F = self.scaler[0].inverse_transform(F)

            # E from predict & expand, assign_dict the same (common EEG-brain region mapping)
            E = []
            for k in range(len(assign_dict)):
                E.append(predict[:,:,assign_dict[k],:].mean(2, keepdim=True))
            E = torch.cat(E, dim=2)
            # E = self.scaler[1].inverse_transform(E)
            
        # loss = (self.loss(F.cpu(), real_F, 0.0) + self.loss(predict.cpu(), real_E, 0.0)).to(self.device)
        real_F = real_F.to(self.device)
        real_E = real_E.to(self.device)
        # ipdb.set_trace()
        loss = self.loss(F, real_F, 0.0) + self.loss(E, real_E, 0.0)
        # loss = self.loss(E.cpu(), real_E, 0.0).to(self.device)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(E,real_E,0.0).item()
        mape = util.masked_mape(E,real_E, 0.0).item()
        rmse = util.masked_rmse(E,real_E, 0.0).item()
        return loss.item(), mae, mape, rmse

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
                aptinit = aptinit[adj_idx]

            with torch.no_grad():
                output = self.model(input, supports, aptinit)
        
        predict = output.transpose(1,3)

        if pooltype == 'None':
            predict = output[-1].permute(0,3,1,2)
            predict = self.scaler[1].inverse_transform(predict)

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

        # loss = self.loss(torch.cat((F, predict), 1), real, 0.0)
        loss = self.loss(predict, real, 0.0)
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(), mae, mape, rmse, predict

    def eval_CRASH(self, input, real_F, real_E, assign_dict, adj_idx, pooltype='None'):
        self.model.eval()
        input = nn.functional.pad(input,(1,0,0,0))

        supports = self.state_supports
        supports = [supports[i][adj_idx] for i in range(len(supports))]
        aptinit = self.aptinit[self.state]
        if aptinit is not None:
            aptinit = aptinit[adj_idx]

        with torch.no_grad():
            predict = self.model(input, supports, aptinit)

        if pooltype == 'None':
            F = predict[0].transpose(1,3)
            E = predict[-1].permute(0,3,1,2)
            # E = self.scaler[1].inverse_transform(E)

        if pooltype == 'avg':
            predict = predict.transpose(1,3)
            # E from predict & expand
            E = []
            for k in range(len(assign_dict)):
                E.append(predict[:,:,assign_dict[k],:].mean(2, keepdim=True))
            E = torch.cat(E, dim=2)
            # E = self.scaler[1].inverse_transform(E)
        
        real_F = real_F.to(self.device)    
        real_E = real_E.to(self.device)
        loss = self.loss(E, real_E, 0.0)
        # loss = self.loss(E.cpu(), real_E).to(self.device)
        mae = util.masked_mae(E,real_E,0.0).item()
        mape = util.masked_mape(E,real_E, 0.0).item()
        rmse = util.masked_rmse(E,real_E, 0.0).item()
        return loss.item(), mae, mape, rmse, F, E
