import torch.optim as optim
import torch.nn as nn
from model import *
import Utils.util as util
from torch.utils.checkpoint import checkpoint
import matplotlib.pyplot as plt
import ipdb

def MSPELoss(yhat,y):
    return torch.mean((yhat-y)**2 / y**2)

# class MSLELoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.mse = nn.MSELoss()
        
#     def forward(self, pred, actual):
#         return self.mse(torch.log(pred + 1), torch.log(actual + 1))

def weighted_mse(preds, labels, null_val=np.nan):
    # assign more weights to further away points (preds:[16, 10, 200])
    num_t = preds.shape[1]
    wts = (torch.arange(num_t) + 1.0) / (torch.arange(num_t) + 1.0).sum()

    loss = (preds-labels)**2
    loss = torch.mean(loss, [0,2])

    wts = wts.to(loss.device)
    loss = loss * wts
    
    return torch.mean(loss)

class Regress_Loss_1(torch.nn.Module):
    
    def __init__(self):
        super(Regress_Loss_1,self).__init__()
        
    def forward(self,x,y):
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        
        cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        loss = -cost.mean()
        return loss

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.kaiming_uniform(m.weight)
        # torch.nn.init.xavier_uniform_(m.weight)
        # m.weight.data.fill_(0.01)
        m.bias.data.fill_(0.01)
    elif type(m) == nn.Sequential or type(m) == nn.ModuleList:
        for k in m:
            init_weights(k)

class trainer():
    def __init__(self, scaler, in_dim, seq_length, num_nodes, nhid , 
                dropout, lrate, wdecay, device, supports, gcn_bool,
                addaptadj, aptinit, kernel_size, blocks, layers, 
                out_nodes=None, F_t=None, meta=None, subsample=False, 
                scatter=False, F_only=False, batch_size=None):
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
            else:
                out_dim = seq_length

            if F_only:
                self.model = gwnet_diff_G_Fonly(device, num_nodes, dropout, supports_len, batch_size,
                                   gcn_bool=gcn_bool, addaptadj=addaptadj,
                                   in_dim=in_dim, out_dim=out_dim, out_dim_f=out_dim_f,
                                   residual_channels=nhid, dilation_channels=nhid, 
                                   skip_channels=nhid*8, end_channels=nhid*16,
                                   kernel_size=kernel_size, blocks=blocks, layers=layers, 
                                   out_nodes=out_nodes, meta=meta,
                                   scatter=scatter)
            else:
                self.model = gwnet_diff_G(device, num_nodes, dropout, supports_len,
                                   gcn_bool=gcn_bool, addaptadj=addaptadj,
                                   in_dim=in_dim, out_dim=out_dim, out_dim_f=out_dim_f,
                                   residual_channels=nhid, dilation_channels=nhid, 
                                   skip_channels=nhid*8, end_channels=nhid*16,
                                   kernel_size=kernel_size, blocks=blocks, layers=layers, 
                                   out_nodes=out_nodes, meta=meta,
                                   scatter=scatter)              
                                     
            # ### tmp: validate SC (set support len to 1)
            # self.model = gwnet_diff_G(device, num_nodes, dropout, 1,
            #                    gcn_bool=gcn_bool, addaptadj=addaptadj,
            #                    in_dim=in_dim, out_dim=out_dim, out_dim_f=out_dim_f,
            #                    residual_channels=nhid, dilation_channels=nhid, 
            #                    skip_channels=nhid*8, end_channels=nhid*16,
            #                    kernel_size=kernel_size, blocks=blocks, layers=layers, 
            #                    out_nodes=out_nodes, meta=meta,
            #                    scatter=scatter)
            # ###
        else:
            self.model = gwnet(device, num_nodes, dropout, supports=supports, 
                               gcn_bool=gcn_bool, addaptadj=addaptadj, aptinit=aptinit, 
                               in_dim=in_dim, out_dim=int(seq_length//F_t), residual_channels=nhid, 
                               dilation_channels=nhid, skip_channels=nhid*8, end_channels=nhid*16,
                               kernel_size=kernel_size, blocks=blocks, layers=layers)
        # self.model.apply(init_weights)
        self.model.to(device)
        print(self.model)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        # self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        # self.loss = util.masked_mae #util.masked_mse
        # self.loss = nn.L1Loss()
        self.loss = nn.MSELoss() # MSELoss
        # self.loss = weighted_mse
        # self.loss = nn.SmoothL1Loss() # HuberLoss
        # self.loss = nn.CosineEmbeddingLoss()
        self.loss2 = Regress_Loss_1()
        self.scaler = scaler
        self.clip = 1 # TODO: tune, original 5
        self.supports = supports
        self.aptinit = aptinit
        self.state = None
        self.device = device
        self.F_t = F_t
        self.meta = meta
        if scatter == True:
            assert self.meta is not None, 'need meta info if using scattering'
        self.scatter = scatter
        self.F_only = F_only

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
            
            aptinit = self.aptinit
            if aptinit is not None:
                aptinit = aptinit[self.state]
                if aptinit is not None:
                    aptinit = torch.Tensor(aptinit[adj_idx]).to(self.device)

            output = self.model(input, supports, aptinit)
            # output = self.model(input) # validate SC
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
            loss = self.loss(predict, real)
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

    # """
    ### diffG 
    def train_CRASH(self, input, real_F, real_E, assign_dict, adj_idx):
        '''output p=1 sequence, then deteministically subsample/average to F and E'''

        # self.model = torch.nn.DataParallel(self.model, device_ids=[0, 1])
        self.model.train()
        self.optimizer.zero_grad()
        # input = nn.functional.pad(input,(1,0,0,0))
        
        assert self.state is not None, 'set train/val/test state first'

        supports = self.state_supports
        supports = [s[adj_idx] for s in supports]
        
        aptinit = self.aptinit
        if aptinit is not None:
            aptinit = aptinit[self.state]
            if aptinit is not None:
                aptinit = torch.Tensor(aptinit[adj_idx]).to(self.device)

        output = self.model(input, supports, aptinit)
        # output = self.model(input) # validate SC

        if self.F_only:
            F = output.squeeze()
        else:
            if self.meta is None:
                E = output.squeeze()
            else: # with scattering
                E = output[0].squeeze()
                predict = output[1].squeeze()

        ##### loss #####
        if self.meta is None:
            if self.F_only:
                real_F = real_F.to(self.device).squeeze()
                # plt.plot(F[0,0].detach().cpu().numpy())
                # plt.plot(F[0,1].detach().cpu().numpy())
                # plt.show()
                loss = self.loss(F, real_F) #+ self.loss2(F, real_F)
            else:
                real_E = real_E.to(self.device).squeeze()
                # plt.plot(E[0,0].detach().cpu().numpy())
                # plt.plot(E[0,1].detach().cpu().numpy())
                # plt.show()
                # condition = Variable(torch.ones(1, 1), requires_grad=False).to(self.device) # for CosineEmbeddingLoss
                loss = self.loss(E, real_E) #+ self.loss(F, real_F)

        else: # with scattering
            ipdb.set_trace()
            # loss = self.loss(E, real_E[0], 0.0) + self.loss(predict[...,self.meta[0],:], real_E[1][...,self.meta[0],:], 0.0)\
                    # + 0.5 * self.loss(predict[...,self.meta[1],:], real_E[1].squeeze()[...,self.meta[1],:], 0.0)
            loss = 0.5*self.loss(E, real_E[0]) + self.loss(predict[...,self.meta[0],:], real_E[1][...,self.meta[0],:])\
                    + 0.5 * self.loss(predict[...,self.meta[1],:], real_E[1].squeeze()[...,self.meta[1],:])\
                   + 0.5*self.loss2(E, real_E[0]) + self.loss2(predict[...,self.meta[0],:], real_E[1][...,self.meta[0],:])\
                    + 0.5 * self.loss2(predict[...,self.meta[1],:], real_E[1].squeeze()[...,self.meta[1],:])

        #########################

        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()

        if self.scatter:
            mae = util.masked_mae(E,real_E[0],0).item()
            mape = util.masked_mape(E,real_E[0],0).item()
            rmse = util.masked_rmse(E,real_E[0],0).item()
        else:
            if self.F_only:      
                mae = util.masked_mae(F,real_F,0).item()
                mape = util.masked_mape(F,real_F,0).item()
                rmse = util.masked_rmse(F,real_F,0).item()
            else:
                mae = util.masked_mae(E,real_E,0).item()
                mape = util.masked_mape(E,real_E,0).item()
                rmse = util.masked_rmse(E,real_E,0).item()                
        # ipdb.set_trace()
        return loss.item(), mae, mape, rmse, #\
        [abs(self.model.end_module[1].weight.grad).mean(), #abs(self.model.end_mlp_e[1].weight.grad).mean(), 
         abs(self.model.gconv[1].mlp.mlp.weight.grad).mean(), abs(self.model.skip_convs[0].weight.grad).mean(), 
         abs(self.model.filter_convs[1].weight.grad).mean(), abs(self.model.start_conv.weight.grad).mean()]
    
    
    ### sameG debugging purpose
    def train_CRASH2(self, input, real_F, real_E):
        '''output p=1 sequence, then deteministically subsample/average to F and E'''

        # self.model = torch.nn.DataParallel(self.model, device_ids=[0, 1])
        self.model.train()
        self.optimizer.zero_grad()
        # input = nn.functional.pad(input,(1,0,0,0))
        output = self.model(input)
        # output = self.model(input) # validate SC

        if self.F_only:
            F = output.squeeze()
        else:
            if self.meta is None:
                F = output[0].squeeze()
                E = output[1].squeeze()
            else:
                E = output[0].squeeze()
                predict = output[1].squeeze()

        ##### loss #####
        if self.meta is None:
            real_F = real_F.to(self.device).squeeze()
            if self.F_only:
                # plt.plot(F[0,0].detach().cpu().numpy())
                # plt.plot(F[0,1].detach().cpu().numpy())
                # plt.show()
                loss = self.loss(F, real_F) #+ self.loss2(F, real_F)
            else:
                real_E = real_E.to(self.device).squeeze()
                # plt.plot(E[0,0].detach().cpu().numpy())
                # plt.plot(E[0,1].detach().cpu().numpy())
                # plt.show()
                # condition = Variable(torch.ones(1, 1), requires_grad=False).to(self.device) # for CosineEmbeddingLoss
                loss = self.loss(E, real_E) #+ self.loss(F, real_F)

        else:
            if self.scatter:
                # ipdb.set_trace()
                # loss = self.loss(E, real_E[0], 0.0) + self.loss(predict[...,self.meta[0],:], real_E[1][...,self.meta[0],:], 0.0)\
                        # + 0.5 * self.loss(predict[...,self.meta[1],:], real_E[1].squeeze()[...,self.meta[1],:], 0.0)
                loss = 0.5*self.loss(E, real_E[0]) + self.loss(predict[...,self.meta[0],:], real_E[1][...,self.meta[0],:])\
                        + 0.5 * self.loss(predict[...,self.meta[1],:], real_E[1].squeeze()[...,self.meta[1],:])\
                       + 0.5*self.loss2(E, real_E[0]) + self.loss2(predict[...,self.meta[0],:], real_E[1][...,self.meta[0],:])\
                        + 0.5 * self.loss2(predict[...,self.meta[1],:], real_E[1].squeeze()[...,self.meta[1],:])
                # real_E = real_E.squeeze()
                # loss = self.loss(E, real_E, 0.0)
            else:
                # more penalty on the first order
                loss = 4 * self.loss(E[:,:,self.meta[0]], real_E[:,:,self.meta[0]], 
                    0.0) / 7 + 2 * self.loss(E[:,:,self.meta[1]], real_E[:,:,self.meta[1]], 
                    0.0) / 7 + self.loss(E[:,:,self.meta[2]], real_E[:,:,self.meta[2]], 0.0)
                
                # train on the first order
                # real_E = real_E[:,:,self.meta[0]]
        #########################

        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()

        if self.scatter:
            mae = util.masked_mae(E,real_E[0],0).item()
            mape = util.masked_mape(E,real_E[0],0).item()
            rmse = util.masked_rmse(E,real_E[0],0).item()
        else:
            if self.F_only:      
                mae = util.masked_mae(F,real_F,0).item()
                mape = util.masked_mape(F,real_F,0).item()
                rmse = util.masked_rmse(F,real_F,0).item()
            else:
                mae = util.masked_mae(E,real_E,0).item()
                mape = util.masked_mape(E,real_E,0).item()
                rmse = util.masked_rmse(E,real_E,0).item()                
        # ipdb.set_trace()
        return loss.item(), mae, mape, rmse, \
        [abs(self.model.end_conv_1.weight.grad).mean(), #abs(self.model.end_mlp_e[1].weight.grad).mean(), 
         abs(self.model.gconv[1].mlp.mlp.weight.grad).mean(), abs(self.model.skip_convs[0].weight.grad).mean(), 
         abs(self.model.filter_convs[1].weight.grad).mean(), abs(self.model.start_conv.weight.grad).mean()]
    
    
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

    def eval_syn(self, input, real, G, adj_idx=None, viz=False):
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
            supports = [s[adj_idx] for s in supports]
            aptinit = self.aptinit[self.state]
            if aptinit is not None:
                aptinit = torch.Tensor(aptinit[adj_idx]).to(self.device)

            with torch.no_grad():
                output = self.model(input, supports, aptinit, viz=viz)
                # output = self.model(input) # validate SC

        pred_sig = None
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

        if self.scatter:
            loss = self.loss(pred_sig, real[0].squeeze(), 0.0) + self.loss(predict, real[1].squeeze(), 0.0)\
                    + 0.8 * self.loss(predict[...,self.meta[1],:], real[1].squeeze()[...,self.meta[1],:], 0.0)
            # loss = self.loss(pred_sig, real[0].squeeze()) + self.loss(predict, real[1].squeeze())
        else:
            real = real.squeeze()
            if viz:
                print(predict.shape)
                plt.figure()
                for j in range(5):
                    plt.plot(predict.detach().cpu().numpy()[0,j,:])
                plt.figure()
                for j in range(5):
                    plt.plot(real.detach().cpu().numpy()[0,j,:])
                plt.show()         
            loss = self.loss(predict, real)
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
            cc, _, _ = util.get_cc(predict, real)

        return loss.item(), mae, mape, rmse, cc, pred_sig, predict

    # """
    def eval_CRASH(self, input, real_F, real_E, assign_dict, adj_idx, viz=False):
        self.model.eval()
        # input = nn.functional.pad(input,(1,0,0,0))

        supports = self.state_supports
        supports = [s[adj_idx] for s in supports]
        aptinit = self.aptinit
        if aptinit is not None:
            aptinit = aptinit[self.state]
            if aptinit is not None:
                aptinit = torch.Tensor(aptinit[adj_idx]).to(self.device)

        with torch.no_grad():
            output = self.model(input, supports, aptinit, viz=viz)
            # output = self.model(input) # validate SC

        F = None
        E = None
        predict = None # E's scattering coefficients

        if self.F_only:
            F = output.squeeze()
        else:
            if self.meta is None:
                E = output.squeeze()
            else: # with scattering
                E = output[0].squeeze()
                predict = output[1].squeeze()

        ##### loss #####
        if self.meta is None:
            if viz:
                plt.figure('pred')
                for j in range(10):
                    if F is not None:
                        plt.plot(F[0,:,j].cpu().numpy())
                    if E is not None:
                        plt.plot(E[0,:,j].cpu().numpy())
                plt.figure('gt')
                for j in range(10):
                    if real_F is not None:
                        plt.plot(real_F[0,:,j].cpu().numpy())
                    if real_E is not None:
                        plt.plot(real_E[0,:,j].cpu().numpy())
                plt.show()
                ipdb.set_trace()          
            if self.F_only:
                real_F = real_F.to(self.device).squeeze()
                loss = self.loss(F, real_F) #+ self.loss2(F, real_F)
            else:
                real_E = real_E.to(self.device).squeeze()
                # condition = Variable(torch.ones(1, 1), requires_grad=False).to(self.device) # for CosineEmbeddingLoss
                loss = self.loss(E, real_E) #+ self.loss(F, real_F)

        else: # with scattering
            real_E[0] = real_E[0].squeeze()
            real_E[1] = real_E[1].squeeze()
            # loss = self.loss(E, real_E[0], 0.0) + self.loss(predict[...,self.meta[0],:], real_E[1][...,self.meta[0],:], 0.0)\
                    # + 0.5 * self.loss(predict[...,self.meta[1],:], real_E[1].squeeze()[...,self.meta[1],:], 0.0)
            loss = 0.5*self.loss(E, real_E[0]) + self.loss(predict[...,self.meta[0],:], real_E[1][...,self.meta[0],:])\
                    + 0.5 * self.loss(predict[...,self.meta[1],:], real_E[1].squeeze()[...,self.meta[1],:])\
                   + 0.5*self.loss2(E, real_E[0]) + self.loss2(predict[...,self.meta[0],:], real_E[1][...,self.meta[0],:])\
                    + 0.5 * self.loss2(predict[...,self.meta[1],:], real_E[1].squeeze()[...,self.meta[1],:])
        #########################

        ##### metrics #####
        if self.scatter:
            mae = util.masked_mae(E,real_E[0],0).item()
            mape = util.masked_mape(E,real_E[0],0).item()
            rmse = util.masked_rmse(E,real_E[0],0).item()
            
        else:
            if self.F_only:      
                mae = util.masked_mae(F,real_F,0).item()
                mape = util.masked_mape(F,real_F,0).item()
                rmse = util.masked_rmse(F,real_F,0).item()
                cc, _, _ = util.get_cc(F, real_F)
            else:
                mae = util.masked_mae(E,real_E,0).item()
                mape = util.masked_mape(E,real_E,0).item()
                rmse = util.masked_rmse(E,real_E,0).item()
                cc, _, _ = util.get_cc(E, real_E)
        return loss.item(), mae, mape, rmse, cc, F, E, predict
    
    
    ### sameG debugging purpose
    def eval_CRASH2(self, input, real_F, real_E, viz=False):
        self.model.eval()
        # input = nn.functional.pad(input,(1,0,0,0))

        with torch.no_grad():
            output = self.model(input)
            # output = self.model(input) # validate SC

        F = None
        E = None
        predict = None # E's scattering coefficients

        if self.F_only:
            F = output.squeeze()
        else:
            if self.meta is None:
                F = output[0].squeeze()
                E = output[1].squeeze()
            else:
                if self.scatter:
                    E = output[0].squeeze()
                    predict = output[1].squeeze()

                else:
                    predict = torch.cat(output,-1).transpose(2,3)

        ##### loss #####
        if self.meta is None:
            real_F = real_F.to(self.device).squeeze()
            if viz:
                plt.figure('pred')
                for j in range(10):
                    plt.plot(F[0,:,j].cpu().numpy())
                plt.figure('gt')
                for j in range(10):
                    plt.plot(real_F[0,:,j].cpu().numpy())
                plt.show()
            if self.F_only:
                loss = self.loss(F, real_F) #+ self.loss2(F, real_F)
            else:
                real_E = real_E.to(self.device).squeeze()
                # condition = Variable(torch.ones(1, 1), requires_grad=False).to(self.device) # for CosineEmbeddingLoss
                loss = self.loss(E, real_E) #+ self.loss(F, real_F)

        else:
            if self.scatter:
                real_E[0] = real_E[0].squeeze()
                real_E[1] = real_E[1].squeeze()
                # loss = self.loss(E, real_E[0], 0.0) + self.loss(predict[...,self.meta[0],:], real_E[1][...,self.meta[0],:], 0.0)\
                        # + 0.5 * self.loss(predict[...,self.meta[1],:], real_E[1].squeeze()[...,self.meta[1],:], 0.0)
                loss = 0.5*self.loss(E, real_E[0]) + self.loss(predict[...,self.meta[0],:], real_E[1][...,self.meta[0],:])\
                        + 0.5 * self.loss(predict[...,self.meta[1],:], real_E[1].squeeze()[...,self.meta[1],:])\
                       + 0.5*self.loss2(E, real_E[0]) + self.loss2(predict[...,self.meta[0],:], real_E[1][...,self.meta[0],:])\
                        + 0.5 * self.loss2(predict[...,self.meta[1],:], real_E[1].squeeze()[...,self.meta[1],:])
            else:
                # more penalty on the first order
                loss = 4 * self.loss(E[:,:,self.meta[0]], real_E[:,:,self.meta[0]], 
                    0.0) / 7 + 2 * self.loss(E[:,:,self.meta[1]], real_E[:,:,self.meta[1]], 
                    0.0) / 7 + self.loss(E[:,:,self.meta[2]], real_E[:,:,self.meta[2]], 0.0)
        #########################

        ##### metrics #####
        if self.scatter:
            mae = util.masked_mae(E,real_E[0],0).item()
            mape = util.masked_mape(E,real_E[0],0).item()
            rmse = util.masked_rmse(E,real_E[0],0).item()
            
        else:
            if self.F_only:      
                mae = util.masked_mae(F,real_F,0).item()
                mape = util.masked_mape(F,real_F,0).item()
                rmse = util.masked_rmse(F,real_F,0).item()
                cc, _, _ = util.get_cc(F, real_F)
            else:
                mae = util.masked_mae(E,real_E,0).item()
                mape = util.masked_mape(E,real_E,0).item()
                rmse = util.masked_rmse(E,real_E,0).item()
                cc, _, _ = util.get_cc(E, real_E)
        return loss.item(), mae, mape, rmse, cc, F, E, predict
    