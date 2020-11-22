import torch.optim as optim
from model import *
import Utils.util as util
from torch.utils.checkpoint import checkpoint
import ipdb
# import GPUtil
from scipy import stats
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D 
import numpy as np
import math

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
                               residual_channels=64, dilation_channels=64, 
                               skip_channels=256, end_channels=nhid*16,
                               kernel_size=kernel_size, blocks=blocks, layers=layers, 
                               out_nodes=out_nodes)
        else:
            self.model = gwnet(device, num_nodes, dropout, supports=supports, 
                               gcn_bool=gcn_bool, addaptadj=addaptadj, aptinit=aptinit, 
                               in_dim=in_dim, out_dim=seq_length, residual_channels=nhid, 
                               dilation_channels=nhid, skip_channels=nhid*8, end_channels=nhid*16,
                               kernel_size=kernel_size, blocks=blocks, layers=layers)
        #GPUtil.showUtilization()
        self.model.to(device)
        #print("After model is loaded to device")
        #GPUtil.showUtilization()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = util.masked_rmse #util.masked_mae
        self.scaler = scaler
        self.clip = None # TODO: tune, original 5
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

        # GPUtil.showUtilization()
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
            predict = output[-1].transpose(1,3)
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
    
    def plot_grad_flow(self, named_parameters):
        ave_grads = []
        max_grads= []
        layers = []
        for n, p in named_parameters:
            #print(n)
            #print(p.grad)
            if(p.requires_grad) and ("bias" not in n) and (p.grad != None):
                layers.append(n)
                ave_grads.append(p.grad.abs().mean())
                max_grads.append(p.grad.abs().max())
        plt.figure()
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
        plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
        plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.legend([Line2D([0], [0], color="c", lw=4),
                    Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
        plt.show()

    def printgradnorm(self, grad_input, grad_output):
        print('Inside ' + self.__class__.__name__ + ' backward')
        print('Inside class:' + self.__class__.__name__)
        print('')
        print('grad_input: ', type(grad_input))
        print('grad_input[0]: ', type(grad_input[0]))
        print('grad_output: ', type(grad_output))
        print('grad_output[0]: ', type(grad_output[0]))
        print('')
        print('grad_input size:', grad_input[0].size())
        print('grad_output size:', grad_output[0].size())
        print('grad_input norm:', grad_input[0].norm())

    def train_CRASH(self, input, real_F, real_E, assign_dict, adj_idx, pooltype='None'):
        '''output p=1 sequence, then deteministically subsample/average to F and E'''

        # self.model = torch.nn.DataParallel(self.model, device_ids=[0, 1])
        self.model.train()
        self.optimizer.zero_grad()
        #input = nn.functional.pad(input,(1,0,0,0))

        assert self.state is not None, 'set train/val/test state first'

        supports = self.state_supports
        supports = [supports[i][adj_idx] for i in range(len(supports))]
        aptinit = self.aptinit[self.state]
        if aptinit is not None:
            aptinit = torch.Tensor(aptinit[adj_idx]).to(self.device)
        
        #self.model.gate_convs.register_backward_hook(self.printgradnorm)
        #GPUtil.showUtilization()
        #print(input.shape)
        #print(real_F.shape)
        #plt.figure()
        #plt.plot(input[0,0,0,:].detach().cpu(), label = 'real E')
        #plt.show()
        #print(torch.stack(supports)>0.1)
        #print(torch.stack(supports)[supports > 0.1])
        predict = self.model(input, supports, aptinit)
        #print(self.model.end_module_2[1].weight.mean())
        #print(self.model.end_module[1].weight.mean())
        #print(self.model.filter_convs[0].weight.mean())
        #print(self.model.start_conv[0].weight.mean())
        #for i in range(6):
        #    print(self.model.skip_convs[i].weight.mean())
             
        if pooltype == 'None':
            F = predict
            #print(F.shape)
            #E = predict.transpose(1,3)
            #E = predict
            #num_subj,_, n_f, t_f = predict.shape
            #F = predict.reshape(num_subj*n_f, -1)
            #F = self.scaler.inverse_transform(F.cpu().detach().numpy())

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
            
        #for i in range(5):
        #    plt.figure()
        #    plt.plot(real_F[0,0,i,:], label = 'real F')
        #    plt.plot(F[0,0,i,:].detach().cpu(), label = 'pred F')
        #    plt.legend()
        #    plt.show()

        # loss = (self.loss(F.cpu(), real_F, 0.0) + self.loss(predict.cpu(), real_E, 0.0)).to(self.device)
        real_F = real_F.to(self.device)
        #print(real_F.shape)
        #real_E = real_E.to(self.device)
        # ipdb.set_trace()
        #loss = self.loss(F, real_F, 0.0) + self.loss(E, real_E, 0.0)
        loss = self.loss(F, real_F, 0.0)
        #print(loss)
        #loss = self.loss(E.cpu(), real_E, 0.0).to(self.device)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        #for p,n in zip(self.model.parameters(),self.model.all_weights[0]):
            #if n[:6] == 'weight':
                #print('===========\ngradient:{}\n----------\n{}'.format(n,p.grad))
        #print(self.model.start_conv[0].weight.grad.abs().mean())
        #print(self.model.start_conv[1].weight.sum())
        #print(self.model.start_conv[2].weight.sum())
        #print(self.model.start_conv[3].weight.sum())
        #print(self.model.start_conv[4].weight.sum())
        #print(self.model.filter_convs[1].weight.grad.abs().mean())
        #print("Sum of grads: " + str(self.model.filter_convs[1].weight.grad.sum()))
        #print("Sum of abs grads: " + str(self.model.filter_convs[1].weight.grad.abs().sum()))
        #print("Len of grads: " + str(self.model.filter_convs[1].weight.grad.shape))
        #print(self.model.filter_convs[0].weight.grad.abs().mean())
        #print(self.model.gate_convs[1].weight.grad.abs().mean())
        #print(self.model.residual_convs[0].weight.grad.mean())
        #print(self.model.skip_convs[0].weight.grad.abs().mean())
        #print(self.model.skip_convs[1].weight.grad.abs().mean())
        #print(self.model.end_module_2[1].weight.grad.mean())
        #print(self.model.end_module[1].weight.grad.mean())
        #print(self.model.filter_convs[5].weight.grad.mean())
        #a = self.model.end_module_2[1].weight.grad.cpu().numpy()
        #print(a.shape)
        #plt.hist(a[:,0,0,:])
        #plt.show()
        #a = self.model.end_module[0].weight.grad.cpu().numpy()
        #print(self.model.end_module[0].weight.grad.mean())
        #print(self.model.filter_convs[0].weight.mean())
        #b = self.model.skip_convs[5].weight.grad.cpu().numpy()
        #plt.hist(b[:,:,0,0])
        #plt.show()
        #b = self.model.skip_convs[3].weight.grad.cpu().numpy()
        #plt.hist(b[:,:,0,0])
        #plt.show()
        
        #print(self.model.end_module_2[1].weight.grad.abs().mean())
        #print(self.model.end_module[1].weight.grad.abs().mean())
        #print(self.model.skip_convs[3].weight.grad.abs().mean())
        #print(self.model.skip_convs[0].weight.grad.abs().mean())
        #print(self.model.filter_convs[5].weight.grad.abs().mean())
        #print(self.model.filter_convs[0].weight.grad.abs().mean())
        #print(self.model.start_conv[0].weight.grad.abs().mean()) 
        #self.plot_grad_flow(self.model.named_parameters())
        #for p in self.model.parameters():
        #    if(p.grad is not None):
        #        print(p.grad[p.grad == 0], 'gradient')
        #self.plot_grad_flow(self.model.named_parameters())
        mae = util.masked_mae(F,real_F,0.0).item()
        mape = util.masked_mape(F,real_F, 0.0).item()
        rmse = util.masked_rmse(F,real_F, 0.0).item()
        corr_mean = []
        for subject in range(F.shape[0]):
            corr_coeff = []
            for node in range(F.shape[2]):
                #print(F[subject,:,node,:].reshape(-1).shape) 
                #print(real_F[subject,:,node,:].reshape(-1).shape)
                corr_coeff.append(stats.pearsonr(F[subject,:,node,:].reshape(-1).detach().cpu(), real_F[subject,:,node,:].reshape(-1).detach().cpu())[0])
                #if(math.isnan(corr_coeff[node])):
                #    print("subject : " + str(subject))
                #    print("node : " + str(node))
                #    plt.figure()
                #    plt.plot(real_F[subject,:,node,:].reshape(-1).detach().cpu())
                #    plt.show()
                #    print(real_F[subject,:,node,:].reshape(-1))
                #    print(F[subject,:,node,:].reshape(-1))
                #print(corr_coeff[node])
            #if(len(np.argwhere(np.isnan(corr_coeff))) != 0):
             #   print(real_F[subject,:,node,:].reshape(-1))
             #   print(F[subject,:,node,:].reshape(-1))
            #print("subject: " + str(subject) + " ")
            #print(corr_coeff.shape)
            #print(np.argwhere(np.isnan(corr_coeff)))    
            corr_mean.append(np.nanmean(corr_coeff))
        corr_total_mean = np.nanmean(corr_mean)
        #print(corr_total_mean)
        return loss.item(), mae, mape, rmse, F, corr_total_mean

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

        if pooltype == 'None':
            predict = output[-1].transpose(1, 3)
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
        #input = nn.functional.pad(input,(1,0,0,0))

        supports = self.state_supports
        supports = [supports[i][adj_idx] for i in range(len(supports))]
        aptinit = self.aptinit[self.state]
        if aptinit is not None:
            aptinit = torch.Tensor(aptinit[adj_idx]).to(self.device)    
        
        with torch.no_grad():
            predict = self.model(input, supports, aptinit)

        #print(self.model.start_conv[0].weight.sum())
        #print(self.model.start_conv[1].weight.sum())
        #print(self.model.start_conv[2].weight.sum())
        #print(self.model.start_conv[3].weight.sum())
        #print(self.model.start_conv[4].weight.sum())
        #print(self.model.filter_convs[5].weight.grad.mean())
        #print(self.model.filter_convs[0].weight.grad.mean())
        #print(self.model.gate_convs[5].weight.grad.mean())
        #print(self.model.residual_convs[0].weight.grad.mean())
        #print(self.model.end_module_2[1].weight.grad.mean())
        #print(self.model.end_module[1].weight.grad.mean())
        #print(self.model.filter_convs[0].weight.grad.mean())
        #print(self.model.skip_convs[5].weight.grad.mean())
        #print(self.model.skip_convs[3].weight.grad.mean())
        #print(self.model.skip_convs[0].weight.grad.mean())       
        
        if pooltype == 'None':
            #F = predict.transpose(1,3)
            #F = predict.transpose(1,3)
            F = predict
            #print(F.shape)
            #plt.figure()
            #plt.plot(real_F[0,0,0,:], label = 'real F')
            #plt.plot(F[0,0,0,:].detach().cpu(), label = 'pred F')
            #plt.legend()
            #plt.show()
            #num_subj,_, n_f, t_f = predict.shape
            #F = predict.reshape(num_subj*n_f, -1)
            #F = self.scaler.inverse_transform(F.cpu().detach().numpy())
            #E = predict
            # E = self.scaler[1].inverse_transform(E)

        elif pooltype == 'avg':
            predict = predict.transpose(1,3)
            # E from predict & expand
            E = []
            for k in range(len(assign_dict)):
                E.append(predict[:,:,assign_dict[k],:].mean(2, keepdim=True))
            E = torch.cat(E, dim=2)
            # E = self.scaler[1].inverse_transform(E)
        
        #real_E = real_E.to(self.device)    
        real_F = real_F.to(self.device)
        loss = self.loss(F, real_F, 0.0)
        #print(loss)
        # loss = self.loss(E.cpu(), real_E).to(self.device)
        mae = util.masked_mae(F,real_F,0.0).item()
        mape = util.masked_mape(F,real_F, 0.0).item()
        rmse = util.masked_rmse(F,real_F, 0.0).item()
        corr_mean = []
        for subject in range(F.shape[0]):
            corr_coeff = []
            for node in range(F.shape[2]):
                #print(F[subject,:,node,:].reshape(-1).shape) 
                #print(real_F[subject,:,node,:].reshape(-1).shape)
                corr_coeff.append(stats.pearsonr(F[subject,:,node,:].reshape(-1).detach().cpu(), real_F[subject,:,node,:].reshape(-1).detach().cpu())[0])
                #print(corr_coeff[node])
            corr_mean.append(np.mean(corr_coeff))
        corr_total_mean = np.mean(corr_mean)
        return loss.item(), mae, mape, rmse, F, corr_total_mean
