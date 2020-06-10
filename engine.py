import torch.optim as optim
from model import *
import Utils.util as util
import ipdb

class trainer():
    def __init__(self, scaler, in_dim, seq_length, num_nodes, nhid , 
                dropout, lrate, wdecay, device, supports, gcn_bool,
                addaptadj, aptinit, blocks, layers):

        self.model = gwnet(device, num_nodes, dropout, supports=supports, 
                           gcn_bool=gcn_bool, addaptadj=addaptadj, aptinit=aptinit, 
                           in_dim=in_dim, out_dim=seq_length, residual_channels=nhid, 
                           dilation_channels=nhid, skip_channels=nhid*8, end_channels=nhid*16,
                           blocks=blocks, layers=layers)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = util.masked_mae
        self.scaler = scaler
        self.clip = 5

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
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mape,rmse

    def train_syn(self, input, real, F_t, G, pooltype='avg'):
        '''output p=1 sequence, then deteministically subsample/average to F and E'''

        self.model.train()
        self.optimizer.zero_grad()
        input = nn.functional.pad(input,(1,0,0,0))
        output = self.model(input)  #[batch_size, seq_len, num_nodes, 1]
        output = output.transpose(1,3)
        predict = self.scaler.inverse_transform(output)
        
        if pooltype == 'avg':
            # F from predict & expand
            F = predict.reshape(*predict.shape[:-1], -1, F_t).mean(-1)
            F = F.unsqueeze(-1).repeat(*[1]*len(F.shape), F_t)
            F = F.view(*F.shape[:-2], -1)
            # E from predict & expand
            if not type(G) == list:
                # if all the graphs share a same cluster structure
                assign_dict = G.assign_dict
                for k in range(len(assign_dict)):
                    predict[:,:,assign_dict[k],:] = predict[:,:,assign_dict[k],:].\
                                mean(2, keepdim=True).repeat(1,1,len(assign_dict[k]),1)
        elif pooltype == 'subsample':
            pass #TODO

        loss = self.loss(torch.cat((F, predict), 1), real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mape,rmse

    def eval(self, input, real_val):
        self.model.eval()
        input = nn.functional.pad(input,(1,0,0,0))
        output = self.model(input)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mape,rmse
