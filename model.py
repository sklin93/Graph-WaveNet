import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
# from torch_geometric.nn import SAGPooling, TopKPooling, ASAPooling, PANPooling, max_pool 
import numpy as np
import math
from torch.autograd import Variable
from torch.utils.checkpoint import checkpoint
from kymatio.torch import Scattering1D
import sys
import matplotlib.pyplot as plt
import ipdb
# from torch_geometric.nn import dense_diff_pool

EPS = 1e-15

def dense_diff_pool(x, s, *support):
    # _out = [x]
    s = s.transpose(1,2) # (64,200,100,10) bnct
    s = torch.softmax(s, dim=-2)
    # ipdb.set_trace()
    x = torch.einsum('bfnt,bnct->bfct', x, s).contiguous() #(64, 32, 100, 10) 
    _s = torch.softmax(s.transpose(1,2), dim=-2) # (64, 100, 200, 10) bcnt
    out = torch.einsum('bfct,bcnt->bfnt', x, _s).contiguous()#(64, 32, 200, 10)
    # for a in support:
    #     x = x.unsqueeze(0) if x.dim() == 2 else x
    #     a = a.unsqueeze(0) if a.dim() == 2 else a
    #     s = s.unsqueeze(0) if s.dim() == 2 else s
    #     ipdb.set_trace()
    #     batch_size, num_nodes, _, _ = x.size()
    #     s = torch.softmax(s, dim=-2) #dim -1 or -2 here? originally it is -1
    #     ipdb.set_trace()
    #     # out = torch.matmul(s.transpose(2, 3), x)
    #     out = torch.einsum('bcnt,bfnt->bcft', s, x)
    #     # out_adj = torch.matmul(torch.matmul(s.transpose(2, 3), a), s)
    #     # link_loss = a - torch.matmul(s, s.transpose(2, 3))
    #     # link_loss = torch.norm(link_loss, p=2)
    #     # link_loss = link_loss / a.numel()
    #     # ent_loss = (-s * torch.log(s + EPS)).sum(dim=-1).mean()
    #     _out.append(out)      

    # h = torch.cat(out, dim=1)
    # h_adj = torch.cat(out, dim = 1)
    
    return out #, h_adj, link_loss, ent_loss


# Fourier feature mapping
def fourier_mapping(x, B):
    if B is None:
        return x
    else:
        x_proj = torch.einsum('ncvl,cw->nwvl',(2.*math.pi*x,B.T))
#         x_proj = (2.*math.pi*x) @ B.T
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], axis=1)

class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        # A=A.tranpose(-1,-2)
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class nconv2(nn.Module):
    def __init__(self):
        super(nconv2,self).__init__()

    def forward(self,x, A):
        ### we want: (Ax^T)^T for A's 1,2 channel & x's 1,2 channel == xA^T
        # x = torch.einsum('ncvl,nvw->ncwl',(x,torch.transpose(A, -1, -2)+torch.eye(len(A[0]))[None,...].repeat(len(A),1,1).to(A.device))) # nothing wrong... emm
        x = torch.einsum('ncvl,nvw->ncwl',(x,torch.transpose(A, -1, -2))) #this one's [0,...,0] == torch.matmul(x[0,...,0], A[0])
        # x = torch.einsum('ncwl,nvw->ncvl',(x,A)) #this one's [0,0] == torch.matmul(A[0], x[0,0]))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in #(2*3+1)*32=224
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:
            # a: [207, 207], x:[64, 32, 207, 12]
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1): # k-hop, diffuse one more time per loop
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2
        # len(out)=order*support_len+1 (7); each: [64, 32, 207, 12]
        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class gcn2(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(gcn2,self).__init__()
        self.nconv = nconv2()
        c_in = (order*support_len+1)*c_in #(2*3+1)*32=224
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x, dummy_tensor, *support):
        out = [x]
        for a in support:
            # a: [64, 80, 80], x:[64, 32, 80, 15]
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1): # k-hop, diffuse one more time per loop
                x2 = F.relu(self.nconv(x1,a))
                out.append(x2)
                x1 = x2
        # len(out)=order*support_len+1 (7); each:[64, 32, 80, 15]
        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class gcn3(nn.Module): #GCN module for pooling embedding
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2, lin = False):
        super(gcn3,self).__init__()
        self.nconv = nconv2()
        c_in = (order*support_len+1)*c_in #(2*3+1)*32=224
        self.mlp = linear(c_in,c_out)
        self.mlp2 = linear(200,100) #hard coded
        self.dropout = dropout
        self.order = order
        self.lin = lin

    def forward(self,x, dummy_tensor, *support):
        out = [x]
        for a in support:
            # a: [64, 80, 80], x:[64, 32, 80, 15]
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1): # k-hop, diffuse one more time per loop
                x2 = F.relu(self.nconv(x1,a))
                out.append(x2)
                x1 = x2
        # len(out)=order*support_len+1 (7); each:[64, 32, 80, 15]
        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        if self.lin:
            ipdb.set_trace()
            h = F.relu(self.mlp2(h.transpose(1,2)))
            h = F.dropout(h, self.dropout, training=self.training)
        return h.transpose(1,2)

def CausalConv2d(in_channels, out_channels, kernel_size, dilation=(1,1), **kwargs):
    pad = (kernel_size[1] - 1) * dilation[1]
    return nn.Conv1d(in_channels, out_channels, kernel_size, padding=(0,pad), dilation=dilation, **kwargs)

#strictly pooling at end
class pool(torch.nn.Module):
    def __init__(self,in_channels,num_nodes_eeg,dropout,support_len, non_linearity=torch.tanh):
        super(pool,self).__init__()
        self.in_channels = in_channels
        self.score_layer = gcn2(in_channels, 1, dropout, support_len)
        self.num_nodes_eeg = num_nodes_eeg
        self.non_linearity = non_linearity
    def forward(self, x, *support):
        #x = x.unsqueeze(-1) if x.dim() == 1 else x
        score = self.score_layer(x,*support)
        _,perm = torch.topk(score.squeeze(), self.num_nodes_eeg)
        x = x.permute(0,2,1,3)
        perm = torch.unsqueeze(perm, 2)
        perm = torch.unsqueeze(perm, 3)
        x = torch.gather(x, 1, perm.expand(-1,-1,x.size(2),x.size(3)))
        x = x.permute(0,2,1,3)
        perm = perm.permute(0,2,1,3)
        score = torch.gather(score, 2, perm)
        #find way to index topk nodes from x and from score layer
        x = x * self.non_linearity(score)
        return x

# from gae.layers import GraphConvolution
  
import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        output = self.act(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

# class GCNModelVAE(nn.Module):
#     def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
#         super(GCNModelVAE, self).__init__()
#         self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
#         self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
#         self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
#         self.dc = InnerProductDecoder(dropout, act=lambda x: x)

#     def encode(self, x, adj):
#         hidden1 = self.gc1(x, adj)
#         return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

#     def reparameterize(self, mu, logvar):
#         if self.training:
#             std = torch.exp(logvar)
#             eps = torch.randn_like(std)
#             return eps.mul(std).add_(mu)
#         else:
#             return mu

#     def forward(self, x, adj):
#         mu, logvar = self.encode(x, adj)
#         z = self.reparameterize(mu, logvar)
#         return self.dc(z), mu, logvar


# class InnerProductDecoder(nn.Module):
#     """Decoder for using inner product for prediction."""

#     def __init__(self, dropout, act=torch.sigmoid):
#         super(InnerProductDecoder, self).__init__()
#         self.dropout = dropout
#         self.act = act

#     def forward(self, z):
#         z = F.dropout(z, self.dropout, training=self.training)
#         adj = self.act(torch.mm(z, z.t()))
#         return adj

  
class gwnet(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, gcn_bool=True, addaptadj=True, aptinit=None, in_dim=2,out_dim=12,residual_channels=32,dilation_channels=32,skip_channels=256,end_channels=512,kernel_size=2,blocks=4,layers=2):
        super(gwnet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        self.supports = supports

        receptive_field = 1
        multi_factor = 2 #kernel_size

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if gcn_bool and addaptadj:
            if aptinit is None:
                if supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
                self.supports_len +=1
            else:
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
                self.supports_len += 1




        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,kernel_size),dilation=new_dilation))

                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= multi_factor
                receptive_field += additional_scope
                additional_scope *= multi_factor
                if self.gcn_bool:
                    self.gconv.append(gcn(dilation_channels,residual_channels,dropout,support_len=self.supports_len))



        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)

        self.receptive_field = receptive_field



    def forward(self, input):
        in_len = input.size(3)
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + --> *input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + -------------> *skip*

            #(dilation, init_dilation) = self.dilations[i]

            #residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip


            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x,self.supports)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]


            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x

class gwnet_diff_G(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports_len=0,
                gcn_bool=True, addaptadj=True,
                in_dim=2, out_dim=12, out_dim_f=5,
                residual_channels=32, dilation_channels=32, skip_channels=256,
                # end_channels=512, kernel_size=2, blocks=4, layers=2,
                end_channels=2048, kernel_size=2, blocks=4, layers=2,
                out_nodes=64, meta=None, scatter=False):

        super(gwnet_diff_G, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj
        self.device = device
        self.num_nodes = num_nodes
        self.scatter = scatter
        self.out_dim = out_dim

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)

        if self.gcn_bool and self.addaptadj:
            self.nodevec = nn.Parameter(torch.randn(10, 10).to(self.device), 
                                    requires_grad=True).to(self.device)

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))

        receptive_field = 1
        multi_factor = kernel_size #2
        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1 #4
            for i in range(layers):
                # dilated convolutions
                # TODO: change kernel_size and stride
                self.filter_convs.append(CausalConv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,kernel_size),
                                                   # stride=(1,2),
                                                   dilation=(1,new_dilation)))

                self.gate_convs.append(CausalConv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), 
                                                 # stride=(1,2),
                                                 dilation=(1,new_dilation)))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels)) # comment off for overfitting
                # self.bn.append(nn.Sequential(
                #     nn.Conv2d(in_channels=residual_channels*3, out_channels=residual_channels,
                #               kernel_size=(1,1)),                    
                #     # nn.BatchNorm2d(residual_channels)
                #     ))
                new_dilation *= multi_factor
                receptive_field += additional_scope
                additional_scope *= multi_factor
                if self.gcn_bool:
                    self.gconv.append(gcn2(dilation_channels, residual_channels, dropout,
                                                              support_len=supports_len))

        self.B = 10*np.random.normal(size=(skip_channels*2, skip_channels)).astype(np.float32)
        self.B = torch.tensor(self.B).to(device)
        # self.end_module = nn.Sequential(
        #     # nn.Tanh(), 
        #     nn.LeakyReLU(),
        #     # nn.ReLU(),
        #     nn.Conv2d(in_channels=skip_channels, out_channels=end_channels,
        #               kernel_size=(1,1)),#, bias=True),
        #     # nn.Tanh(), 
        #     nn.LeakyReLU(),
        #     # nn.ReLU(),
        #     nn.Conv2d(in_channels=end_channels, out_channels=end_channels*2,
        #               kernel_size=(1,1)),#, bias=True),            
        #     )
        self.end_module = nn.Sequential(
                # nn.Tanh(), 
                nn.LeakyReLU(),
                # nn.ReLU(),
                nn.Conv2d(in_channels=skip_channels*4, out_channels=end_channels,
                          kernel_size=(1,1), bias=True)
                )     
        # ## temporal transConv
        # convTransK = 7
        # convTransD = 7
        # self.end_module = nn.Sequential(
        #     nn.LeakyReLU(),
        #     nn.ConvTranspose2d(in_channels=skip_channels, out_channels=skip_channels//2,
        #                        kernel_size=(1,convTransK), dilation=convTransD),
        #     nn.LeakyReLU(),
        #     nn.ConvTranspose2d(in_channels=skip_channels//2, out_channels=residual_channels,
        #                        kernel_size=(1,convTransK), dilation=convTransD),
        #     nn.LeakyReLU(),
        #     nn.ConvTranspose2d(in_channels=residual_channels, out_channels=1,
        #                        kernel_size=(1,convTransK), dilation=convTransD),
        #     )

        # ## temporal upsample
        # upScale = 5
        # self.end_module = nn.Sequential(
        #     nn.LeakyReLU(),
        #     nn.Upsample(scale_factor=(1, upScale)),
        #     nn.Conv2d(in_channels=skip_channels, out_channels=skip_channels//2,
        #               kernel_size=(1,1)),#, bias=True),
        #     nn.LeakyReLU(),
        #     nn.Upsample(scale_factor=(1, upScale)),
        #     nn.Conv2d(in_channels=skip_channels//2, out_channels=residual_channels,
        #               kernel_size=(1,1)),#, bias=True),
        #     nn.LeakyReLU(),
        #     nn.Upsample(scale_factor=(1, upScale)),
        #     nn.Conv2d(in_channels=residual_channels, out_channels=1,
        #               kernel_size=(1,1)),#, bias=True),
        #     )

        self.end_module_add = nn.Sequential(
            # nn.Tanh(), 
            nn.LeakyReLU(),
            # nn.ReLU(),
            nn.Conv2d(in_channels=end_channels, out_channels=out_dim,
                      kernel_size=(1,1), bias=True),
            # nn.LeakyReLU(),
            # # nn.ReLU(),
            # nn.Conv2d(in_channels=end_channels*2, out_channels=out_dim,
            #           kernel_size=(1,1), bias=True)            
            )

        self.end_mlp_e = nn.Sequential(
            # nn.Tanh(), 
            # nn.LeakyReLU(),
            # nn.ReLU(),
            nn.Conv2d(in_channels=num_nodes, out_channels=out_nodes,
                      kernel_size=(1,1), bias=True)
            )
        # self.end_mlp_e2 = nn.Sequential(
        #     # nn.Tanh(), 
        #     nn.LeakyReLU(), 
        #     # nn.ReLU(),
        #     nn.Conv2d(in_channels=num_nodes, out_channels=out_nodes,
        #               kernel_size=(1,1), bias=True)
        #     )
        # self.end_mlp_e3 = nn.Sequential(
        #     # nn.Tanh(), 
        #     nn.LeakyReLU(), 
        #     # nn.ReLU(),
        #     nn.Conv2d(in_channels=num_nodes, out_channels=out_nodes,
        #               kernel_size=(1,1), bias=True)
        #     )

        self.end_mlp_f = nn.Sequential(
            # nn.Tanh(),
            nn.LeakyReLU(),
            # nn.ReLU(),
            nn.Conv2d(in_channels=end_channels, out_channels=out_dim_f*4,
                      kernel_size=(1,1), bias=True),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=out_dim_f*4, out_channels=out_dim_f,
                      kernel_size=(1,1), bias=True)
            )
        # self.pooling = pool(out_dim, out_nodes, dropout, supports_len)

        if scatter:
            J = 3
            Q = 9
            self.scattering = Scattering1D(J, out_dim, Q)

        self.receptive_field = receptive_field
        self.meta = meta


    def diconv(self, filter_conv, gate_conv):
        def custom_forward(residual, dummy_tensor):
            # dilated convolution
            filter = filter_conv(residual)
            filter = torch.tanh(filter[..., :-filter_conv.padding[1]])
            gate = gate_conv(residual)
            gate = torch.sigmoid(gate[..., :-gate_conv.padding[1]])
            x = filter * gate

            return x
        return custom_forward

    def skip_part(self, skip_conv, skip):
        def custom_forward(s, dummy_tensor):
            # parametrized skip connection
            s = skip_conv(s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            return skip
        return custom_forward

    def tcn(self, filter_conv, gate_conv, skip_conv, skip):
        '''combine dilated conv + skip parts, entire tcn'''
        def custom_forward(residual, dummy_tensor):
            # dilated convolution
            _filter = filter_conv(residual)
            _filter = torch.tanh(_filter)
            gate = gate_conv(residual)
            gate = torch.sigmoid(gate)
            x = _filter * gate
            # del _filter
            # del _gate

            # parametrized skip connection
            s = x
            s = skip_conv(s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            return x, skip
        return custom_forward            

    def endconv(self, end_conv_1, end_conv_2):
        def custom_forward(x):
            x = F.relu(x)
            x = F.relu(end_conv_1(x))
            x = end_conv_2(x)
            return x
        return custom_forward

    def forward(self, input, supports=None, supports2=None, aptinit=None, viz=False):
        # inputs: [batch_size, 1, num_nodes, in_len+1], supports: len 2, each (batch_size, num_nodes, num_nodes)
        ### deal with supports
        batch_size = len(input)
        if self.gcn_bool and self.addaptadj:
            if supports is None:
                    supports = []
            # if aptinit is None:
            #     nodevec1 = nn.Parameter(torch.randn(batch_size, 
            #                             self.num_nodes, 10).to(self.device), 
            #                             requires_grad=True).to(self.device)
            #     nodevec2 = nn.Parameter(torch.randn(batch_size, 
            #                             10, self.num_nodes).to(self.device), 
            #                             requires_grad=True).to(self.device)

            # else:
            #     m, p, n = torch.svd(aptinit)
            #     _p = torch.diag_embed(p[:, :10] ** 0.5)
            #     initemb1 = torch.bmm(m[:, :, :10], _p)
            #     initemb2 = torch.bmm(_p, torch.transpose(n[:, :, :10], 1, 2))
            #     nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(self.device)
            #     nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(self.device)

        ### normal forward
        # print(self.receptive_field)
        in_len = input.size(3)
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input

        # if viz: # x.shape [16, 1, 200, 15]
        #     for j in range(10):
        #         plt.plot(x.detach().cpu().numpy()[0,0,j,:])
        #     plt.show()

        x = self.start_conv(x)

        if viz: # x.shape [16, 32, 200, 15]
            ### plot features on different channels representing the same node fmri signal
            for j in range(x.shape[1]):
                plt.plot(x.detach().cpu().numpy()[0,j,0,:])
            plt.show()
            ### plot one channel's features of different fmri signals (should be different)
            for j in range(10):
                plt.plot(x.detach().cpu().numpy()[0,0,j,:])
            plt.show()

        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.gcn_bool and self.addaptadj:
            nodevec = torch.einsum('ncl,lv->ncv', (input[:,0,...],self.nodevec))
            adp = F.softmax(F.relu(torch.matmul(nodevec, nodevec.transpose(1,2))), dim=2)
            # if viz:
            #     ipdb.set_trace()
            #     plt.imshow(adp[0].detach().cpu().numpy())
            #     plt.show()
            if len(supports) > 0:
                new_supports = supports + [adp]
            else:
                new_supports = [adp]
            del adp

        # WaveNet layers
        for i in range(self.blocks * self.layers):
            # print(i, x.shape)
            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + --> *input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + -------------> *skip*

            #(dilation, init_dilation) = self.dilations[i]

            #residual = dilation_func(x, dilation, init_dilation, i)
            residual = x #[batch_size, residual_dim, 80, 16]

            x = checkpoint(self.diconv(self.filter_convs[i], self.gate_convs[i]), residual, self.dummy_tensor)
            # skip = checkpoint(self.skip_part(self.skip_convs[i], skip), x, self.dummy_tensor)

            # x, skip = checkpoint(self.tcn(self.filter_convs[i], self.gate_convs[i],
            #                     self.skip_convs[i], skip), residual, self.dummy_tensor)
           
            # # dilated convolution
            # filter = self.filter_convs[i](residual)
            # filter = torch.tanh(filter)
            # gate = self.gate_convs[i](residual)
            # gate = torch.sigmoid(gate)
            # x = filter * gate

            # del filter
            # del gate

            if i % self.layers == self.layers-1:
                x = F.max_pool2d(x, (1,2))
            # parametrized skip connection
            s = x
            s = self.skip_convs[i](s)
            try:
                if s.size(-1)*2 == skip.size(-1):
                    skip = F.max_pool2d(skip,(1,2))
                else:
                    skip = skip[..., -s.size(-1):]
            except:
                skip = 0
            skip = s + skip

            t_rep = x

            if self.gcn_bool:
                if self.addaptadj:
                    # x: [64, 32, 80, 15]
                    # x = self.gconv[i](x, *new_supports)
                    x = checkpoint(self.gconv[i], x, self.dummy_tensor, *new_supports)
                else:
                    assert supports is not None
                    # x = self.gconv[i](x, *supports)
                    x = checkpoint(self.gconv[i], x, self.dummy_tensor, *supports)
            else:
                # x = self.residual_convs[i](x)
                x = checkpoint(self.residual_convs[i], x, self.dummy_tensor)

            # x = x + residual[:, :, :, -x.size(3):]
            # add t representation
            x = x + residual[:, :, :, -x.size(3):]# + t_rep
            # x = torch.cat([x, residual[:, :, :, -x.size(3):], t_rep], axis=1)
            x = self.bn[i](x) # comment off for overfitting
            # print(x.shape)

        # del residual, x
        # skip: [batch_size, hidden_dim, num_nodes, 1]

        # ### test: adding noise to hidden rep
        # skip = skip + torch.normal(torch.zeros_like(skip), 0.1*skip.std()*torch.ones_like(skip))
        # ###
        x = skip
        x = F.relu(x)
        x = fourier_mapping(x,self.B)

        x = self.end_module(x)

        if viz: # x.shape [16, 512, 200, 1]
            # (results look similar) each node's h-D (h being #hidden dim) feature
            for j in range(10):
                plt.plot(x.detach().cpu().numpy()[0,:,j,0])
            plt.show()
            # plot each channel's value (on all nodes)
            for j in range(10): 
                plt.plot(x.detach().cpu().numpy()[0,j,:,0])
            plt.show()
        
        if self.meta is None:
            # ### F prediction
            x_f = self.end_mlp_f(x).transpose(1,3)

            ### E prediction
            x = self.end_module_add(x) #[batch_size, seq_len, num_nodes, 1]
            ########### USING POOL ########### 
            # x = self.pooling(x, *new_supports)
            ########### USING MLP ########### 
            x = x.transpose(1, 2)
            x1 = self.end_mlp_e(x) #[batch_size, out_nodes, seq_len, 1]
            x1 = x1.transpose(1, 2)
            # x2 = self.end_mlp_e2(x)
            # x2 = x2.transpose(1, 2)
            # x3 = self.end_mlp_e3(x)
            # x3 = x3.transpose(1, 2)
            return x1, x_f#, x2, x3

        else: # scatter (skip: [16, 256, 200, 1])
            x = self.end_module_add(x) #[`batch_size, out_length, num_nodes, 1]
            x = x.transpose(1, 2)
            x = self.end_mlp_e(x)
            sig = x.transpose(2,3).contiguous()

            # ### temporal transConv
            # x = x[...,:self.out_dim]        
            # x = x.transpose(1, 2)
            # sig = self.end_mlp_e(x)
            return sig, self.scattering(sig)#[:,:,:,self.meta[1]] # order0 only


class gwnet_diff_G_Fonly(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports_len=0, batch_size=32,
                gcn_bool=True, addaptadj=True, in_dim=2, seq_len=12, out_dim_f=5,
                residual_channels=32, dilation_channels=32, skip_channels=256,
                end_channels=512, kernel_size=2, blocks=4, layers=2,
                out_nodes=64):

        super(gwnet_diff_G_Fonly, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj
        self.device = device
        self.num_nodes = num_nodes
        self.seq_len = seq_len

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)
        
        if self.gcn_bool and self.addaptadj:
            '''If different samples have different supports, theres no way to 
            init nodevec using supports info...'''
            # if aptinit is None:
            self.nodevec = nn.Parameter(torch.randn(seq_len, 5).to(self.device), 
                                    requires_grad=True).to(self.device)

            # else:
            #     ipdb.set_trace()
            #     m, p, n = torch.svd(aptinit)
            #     _p = torch.diag_embed(p[:, :10] ** 0.5)
            #     initemb1 = torch.bmm(m[:, :, :10], _p)
            #     initemb2 = torch.bmm(_p, torch.transpose(n[:, :, :10], 1, 2))
            #     self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(self.device)
            #     self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(self.device)

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))

        receptive_field = 1
        multi_factor = kernel_size #2
        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1 #4
            for i in range(layers):
                # dilated convolutions
                # TODO: change kernel_size and stride
                self.filter_convs.append(CausalConv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,kernel_size),
                                                   # stride=(1,2),
                                                   dilation=(1,new_dilation)))

                self.gate_convs.append(CausalConv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), 
                                                 # stride=(1,2),
                                                 dilation=(1,new_dilation)))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= multi_factor
                receptive_field += additional_scope
                additional_scope *= multi_factor
                if self.gcn_bool:
                    self.gconv.append(gcn2(dilation_channels, residual_channels, dropout,
                                                              support_len=supports_len))



        # self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
        #                           out_channels=end_channels,
        #                           kernel_size=(1,1),
        #                           bias=True)

        # self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
        #                             out_channels=out_dim_f,
        #                             kernel_size=(1,1),
        #                             bias=True)

        self.end_conv_1 = nn.Conv2d(skip_channels, skip_channels//2, 1, bias=True)
        self.end_conv_2 = nn.Conv2d(skip_channels//2, out_dim_f, 1, bias=True)
        # self.end_conv_3 = nn.Conv2d(out_dim_f, 1, 1, bias=True)

        self.receptive_field = receptive_field

    def forward(self, input, supports=None, aptinit=None, viz=False):
        # inputs: [batch_size, 1, num_nodes, in_len+1], supports: len 2, each (batch_size, num_nodes, num_nodes)
        if self.gcn_bool and self.addaptadj:
            if supports is None:
                supports = []
        ### normal forward
        # print(self.receptive_field)
        # in_len = input.size(3)
        # if in_len<self.receptive_field:
        #     x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        # else:
        #     x = input
        # x = self.start_conv(x)
        x = self.start_conv(input)
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.gcn_bool and self.addaptadj:
            nodevec = torch.einsum('ncl,lv->ncv', (input[:,0,...],self.nodevec))
            adp = F.softmax(F.relu(torch.matmul(nodevec, nodevec.transpose(1,2))), dim=2)
            if viz: 
                plt.imshow(adp[0].detach().cpu().numpy())
                plt.show()
                # plot learned theta
                plt.imshow(self.nodevec.detach().cpu().numpy())
                plt.show()
                ipdb.set_trace()
                # adp.sum(1)
                # _, idx = torch.sort(adp.sum(1)) 
                # top10 = idx.cpu().numpy()[:,::-1][:,:10]

            if len(supports) > 0:
                new_supports = supports + [adp]
            else:
                new_supports = [adp]
            del adp

        # # WaveNet layers
        # if viz:
        #     tmp = x.detach().cpu().numpy()
        #     for j in range(5):
        #         plt.plot(tmp[0,:,j,4], label='4,'+str(j))
        #         plt.plot(tmp[0,:,j,10], label='10,'+str(j))
        #     plt.legend()
        #     plt.title('after start_conv, before embedding layers')
        #     plt.show()
        for i in range(self.blocks * self.layers):       
            # print(i, x.shape)
            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + --> *input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + -------------> *skip*

            #(dilation, init_dilation) = self.dilations[i]

            #residual = dilation_func(x, dilation, init_dilation, i)
            residual = x #[batch_size, residual_dim, 80, 16]

            # x = checkpoint(self.diconv(self.filter_convs[i], self.gate_convs[i]), residual, self.dummy_tensor)
            # skip = checkpoint(self.skip_part(self.skip_convs[i], skip), x, self.dummy_tensor)

            # x, skip = checkpoint(self.tcn(self.filter_convs[i], self.gate_convs[i],
            #                     self.skip_convs[i], skip), residual, self.dummy_tensor)
           
            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter[..., :-self.filter_convs[i].padding[1]])
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate[..., :-self.gate_convs[i].padding[1]])
            x = filter * gate
            if i % self.layers == self.layers-1:
                x = F.max_pool2d(x, (1,2))
            # parametrized skip connection
            s = x
            s = self.skip_convs[i](s)
            try:
                if s.size(-1)*2 == skip.size(-1):
                    skip = F.max_pool2d(skip,(1,2))
                else:
                    skip = skip[..., -s.size(-1):]
            except:
                skip = 0
            skip = s + skip

            t_rep = x

            if self.gcn_bool:
                if self.addaptadj:
                    # x: [64, 32, 80, 15]
                    x = self.gconv[i](x, self.dummy_tensor, *new_supports)
                    # x = checkpoint(self.gconv[i], x, self.dummy_tensor, *new_supports)
                else:
                    assert supports is not None
                    x = self.gconv[i](x, self.dummy_tensor, *supports)
                    # x = checkpoint(self.gconv[i], x, self.dummy_tensor, *supports)
            else:
                x = self.residual_convs[i](x)
                # x = checkpoint(self.residual_convs[i], x, self.dummy_tensor)

            x = x + residual[..., -x.size(-1):]
            # add t representation
            # x = torch.cat([x, residual[:, :, :, -x.size(3):], t_rep], axis=1)
            x = self.bn[i](x) #torch.Size([8, 32, 200, 15])

        # if viz:
        #     print(skip.shape)
        #     for j in range(10):
        #         plt.plot(tmp[0,:,j,0], label=str(j))
        #     plt.legend()
        #     plt.title('hidden rep')                
        #     plt.show()
        # ipdb.set_trace()
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        # x = self.end_conv_3(F.relu(x))

        return x, new_supports[-1]#.sum(1)

class gwnet_diff_G2(nn.Module): # for model testing, f in, same f out
    def __init__(self, device, num_nodes, dropout=0.3, supports_len=0,
                gcn_bool=True, addaptadj=True,
                in_dim=2, out_dim=12, out_dim_f=5,
                residual_channels=32, dilation_channels=32, skip_channels=256,
                end_channels=512, kernel_size=2, blocks=4, layers=2,
                out_nodes=64, meta=None, scatter=False):

        super(gwnet_diff_G2, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj
        self.device = device
        self.num_nodes = num_nodes
        self.scatter = scatter
        self.out_dim = out_dim

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))

        receptive_field = 1
        multi_factor = kernel_size #2
        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1 #4
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(CausalConv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,kernel_size),
                                                   # stride=(1,2),
                                                   dilation=(1,new_dilation)))

                self.gate_convs.append(CausalConv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), 
                                                 # stride=(1,2),
                                                 dilation=(1,new_dilation)))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels)) # comment off for overfitting
                # self.bn.append(nn.Sequential(
                #     nn.Conv2d(in_channels=residual_channels*3, out_channels=residual_channels,
                #               kernel_size=(1,1)),                    
                #     # nn.BatchNorm2d(residual_channels)
                #     ))
                new_dilation *= multi_factor
                receptive_field += additional_scope
                additional_scope *= multi_factor
                if self.gcn_bool:
                    self.gconv.append(gcn2(dilation_channels, residual_channels, dropout,
                                                              support_len=supports_len))

        self.end_conv_1 = nn.Conv2d(skip_channels, skip_channels//2, 1, bias=True)
        self.end_conv_2 = nn.Conv2d(skip_channels//2, out_dim_f, 1, bias=True)

        self.receptive_field = receptive_field
        self.meta = meta

    def forward(self, input, supports=None, supports2=None, aptinit=None, viz=False):
        # inputs: [batch_size, 1, num_nodes, in_len+1], supports: len 2, each (batch_size, num_nodes, num_nodes)
        ### deal with supports
        batch_size = len(input)
        if self.gcn_bool and self.addaptadj:
            if supports is None:
                    supports = []
            if aptinit is None:
                nodevec1 = nn.Parameter(torch.randn(batch_size, 
                                        self.num_nodes, 10).to(self.device), 
                                        requires_grad=True).to(self.device)
                nodevec2 = nn.Parameter(torch.randn(batch_size, 
                                        10, self.num_nodes).to(self.device), 
                                        requires_grad=True).to(self.device)

            else:
                m, p, n = torch.svd(aptinit)
                _p = torch.diag_embed(p[:, :10] ** 0.5)
                initemb1 = torch.bmm(m[:, :, :10], _p)
                initemb2 = torch.bmm(_p, torch.transpose(n[:, :, :10], 1, 2))
                nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(self.device)
                nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(self.device)

        ### normal forward
        # # print(self.receptive_field)
        # in_len = input.size(3)
        # if in_len<self.receptive_field:
        #     x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        # else:
        #     x = input

        # x = self.start_conv(x)
        # x = nn.functional.pad(input,(1,0,0,0))
        x = self.start_conv(input)

        if viz: # x.shape [16, 32, 200, 15]
            ### plot features on different channels representing the same node fmri signal
            for j in range(x.shape[1]):
                plt.plot(x.detach().cpu().numpy()[0,j,0,:])
            plt.show()
            ### plot one channel's features of different fmri signals (should be different)
            for j in range(10):
                plt.plot(x.detach().cpu().numpy()[0,0,j,:])
            plt.show()

        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.gcn_bool and self.addaptadj:
            adp = F.softmax(F.relu(torch.matmul(nodevec1, nodevec2)), dim=2)
            if len(supports) > 0:
                new_supports = supports + [adp]
            else:
                new_supports = [adp]
            del adp
            del nodevec1, nodevec2

        # WaveNet layers
        for i in range(self.blocks * self.layers):
            # print(i, x.shape)
            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + --> *input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + -------------> *skip*

            #(dilation, init_dilation) = self.dilations[i]

            #residual = dilation_func(x, dilation, init_dilation, i)
            residual = x #[batch_size, residual_dim, 80, 16]

            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter[..., :-self.filter_convs[i].padding[1]])
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate[..., :-self.gate_convs[i].padding[1]])
            x = filter * gate
            if i % self.layers == self.layers-1:
                x = F.max_pool2d(x, (1,2))

            # parametrized skip connection
            s = x
            s = self.skip_convs[i](s)
            try:
                if s.size(-1)*2 == skip.size(-1):
                    skip = F.max_pool2d(skip,(1,2))
                else:
                    skip = skip[..., -s.size(-1):]
            except:
                skip = 0
            skip = s + skip

            # t_rep = x

            if self.gcn_bool:
                if self.addaptadj:
                    # x: [64, 32, 80, 15]
                    x = self.gconv[i](x, None, *new_supports)
                else:
                    assert supports is not None
                    x = self.gconv[i](x, None, *supports)
            else:
                x = self.residual_convs[i](x)

            # x = x + residual[:, :, :, -x.size(3):]
            # add t representation
            x = x + residual[..., -x.size(-1):]# + t_rep
            # x = torch.cat([x, residual[:, :, :, -x.size(3):], t_rep], axis=1)
            x = self.bn[i](x) # comment off for overfitting
            # print(x.shape)

        # del residual, x
        # skip: [batch_size, hidden_dim, num_nodes, 1]

        # ### test: adding noise to hidden rep
        # skip = skip + torch.normal(torch.zeros_like(skip), 0.1*skip.std()*torch.ones_like(skip))
        # ###
   
        x = F.relu(skip)
        # for i in range(3):#range(self.blocks * self.layers):       
        #     x = self.end_module_up[i](x)
        #     x = F.relu(self.end_module_conv[i](x))
        #     x = self.end_module_gcn[i](x, None, *supports2)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x).transpose(1,2)
        return x#.transpose(1,2)

class gwnet_vgae(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports_len=0, batch_size=32,
                gcn_bool=True, addaptadj=True, in_dim=2, seq_len=12, out_dim_f=5,
                residual_channels=32, dilation_channels=32, skip_channels=256,
                end_channels=512, kernel_size=3, blocks=4, layers=2,
                out_nodes=61, hidden_dim=16):

        super(gwnet_vgae, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj
        self.device = device
        self.num_nodes = num_nodes
        self.seq_len = seq_len

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.gconv_pool = nn.ModuleList()
        self.dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)
        
        if self.gcn_bool and self.addaptadj:
            '''If different samples have different supports, theres no way to 
            init nodevec using supports info...'''
            # if aptinit is None:
            self.nodevec = nn.Parameter(torch.randn(seq_len, 5).to(self.device), 
                                    requires_grad=True).to(self.device)

            # else:
            #     ipdb.set_trace()
            #     m, p, n = torch.svd(aptinit)
            #     _p = torch.diag_embed(p[:, :10] ** 0.5)
            #     initemb1 = torch.bmm(m[:, :, :10], _p)
            #     initemb2 = torch.bmm(_p, torch.transpose(n[:, :, :10], 1, 2))
            #     self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(self.device)
            #     self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(self.device)

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))

        receptive_field = 1
        multi_factor = kernel_size #2
        clusters = [150,110,85,65,50,30,15,7]

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1 #4
            for i in range(layers):
                # dilated convolutions
                # TODO: change kernel_size and stride
                self.filter_convs.append(CausalConv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,kernel_size),
                                                   # stride=(1,2),
                                                   dilation=(1,new_dilation)))

                self.gate_convs.append(CausalConv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), 
                                                 # stride=(1,2),
                                                 dilation=(1,new_dilation)))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= multi_factor
                receptive_field += additional_scope
                additional_scope *= multi_factor
                if self.gcn_bool:
                    self.gconv.append(gcn2(dilation_channels, residual_channels, dropout, support_len=supports_len))
                    if i == self.layers-1:
                        self.gconv_pool.append(gcn2(dilation_channels, clusters[b], dropout, support_len=supports_len))

        # self.gcn_pool0 = gcn2(dilation_channels, 100, dropout, support_len=supports_len)
        # self.gcn_pool1 = gcn2(dilation_channels, 50, dropout, support_len=supports_len) 
        # self.gcn_pool2 = gcn2(dilation_channels, 20, dropout, support_len=supports_len)
        # self.gcn_pool3 = gcn2(dilation_channels, 17, dropout, support_len=supports_len)
        # self.gcn_embed = gcn2(dilation_channels, residual_channels, dropout, support_len=supports_len)

        self.gcn_mean = gcn2(skip_channels, hidden_dim, dropout=0, support_len=supports_len)
        self.gcn_logstddev = gcn2(skip_channels, hidden_dim, dropout=0, support_len=supports_len)

        self.end_conv_1 = nn.Conv2d(skip_channels, skip_channels//2, 1, bias=True)
        # self.end_conv_2 = nn.Conv2d(skip_channels//2, skip_channels//2, 1, bias=True)
        self.end_conv_3 = nn.Conv2d(skip_channels//2, out_dim_f, 1, bias=True)

        # self.decode_gcn1 = gcn2(skip_channels, hidden_dim, dropout, support_len=1)
        # self.decode_gcn2 = gcn2(hidden_dim, 1, dropout, support_len=1)

        # self.decode_gcn1 = nn.Conv2d(skip_channels, hidden_dim, 1, bias = True)
        # self.decode_gcn2 = nn.Conv2d(hidden_dim, 1, 1, bias = True)

        self.receptive_field = receptive_field

    def forward(self, input, supports=None, aptinit=None, viz=False):
        # inputs: [batch_size, 1, num_nodes, in_len+1], supports: len 2, each (batch_size, num_nodes, num_nodes)
        if self.gcn_bool and self.addaptadj:
            if supports is None:
                supports = []
        ### normal forward
        # print(self.receptive_field)
        # in_len = input.size(3)
        # if in_len<self.receptive_field:
        #     x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        # else:
        #     x = input
        # x = self.start_conv(x)
        x = self.start_conv(input)
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.gcn_bool and self.addaptadj:
            nodevec = torch.einsum('ncl,lv->ncv', (input[:,0,...],self.nodevec))
            adp = F.softmax(F.relu(torch.matmul(nodevec, nodevec.transpose(1,2))), dim=2)
            if viz: 
                plt.imshow(adp[0].detach().cpu().numpy())
                plt.show()
                # plot learned theta
                plt.imshow(self.nodevec.detach().cpu().numpy())
                plt.show()
                ipdb.set_trace()
                # adp.sum(1)
                # _, idx = torch.sort(adp.sum(1)) 
                # top10 = idx.cpu().numpy()[:,::-1][:,:10]

            if len(supports) > 0:
                new_supports = supports + [adp]
            else:
                new_supports = [adp]
            del adp

        u = 0
        # WaveNet layers
        for i in range(self.blocks * self.layers):       
            # print(i, x.shape)
            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + --> *input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + -------------> *skip*

            #(dilation, init_dilation) = self.dilations[i]

            #residual = dilation_func(x, dilation, init_dilation, i)
            residual = x #[batch_size, residual_dim, 80, 16]
           
            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter[..., :-self.filter_convs[i].padding[1]])
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate[..., :-self.gate_convs[i].padding[1]])
            x = filter * gate
            if i % self.layers == self.layers-1:
                x = F.max_pool2d(x, (1,2))
            # parametrized skip connection
            s = x
            s = self.skip_convs[i](s)
            try:
                if s.size(-1)*2 == skip.size(-1):
                    skip = F.max_pool2d(skip,(1,2))
                else:
                    skip = skip[..., -s.size(-1):]
            except:
                skip = 0
            skip = s + skip

            t_rep = x

            # S = self.gcn_pool(x, self.dummy_tensor, *new_supports)
            # x = self.gcn_embed(x, self.dummy_tensor, *new_supports)
            # x, adj, l1, e1 = torch_geometric.nn.dense.diff_pool(x, *new_supports, S)

            if self.gcn_bool:
                if self.addaptadj:
                    x = self.gconv[i](x, self.dummy_tensor, *new_supports)
                    # x = checkpoint(self.gconv[i], x, self.dummy_tensor, *new_supports)
                else:
                    assert supports is not None
                    x = self.gconv[i](x, self.dummy_tensor, *supports)
                    # x = checkpoint(self.gconv[i], x, self.dummy_tensor, *supports)
                if i % self.layers == self.layers-1:
                    # ipdb.set_trace()
                    S = self.gconv_pool[i // self.layers](x, self.dummy_tensor, *new_supports)
                    x = dense_diff_pool(x, S, *new_supports)
                    # S = eval("self.gcn_pool" + str(u) + "(x, self.dummy_tensor, *new_supports)")
                    # x = self.gconv[i](x, self.dummy_tensor, *new_supports)
                    # # S = self.gcn_pool(x, self.dummy_tensor, *new_supports)
                    # # x = self.gcn_embed(x, self.dummy_tensor, *new_supports)
                    # # ipdb.set_trace()
                    # x = dense_diff_pool(x, S, *new_supports)
                    # u += 1                    
            else:
                x = self.residual_convs[i](x)
                # x = checkpoint(self.residual_convs[i], x, self.dummy_tensor)

            x = x + residual[..., -x.size(-1):]
            # add t representation
            # x = torch.cat([x, residual[:, :, :, -x.size(3):], t_rep], axis=1)
            x = self.bn[i](x) #torch.Size([8, 32, 200, 15])

        x = F.relu(skip) # x: hidden embedding
        # ipdb.set_trace()

        self.mean = self.gcn_mean(x, self.dummy_tensor, *new_supports).squeeze()
        self.logstd = self.gcn_logstddev(x, self.dummy_tensor, *new_supports).squeeze()
        gaussian_noise = torch.randn_like(self.mean)
        sampled_z = gaussian_noise*torch.exp(self.logstd) + self.mean
        
        decoder_A = torch.sigmoid(torch.matmul(sampled_z.transpose(1,2), sampled_z))

        # ipdb.set_trace()
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_3(x)
        # x = self.end_conv_3(F.relu(x))

        # x = F.relu(self.decode_gcn1(x, self.dummy_tensor, decoder_A))
        # x = self.decode_gcn2(x, self.dummy_tensor, decoder_A)

        return x, new_supports[-1]#.sum(1)