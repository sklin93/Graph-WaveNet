import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch.utils.checkpoint import checkpoint
from kymatio.torch import Scattering1D
import sys
import ipdb

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
        A = torch.transpose(A, -1, -2)
        x = torch.einsum('ncvl,nvw->ncwl',(x,A))
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
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2
        # len(out)=order*support_len+1 (7); each:[64, 32, 80, 15]
        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

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

class gwnet(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, 
                gcn_bool=True, addaptadj=True, aptinit=None, 
                in_dim=2,out_dim=12,residual_channels=32,dilation_channels=32,
                skip_channels=256,end_channels=512,kernel_size=2,blocks=4,layers=2):

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
            new_dilation = 4#1
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
                new_dilation *=2
                receptive_field += additional_scope
                additional_scope *= 2
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



    def forward(self, input): #[64, 2, 207, 13]
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
        # new supports: len 3 list, each n*n dim

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

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
                    # x dim: [64, 32, 207, 12]
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x,self.supports)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]

            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x) #[64, 12, 207, 1]
        return x

class gwnet_diff_G(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports_len=0,
                gcn_bool=True, addaptadj=True,
                in_dim=2, out_dim=12, out_dim_f=5,
                residual_channels=32, dilation_channels=32, skip_channels=256,
                end_channels=512, kernel_size=2, blocks=4, layers=2,
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
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,kernel_size),
                                                   # stride=(1,2),
                                                   dilation=new_dilation))

                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), 
                                                 # stride=(1,2),
                                                 dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                # self.bn.append(nn.BatchNorm2d(residual_channels))
                self.bn.append(nn.Sequential(
                    nn.Conv2d(in_channels=residual_channels*3, out_channels=residual_channels,
                              kernel_size=(1,1)),                    
                    # nn.BatchNorm2d(residual_channels)
                    ))
                new_dilation *= multi_factor
                receptive_field += additional_scope
                additional_scope *= multi_factor
                if self.gcn_bool:
                    self.gconv.append(gcn2(dilation_channels, residual_channels, dropout,
                                                              support_len=supports_len))

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
                nn.Conv2d(in_channels=skip_channels, out_channels=end_channels,
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

        if (meta is None) or scatter:
            self.end_module_add = nn.Sequential(
                # nn.Tanh(), 
                nn.LeakyReLU(),
                # nn.ReLU(),
                nn.Conv2d(in_channels=end_channels, out_channels=out_dim,
                          kernel_size=(1,1), bias=True)
                )

            self.end_mlp_e = nn.Sequential(
                # nn.Tanh(), 
                nn.LeakyReLU(), 
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
            if meta is None:
                self.end_mlp_f = nn.Sequential(
                    # nn.Tanh(),
                    nn.LeakyReLU(), 
                    # nn.ReLU(),
                    nn.Conv2d(in_channels=end_channels, out_channels=out_dim_f,
                              kernel_size=(1,1), bias=True)
                    )
                # self.pooling = pool(out_dim, out_nodes, dropout, supports_len)
            if scatter:
                J = 3
                Q = 9
                self.scattering = Scattering1D(J, out_dim, Q)   
         
        else: # directly predicting scattering coefficient using conv layers
            nrow = len(meta[0]) + len(meta[1]) + len(meta[2])
            assert out_dim%nrow == 0
            ncol = int(out_dim / nrow)
            assert meta is not None
            assert ncol > len(meta[1]) and ncol > len(meta[2])
            # pad rectangle shaped coeff (1st & 2nd orders)
            self.ncol = ncol
            self.coeff_pad1 = ncol - len(meta[1])
            self.coeff_pad2 = ncol - len(meta[2])
            # increase the num of feature dimension to ncol**2
            self.coeff_conv = nn.Sequential(
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=end_channels*2, 
                          out_channels=ncol**2,
                          kernel_size=(1,1), bias=True)
                )
            self.end_module_add1 = nn.Sequential(
                nn.LeakyReLU(),
                # nn.Conv2d(in_channels=end_channels*2, 
                #           out_channels=len(meta[0]) * ncol,
                #           kernel_size=(1,1), bias=True)
                # nn.Conv2d(in_channels=ncol**2, 
                #           out_channels=len(meta[0]) * ncol,
                #           kernel_size=(1,1), bias=True),
                nn.Conv2d(in_channels=num_nodes,
                          out_channels=num_nodes,
                          kernel_size=(1,13), bias=True),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=num_nodes,
                          out_channels=num_nodes,
                          kernel_size=(1,13), bias=True),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=num_nodes,
                          out_channels=num_nodes,
                          kernel_size=(1,11), bias=True),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=num_nodes,
                          out_channels=num_nodes,
                          kernel_size=(1,11), bias=True),
                )
            self.end_module_add2 = nn.Sequential(
                nn.LeakyReLU(),
                # nn.Conv2d(in_channels=end_channels*2, 
                #           out_channels=len(meta[1]) * ncol,
                #           kernel_size=(1,1), bias=True)
                nn.Conv2d(in_channels=num_nodes,
                          out_channels=num_nodes,
                          kernel_size=(1,11), bias=True),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=num_nodes,
                          out_channels=num_nodes,
                          kernel_size=(1,11), bias=True),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=num_nodes,
                          out_channels=num_nodes,
                          kernel_size=(1,7), bias=True),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=num_nodes,
                          out_channels=num_nodes,
                          kernel_size=(1,7), bias=True),
                )
            self.end_module_add3 = nn.Sequential(
                nn.LeakyReLU(),
                # nn.Conv2d(in_channels=end_channels*2,
                #           out_channels=len(meta[2]) * ncol,
                #           kernel_size=(1,1), bias=True)
                nn.Conv2d(in_channels=num_nodes,
                          out_channels=num_nodes,
                          kernel_size=(1,10), bias=True),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=num_nodes,
                          out_channels=num_nodes,
                          kernel_size=(1,9), bias=True),          
                )                            
            self.end_mlp_e = nn.Sequential(
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=num_nodes, out_channels=out_nodes,
                          kernel_size=(1,1), bias=True)
                )
            self.end_mlp_e2 = nn.Sequential(
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=num_nodes, out_channels=out_nodes,
                          kernel_size=(1,1), bias=True)
                )
            self.end_mlp_e3 = nn.Sequential(
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=num_nodes, out_channels=out_nodes,
                          kernel_size=(1,1), bias=True)
                )

        self.receptive_field = receptive_field
        self.meta = meta


    def diconv(self, filter_conv, gate_conv):
        def custom_forward(residual, dummy_tensor):
            # dilated convolution
            filter = filter_conv(residual)
            filter = torch.tanh(filter)
            gate = gate_conv(residual)
            gate = torch.sigmoid(gate)
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

    def forward(self, input, supports=None, aptinit=None):
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
        # print(self.receptive_field)
        in_len = input.size(3)
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input
        x = self.start_conv(x)
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

            # parametrized skip connection
            s = self.skip_convs[i](x)
            try:
                skip = skip[:, :, :,  -s.size(3):]
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
            # x = x + residual[:, :, :, -x.size(3):] + t_rep
            x = torch.cat([x, residual[:, :, :, -x.size(3):], t_rep], axis=1)
            x = self.bn[i](x)
            # print(x.shape)

        # del residual, x
        # skip: [batch_size, hidden_dim, num_nodes, 1]

        # ### test: adding noise to hidden rep
        # skip = skip + torch.normal(torch.zeros_like(skip), 0.1*skip.std()*torch.ones_like(skip))
        # ###
        x = self.end_module(skip)
        if self.meta is None:
            
            ### F prediction
            x_f = self.end_mlp_f(x).transpose(1,3)
            ### E prediction
            x = self.end_module_add(x) #[batch_size, seq_len, num_nodes, 1]
            ########### USING POOL ########### 
            # x = self.pooling(x, *new_supports)
            ########### USING MLP ########### 
            x = x.transpose(1, 2)
            x1 = self.end_mlp_e(x) #[batch_size, out_nodes, seq_len, 1]
            x1 = x1.transpose(2, 3)
            # x2 = self.end_mlp_e2(x)
            # x2 = x2.transpose(1, 2)
            # x3 = self.end_mlp_e3(x)
            # x3 = x3.transpose(1, 2)
            return x_f, x1#, x2, x3

        else:
            # skip: [16, 256, 200, 1]
            if self.scatter:
                # x = self.end_module_add(x) #[`batch_size, out_length, num_nodes, 1]
                x = x.transpose(1, 2)
                x = self.end_mlp_e(x)
                sig = x.transpose(2,3).contiguous()

                # ### temporal transConv
                # x = x[...,:self.out_dim]          
                # x = x.transpose(1, 2)
                # sig = self.end_mlp_e(x)

                return sig, self.scattering(sig)#[:,:,:,self.meta[1]] # order0 only

            else:
                x = self.coeff_conv(x)

                # x1 = self.end_module_add1(x) # [16, 45, 200, 1]
                # x1 = self.end_mlp_e(x1.transpose(1,2))

                # use conv2d on 2d representations
                x = x.transpose(1,2).reshape(-1, self.num_nodes, self.ncol, self.ncol)
                x1 = self.end_module_add1(x)
                x2 = self.end_module_add2(x)
                x3 = self.end_module_add3(x)
                x1 = self.end_mlp_e(x1)
                x2 = self.end_mlp_e2(x2)
                x3 = self.end_mlp_e3(x3)
                return x1, x2, x3
                # x2 = self.end_mlp_e2(x2.transpose(1,2))
                # x3 = self.end_mlp_e3(x3.transpose(1,2))
                # return x1.transpose(1, 2), x2.transpose(1, 2), x3.transpose(1, 2)


class gwnet_diff_G_Fonly(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports_len=0,
                gcn_bool=True, addaptadj=True,
                in_dim=2, out_dim=12, out_dim_f=5,
                residual_channels=32, dilation_channels=32, skip_channels=256,
                end_channels=512, kernel_size=2, blocks=4, layers=2,
                out_nodes=64, meta=None, scatter=False):

        super(gwnet_diff_G_Fonly, self).__init__()
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
                # TODO: change kernel_size and stride
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,kernel_size),
                                                   # stride=(1,2),
                                                   dilation=new_dilation))

                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), 
                                                 # stride=(1,2),
                                                 dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                # self.bn.append(nn.BatchNorm2d(residual_channels))
                self.bn.append(nn.Sequential(
                    nn.Conv2d(in_channels=residual_channels*3, out_channels=residual_channels,
                              kernel_size=(1,1)),                    
                    # nn.BatchNorm2d(residual_channels)
                    ))
                new_dilation *= multi_factor
                receptive_field += additional_scope
                additional_scope *= multi_factor
                if self.gcn_bool:
                    self.gconv.append(gcn2(dilation_channels, residual_channels, dropout,
                                                              support_len=supports_len))

        self.end_module = nn.Sequential(
                # nn.Tanh(), 
                nn.LeakyReLU(),
                # nn.ReLU(),
                nn.Conv2d(in_channels=skip_channels, out_channels=end_channels,
                          kernel_size=(1,1), bias=True)
                )     

        self.end_mlp_f = nn.Sequential(
            # nn.Tanh(),
            nn.LeakyReLU(), 
            # nn.ReLU(),
            nn.Conv2d(in_channels=end_channels, out_channels=out_dim_f,
                      kernel_size=(1,1), bias=True)
            )

        if scatter:
            J = 2
            Q = 9
            self.scattering = Scattering1D(J, out_dim_f, Q)

        self.receptive_field = receptive_field
        self.meta = meta


    def diconv(self, filter_conv, gate_conv):
        def custom_forward(residual, dummy_tensor):
            # dilated convolution
            filter = filter_conv(residual)
            filter = torch.tanh(filter)
            gate = gate_conv(residual)
            gate = torch.sigmoid(gate)
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

    def forward(self, input, supports=None, aptinit=None):
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
        # print(self.receptive_field)
        in_len = input.size(3)
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input
        x = self.start_conv(x)
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

            # parametrized skip connection
            s = self.skip_convs[i](x)
            try:
                skip = skip[:, :, :,  -s.size(3):]
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
            # x = x + residual[:, :, :, -x.size(3):] + t_rep
            x = torch.cat([x, residual[:, :, :, -x.size(3):], t_rep], axis=1)
            x = self.bn[i](x)
            # print(x.shape)

        x = self.end_module(skip)
        x_f = self.end_mlp_f(x).transpose(1,3)
        # print(x_f.shape)
        # ipdb.set_trace()
        if self.meta is None:
            return x_f

        else:
            return x_f, self.scattering(x_f.contiguous())#[:,:,:,self.meta[1]] # order0 only