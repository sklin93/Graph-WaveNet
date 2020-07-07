import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.checkpoint import checkpoint
import sys
import ipdb

class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class nconv2(nn.Module):
    def __init__(self):
        super(nconv2,self).__init__()

    def forward(self,x, A):
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

    def forward(self,x,*support):
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
                in_dim=2,out_dim=12,residual_channels=32,dilation_channels=32,
                skip_channels=256,end_channels=512,kernel_size=2,blocks=4,layers=2):

        super(gwnet_diff_G, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj
        self.device = device
        self.num_nodes = num_nodes

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

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
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= multi_factor
                receptive_field += additional_scope
                additional_scope *= multi_factor
                if self.gcn_bool:
                    self.gconv.append(gcn2(dilation_channels, residual_channels, dropout,
                                                              support_len=supports_len))

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)

        self.receptive_field = receptive_field


    def diconv(self, filter_conv, gate_conv):
        def custom_forward(residual):
            # dilated convolution
            filter = filter_conv(residual)
            filter = torch.tanh(filter)
            gate = gate_conv(residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            return x
        return custom_forward

    def skip_part(self, skip_conv, skip):
        def custom_forward(s):
            # parametrized skip connection
            s = skip_conv(s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            return skip
        return custom_forward

    def endconv(self, end_conv_1, end_conv_2):
        def custom_forward(x):
            x = F.relu(x)
            x = F.relu(end_conv_1(x))
            x = end_conv_2(x)
            return x
        return custom_forward

    def forward(self, input, supports, aptinit):
        # inputs: [64, 2, 80, 16], supports: len 2, each (64, 80, 80)
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
                ipdb.set_trace() # fix this (one more batch size dimension)
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(self.device)
                nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(self.device)

        ### normal forward
        # print(self.receptive_field)
        in_len = input.size(3)
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input
        x = checkpoint(self.start_conv, x)
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.gcn_bool and self.addaptadj and supports is not None:
            adp = F.softmax(F.relu(torch.matmul(nodevec1, nodevec2)), dim=2)
            new_supports = supports + [adp]
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

            x = checkpoint(self.diconv(self.filter_convs[i], self.gate_convs[i]), residual)
            skip = checkpoint(self.skip_part(self.skip_convs[i], skip), x)

            # # dilated convolution
            # filter = self.filter_convs[i](residual)
            # filter = torch.tanh(filter)
            # gate = self.gate_convs[i](residual)
            # gate = torch.sigmoid(gate)
            # x = filter * gate

            # # del filter
            # # del gate

            # # parametrized skip connection
            # s = x
            # s = self.skip_convs[i](s)
            # try:
            #     skip = skip[:, :, :,  -s.size(3):]
            # except:
            #     skip = 0
            # skip = s + skip

            if self.gcn_bool and supports is not None:
                if self.addaptadj:
                    # x: [64, 32, 80, 15]
                    # x = self.gconv[i](x, new_supports)
                    x = checkpoint(self.gconv[i], x, *new_supports)
                else:
                    # x = self.gconv[i](x, supports)
                    x = checkpoint(self.gconv[i], x, supports)
            else:
                # x = self.residual_convs[i](x)
                x = checkpoint(self.residual_convs[i], x)

            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)
            # print(x.shape)
        # ipdb.set_trace()
        # print(skip.shape)
        del residual, x
        return checkpoint(self.endconv(self.end_conv_1, self.end_conv_2), skip)
        # x = F.relu(skip)
        # x = F.relu(self.end_conv_1(x))
        # x = self.end_conv_2(x) #[64, 12, 207, 1]
        # return x