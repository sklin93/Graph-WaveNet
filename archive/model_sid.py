import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.checkpoint import checkpoint
import sys
import ipdb
import numpy as np
# import GPUtil
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

torch.manual_seed(999)

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
        A = torch.transpose(A, -1, -2)
        x = torch.einsum('ncvl,nvw->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)
        #torch.nn.init.kaiming_uniform_(self.mlp.weight)
    def forward(self,x):
        #m = nn.LeakyReLU()
        #return m(self.mlp(x))
        return self.mlp(x)

class speciallinear(nn.Module):
    def __init__(self,c_in,c_out,kernel_size,stride):
        super(speciallinear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(kernel_size,1), stride =(stride,1), bias=True)

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

    def forward(self,x,*support):
        out = [x]
        print(support)
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

class gcn3(nn.Module):
    def __init__(self,c_in,c_out,dropout,kernel_size, stride,support_len=3,order=2):
        super(gcn3,self).__init__()
        self.nconv = nconv2()
        c_in = (order*support_len+1)*c_in #(2*3+1)*32=224
        self.mlp = speciallinear(c_in,c_out,kernel_size,stride)
        self.dropout = dropout
        self.order = order

    def forward(self,x,*support):
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
        #print(h.shape)
        h = self.mlp(h)
        #print(h.shape)
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
    
    def init_weights(self, m):
        if type(m) == nn.ConvTranspose2d or type(m) == nn.Conv2d:
            torch.nn.init.kaiming_uniform_(m.weight)
            
    def __init__(self, device, num_nodes, dropout=0, supports_len=0,
                gcn_bool=True, addaptadj=True,
                in_dim=1, out_dim=12, out_dim_f=5,
                residual_channels=64, dilation_channels=64, skip_channels=256,
                end_channels=512, kernel_size=3, blocks=3, layers=2,
                out_nodes=64):

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
        self.pool_layers = nn.ModuleList()

        self.dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)

        self.leaky_relu = nn.LeakyReLU()


        self.test_start_conv = nn.Conv2d(in_channels = in_dim, out_channels = residual_channels,
                                            kernel_size = (1,1), stride = (1,1))

        self.start_conv = nn.Sequential(
                #nn.Conv2d(in_channels = in_dim,
                #                    out_channels = int(residual_channels/4),
                #                    kernel_size =(1,1)),
                
                nn.Conv2d(in_channels = in_dim,
                                    out_channels = int(residual_channels/2),
                                    kernel_size =(1,5), stride = (1,5)),
                nn.LeakyReLU(),
                #nn.Upsample(scale_factor = (float(200/64),1), mode = 'nearest')
                
                #nn.BatchNorm2d(int(residual_channels/4)),
                nn.Conv2d(in_channels = int(residual_channels/2),
                                    out_channels = int(residual_channels),
                                    kernel_size =(1,5), stride = (1,3)),
                nn.LeakyReLU(),
                #nn.BatchNorm2d(int(residual_channels/2)),
                nn.Conv2d(in_channels = int(residual_channels),
                                    out_channels = 2*residual_channels,
                                    kernel_size =(1,5), stride = (1,3)),
                nn.LeakyReLU(),
                #nn.BatchNorm2d(int(residual_channels)),
                nn.ConvTranspose2d(in_channels = int(2*residual_channels),
                                    out_channels = int(residual_channels), stride = (1,3),
                                    padding = (0,0), dilation = (1,1), kernel_size = (3,1)),
                nn.LeakyReLU(),
                #nn.BatchNorm2d(int(residual_channels)),
                nn.ConvTranspose2d(in_channels = int(residual_channels),
                                    out_channels = int(residual_channels), stride = (3,2),
                                    padding = (0,0), dilation = (1,1), kernel_size = (5,1)),
                #nn.ReLU(),
                nn.LeakyReLU()
        )
        #self.start_conv.apply(self.init_weights)
        receptive_field = 1
        multi_factor = 2 #2
        #kernel_size = 2
        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 2 #4
            #residual_channels *= 2
            #dilation_channels *= 2
            for i in range(layers):
                #if(i % 2 == 1):
                    #residual_channels *= 2
                    #dilation_channels *= 2
                # dilated convolutions
                # TODO: change kernel_size and stride
                #print("Kernel_size = " + str(kernel_size))
                filter_conv = nn.Conv2d(in_channels= residual_channels,
                                                   out_channels= dilation_channels,
                                                   kernel_size=(1,kernel_size),
                                                   stride=(1,2),
                                                   dilation=(1,new_dilation))
                #torch.nn.init.xavier_uniform_(filter_conv.weight)
                self.filter_convs.append(filter_conv)
                
                gate_conv = nn.Conv1d(in_channels= residual_channels,
                                                 out_channels= dilation_channels,
                                                 kernel_size=(1, kernel_size), 
                                                 stride=(1,2),
                                                 dilation=(1,new_dilation))
                #torch.nn.init.xavier_uniform_(gate_conv.weight)
                self.gate_convs.append(gate_conv)
                                                 
                # 1x1 convolution for residual connection
                resid_conv = nn.Conv2d(in_channels= residual_channels,
                                                     out_channels= dilation_channels,
                                                     kernel_size=(1, 1))
                #torch.nn.init.kaiming_uniform_(resid_conv.weight)
                self.residual_convs.append(resid_conv)
                # 1x1 convolution for skip connection
                skip_conv = nn.Conv1d(in_channels= dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1))
                #torch.nn.init.xavier_uniform_(skip_conv.weight)
                self.skip_convs.append(skip_conv)
                self.pool_layers.append(nn.MaxPool2d(kernel_size=(1, kernel_size),
                                                 stride=(1,2),
                                                 dilation=(1,new_dilation)))
                #residual_channels = dilation_channels
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= multi_factor
                receptive_field += additional_scope
                additional_scope *= multi_factor
                if self.gcn_bool:
                    self.gconv.append(gcn2(dilation_channels, residual_channels, dropout,
                                                              support_len=supports_len))

        
        
        #self.gcnpool = gcn3(skip_channels, int(skip_channels*2),dropout, 1, 1,support_len=supports_len)
        #self.gcnpool1 = gcn3(2912, 2912,dropout, 11, 3,support_len=supports_len)
        #self.gcnpool2 = gcn3(skip_channels, skip_channels,dropout, 3, 3,support_len=supports_len)
        self.end_module = nn.Sequential(
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels = skip_channels, out_channels = int(skip_channels/4), 
                        stride = (1,1), padding = (0,0), dilation = (1,1), kernel_size =(1,5)),
        )
        #self.end_module.apply(self.init_weights)
        self.end_module_2 = nn.Sequential(
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels = int(skip_channels/4), out_channels = 1,
                        stride = (1,2), padding = (0,0),  dilation = (1,1), kernel_size= (1,2)),
        )
        #self.end_module.apply(self.init_weights)
        self.gcnpool3 = gcn3(1, 1,dropout, 11, 3,support_len=supports_len)
        #print(out_dim)
        self.end_mlp_e = nn.Sequential(
            nn.LeakyReLU(), #nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_nodes, out_channels=out_nodes,
                      kernel_size=(1,1), bias=True)
        )
        self.end_mlp_f = nn.Sequential(
            nn.LeakyReLU(), #nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_dim, out_channels=out_dim_f,
                      kernel_size=(1,1), bias=True)
        )

        #self.gcnpool = gcn3(out_dim, out_dim, dropout,support_len=supports_len)
        self.pooling = pool(out_dim, out_nodes, dropout, supports_len)

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

    def tcn(self, filter_conv, gate_conv, skip_conv, skip):
        '''combine dilated conv + skip parts, entire tcn'''
        def custom_forward(residual, dummy_arg = None):
            # dilated convolution
            assert dummy_arg is not None
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

    def forward(self, input, supports, aptinit):
        # inputs: [batch_size, 1, num_nodes, in_len+1], supports: len 2, each (batch_size, num_nodes, num_nodes)
        ### deal with supports
        batch_size = len(input)
        #if self.gcn_bool and self.addaptadj:
            #if supports is None:
            #        supports = []
            #if aptinit is None:
             #   nodevec1 = nn.Parameter(torch.randn(batch_size, 
             #                           self.num_nodes, 10).to(self.device), 
             #                           requires_grad=True).to(self.device)
             #   nodevec2 = nn.Parameter(torch.randn(batch_size, 
             #                           10, self.num_nodes).to(self.device), 
             #                           requires_grad=True).to(self.device)

            #else:
                #m, p, n = torch.svd(aptinit)
                #_p = torch.diag_embed(p[:, :10] ** 0.5)
                #initemb1 = torch.bmm(m[:, :, :10], _p)
                #initemb2 = torch.bmm(_p, torch.transpose(n[:, :, :10], 1, 2))
                #nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(self.device)
                #nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(self.device)

        ### normal forward
        # print(self.receptive_field)
        in_len = input.size(3)
        if in_len<self.receptive_field:
            print("Padded input")
            x = input
            #x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input
        
        #x = self.test_start_conv(x)
        x = self.start_conv(x)
        #print(x.shape)
        #plt.figure()
        #for i in range(5):
        #    plt.plot(x[0,i,0,:].detach().cpu(), label= 'E after startconv')
            #plt.legend()
            #plt.show()
        #plt.legend()
        #plt.show()
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        #new_supports = None
        #if self.gcn_bool and self.addaptadj and supports is not None:
        #    adp = F.softmax(F.relu(torch.matmul(nodevec1, nodevec2)), dim=2)
        #    new_supports = supports + [adp]
        #    del adp
        #    del nodevec1, nodevec2
        
        #print(sum(new_supports))
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
            #print("Mean of residual " +str(residual.mean()))

            # x = checkpoint(self.diconv(self.filter_convs[i], self.gate_convs[i]), residual)
            # skip = checkpoint(self.skip_part(self.skip_convs[i], skip), x)

            #x, skip = self.tcn(self.filter_convs[i], self.gate_convs[i],
#                                 self.skip_convs[i], skip)(residual, self.dummy_tensor)
            #print(x.shape)
            #print(skip.shape)
            # dilated convolution
            filter = self.filter_convs[i](residual)
            #m = nn.LeakyReLU()
            filter = torch.tanh(filter)
            #print("Filter mean: " + str(filter.mean()))
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            #print("Gate mean: " + str(gate.mean()))
            x = filter * gate
            #print("Mean after tcn: " + str(x.mean()))
            #x = filter
            #print(x.mean())
            # del filter
            # del gate

            # parametrized skip connection
            s = x
            s = self.skip_convs[i](s)
            #s = self.leaky_relu(s)
            try:
                #skip = self.pool_layers[i](skip)
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip
            #skip = self.leaky_relu(skip)
            #print("Mean of skip: " +str(skip.mean()))
            
            if self.gcn_bool and supports is not None:
                if self.addaptadj:
                    # x: [64, 32, 80, 15]
                    x = self.gconv[i](x, *new_supports)
                    #print("Mean after gconv: " + str(x.mean()))
                    #x = checkpoint(self.gconv[i], x, *new_supports)
                else:
                    x = self.gconv[i](x, *supports)
                    #print("Mean after gconv: " + str(x.mean()))
                    #x = checkpoint(self.gconv[i], x, *supports)
            else:
                x = self.residual_convs[i](x)
                #x = checkpoint(self.residual_convs[i], x)
            
            #print(x.sum())
            #residual = self.residual_convs[i](residual)
            x = x + residual[:, :, :, -x.size(3):]
            #x = self.leaky_relu(x)
            #print("After resid " + str(x.mean()))
            #x = self.bn[i](x)
            #print("After bm " + str(x.mean()))
            #print(x.sum())
            #print(x.sum())
            #GPUtil.showUtilization()
            #print(x.shape)

        #print("Mean bfore end module 1: " + str(skip.mean()))
        x = self.end_module(skip)
        #print("Mean bfor emodule 2: " +str(x.mean()))
        x = self.end_module_2(x)
        #x = checkpoint(self.gcnpool1, x, *new_supports)
        #print("At end of model")
        #GPUtil.showUtilization()
        return x
