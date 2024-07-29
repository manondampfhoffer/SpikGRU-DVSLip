"""
Implement custom layers
"""

import torch
import torch.nn as nn
import numpy as np

Vth = 1.0
alpha_init_gru = 0.9
alpha_init_conv = 0.9
gamma = 10

class SpikeAct(torch.autograd.Function): 
    @staticmethod
    def forward(ctx, x_input):
        ctx.save_for_backward(x_input)
        output = torch.ge(x_input, Vth) 
        return output.float()

    @staticmethod
    def backward(ctx, grad_output):
        x_input, = ctx.saved_tensors 
        grad_input = grad_output.clone()
         ## derivative of arctan (scaled)
        grad_input = grad_input * 1 / (1 + gamma * (x_input - Vth)**2)
        return grad_input

class SpikeAct_signed(torch.autograd.Function): ## ternact
    @ staticmethod
    def forward(self, x):
        self.save_for_backward(x)
        x_forward = torch.clamp(torch.sign(x + Vth)+torch.sign(x - Vth), min=-1, max=1)
        return x_forward

    @ staticmethod
    def backward(self, grad_output):
        x_input, = self.saved_tensors
        grad_input = grad_output.clone()
        ## derivative of arctan (scaled)
        scale = 1 + 1/(1 + 4*Vth**2*gamma)
        grad_input = grad_input * 1/scale * (1/(1+ gamma * ((x_input - Vth)**2)) \
                                            + 1/(1+ gamma * ((x_input + Vth)**2))) 
        return grad_input


class SCNNlayer(nn.Module):
    """ spiking 2D (or 3D if conv3d=True) convolution layer
        ann mode if ann=True
    """
    def __init__(self, args, height, width, in_channels, out_channels, kernel_size, dilation, stride, padding, useBN, ternact, conv3d=False, ann=False):
        super(SCNNlayer, self).__init__()
        self.conv3d = conv3d
        self.ann = ann
        self.height = height
        self.width = width
        if not self.ann:
            self.alpha = nn.Parameter(torch.zeros(out_channels).uniform_(alpha_init_conv,alpha_init_conv)) #1 per output channel only, as biases
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.args = args
        if ternact:
            self.spikeact = SpikeAct_signed.apply
        else:
            self.spikeact = SpikeAct.apply
        self.useBN = useBN
        if self.conv3d:
            if self.useBN:
                self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=1, bias=False, padding_mode='zeros')
                self.bn = nn.BatchNorm3d(out_channels)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=1, bias=True, padding_mode='zeros')
        else:
            if self.useBN:
                self.bn = nn.BatchNorm2d(out_channels)
                self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=1, bias=False, padding_mode='zeros')
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=1, bias=True, padding_mode='zeros')
            
        self.clamp()

        if self.ann:
            ## resnet init (kaiming normal mode fanout)
            n = out_channels * np.prod(kernel_size)
            nn.init.normal_(self.conv.weight, std= np.sqrt(2 / n))
            if not self.useBN:
                nn.init.zeros_(self.conv.bias)
        else:
            k = np.sqrt(6 / (in_channels*np.prod(kernel_size)))
            nn.init.uniform_(self.conv.weight, a=-k, b=k)

        if self.useBN:
            nn.init.constant_(self.bn.weight, 1)
            nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        ## x : (B, T, Cin, Y, X)
        T = x.size(1)
        B = x.size(0)
        outputs = torch.zeros((B, T, self.out_channels, self.height, self.width), device = x.device)
        mem = torch.zeros((B, self.out_channels, self.height, self.width), device = x.device)
        output_prev = torch.zeros_like(mem)

        # parallel conv and batchnorm
        if self.conv3d:
            x = x.permute(0,2,1,3,4)
            conv_all = self.conv(x)
            if self.useBN:
                conv_all = self.bn(conv_all)
            conv_all = conv_all.permute(0,2,1,3,4)
        else:
            x = x.contiguous()
            x = x.view(-1, x.size(2), x.size(3), x.size(4)) #fuse T and B dim
            conv_all = self.conv(x)
            if self.useBN:
                conv_all = self.bn(conv_all)
            conv_all = conv_all.view(B, T, self.out_channels, self.height, self.width)

        if self.ann:
            conv_all = torch.relu(conv_all)
            outputs = conv_all
        else:
            for t in range(T):
                conv_xt = conv_all[:,t,:,:,:]

                ## SNN LIF
                mem = torch.einsum("abcd,b->abcd", mem, self.alpha) # with 1 time constant per output channel
                mem = mem + conv_xt - Vth * output_prev
                output_prev = self.spikeact(mem)
                outputs[:,t,:,:,:] = output_prev
               
        return outputs

    def clamp(self):
        if not self.ann:
            self.alpha.data.clamp_(0.,1.)


class SBasicBlock(nn.Module):
    """ Spiking Resnet basic block
        ann mode if ann=True
    """ 
    def __init__(self, args, height, width, in_channels, out_channels, kernel_size, dilation, stride, padding, useBN, ternact, ann=False):
        super(SBasicBlock, self).__init__()
        self.ann = ann
        self.height = height
        self.width = width
        if not self.ann:
            self.alpha1 = nn.Parameter(torch.zeros(out_channels).uniform_(alpha_init_conv, alpha_init_conv)) #1 per output channel only, as biases
            self.alpha2 = nn.Parameter(torch.zeros(out_channels).uniform_(alpha_init_conv, alpha_init_conv))
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.args = args
        if ternact:
            self.spikeact = SpikeAct_signed.apply
        else:
            self.spikeact = SpikeAct.apply
        self.useBN = useBN
        
        if self.useBN:
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=False)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, (1, 1), padding=padding, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, (1, 1), padding=padding, bias=True)
        
        if self.ann:
            ## resnet init (kaiming normal mode fanout)
            n = out_channels * np.prod(kernel_size)
            nn.init.normal_(self.conv1.weight, std= np.sqrt(2 / n))
            nn.init.normal_(self.conv2.weight, std= np.sqrt(2 / n))
            if not self.useBN:
                nn.init.zeros_(self.conv1.bias)
                nn.init.zeros_(self.conv2.bias)
        else:
            k1 = np.sqrt(6 /(self.in_channels*np.prod(self.kernel_size)))
            k2 = np.sqrt(6 /(self.out_channels*np.prod(self.kernel_size)))
            nn.init.uniform_(self.conv1.weight, a=-k1, b=k1)
            nn.init.uniform_(self.conv2.weight, a=-k2, b=k2)
        if self.useBN:
            nn.init.constant_(self.bn1.weight, 1)
            nn.init.constant_(self.bn1.bias, 0)
            nn.init.constant_(self.bn2.weight, 1)
            nn.init.constant_(self.bn2.bias, 0)

        if self.stride != (1,1):
            if self.useBN:
                self.downsample = nn.Conv2d(in_channels, out_channels, (1, 1), stride, padding=(0,0), bias=False)
                self.bn3 = nn.BatchNorm2d(out_channels)
                nn.init.constant_(self.bn3.weight, 1)
                nn.init.constant_(self.bn3.bias, 0)
            else:
                self.downsample = nn.Conv2d(in_channels, out_channels, (1, 1), stride, padding=(0,0), bias=True)
            if self.ann:
                # ## resnet init (kaiming normal mode fanout)
                n = out_channels
                nn.init.normal_(self.downsample.weight, std= np.sqrt(2 / n))
                if not self.useBN:
                    nn.init.zeros_(self.downsample.bias)
            else:
                k3 = np.sqrt(6 /(self.in_channels)) # kernel_size == (1,1)
                nn.init.uniform_(self.downsample.weight, a=-k3, b=k3)

        self.clamp()


    def forward(self, x):
        ## x : (B, T, Cin, Y, X)
        T = x.size(1)
        B = x.size(0)
        outputs = torch.zeros((B, T, self.out_channels, self.height, self.width), device = x.device)
        outputs1 = torch.zeros_like(outputs)
        mem1 = torch.zeros((B, self.out_channels, self.height, self.width), device = x.device)
        mem2 = torch.zeros_like(mem1)
        output_prev1 = torch.zeros_like(mem1)
        output_prev2 = torch.zeros_like(mem1)

        if self.ann:
            x = x.contiguous()
            identity = x.view(-1, x.size(2), x.size(3), x.size(4)) #fuse T and B dim
            conv_all = self.conv1(identity)
            if self.useBN:
                conv_all = self.bn1(conv_all)
            conv_all = torch.relu(conv_all) 
            
            outputs1 = conv_all
            conv_all = self.conv2(conv_all)
            if self.useBN:
                conv_all = self.bn2(conv_all)
            if self.stride != (1,1):
                identity = self.downsample(identity)
                if self.useBN:
                    identity = self.bn3(identity)
            conv_all = conv_all + identity
            conv_all = torch.relu(conv_all)
            conv_all = conv_all.view(B, T, conv_all.size(1), conv_all.size(2), conv_all.size(3))
            outputs = conv_all
        else:
            x = x.contiguous()
            identity = x.view(-1, x.size(2), x.size(3), x.size(4)) #fuse T and B dim

            conv1 = self.conv1(identity)
            if self.useBN:
                conv1 = self.bn1(conv1)
            conv1 = conv1.view(B, T, conv1.size(1), conv1.size(2), conv1.size(3))

            for t in range(T):
                ## SNN LIF
                mem1 = torch.einsum("abcd,b->abcd", mem1, self.alpha1) # with 1 time constant per output channel #LIF NEURONS
                mem1 = mem1 + conv1[:,t,:,:,:] - Vth * output_prev1
                output_prev1 = self.spikeact(mem1)
                outputs1[:,t,:,:,:] = output_prev1
            
            outputs1 = outputs1.contiguous()
            input2 = outputs1.view(-1, outputs1.size(2), outputs1.size(3), outputs1.size(4)) #fuse T and B dim
            
            conv2 = self.conv2(input2)
            if self.useBN:
                conv2 = self.bn2(conv2)

            if self.stride != (1,1):
                identity = self.downsample(identity)
                if self.useBN:
                    identity = self.bn3(identity)
            conv_all = conv2 + identity
            conv_all = conv_all.view(B, T, conv_all.size(1), conv_all.size(2), conv_all.size(3))
            
            for t in range(T):
                ## SNN LIF
                mem2 = torch.einsum("abcd,b->abcd", mem2, self.alpha2) # with 1 time constant per output channel #LIF NEURONS
                mem2 = mem2 + conv_all[:,t,:,:,:] - Vth * output_prev2
                output_prev2 = self.spikeact(mem2)
                outputs[:,t,:,:,:] = output_prev2

        return outputs, outputs1

    def clamp(self):
        if not self.ann:
            self.alpha1.data.clamp_(0.,1.)
            self.alpha2.data.clamp_(0.,1.)
        

class SFCLayer(nn.Module):
    """ leaky integrator layer. if stateful=True, implement the stateful synapse version of the leaky integrator
        ann mode (=simple fully connected layer) if ann=True
    """
    def __init__(self, args, in_size, out_size, ann=False, stateful=False):
        super(SFCLayer, self).__init__()
        self.ann = ann
        self.in_size = in_size
        self.out_size = out_size
        self.dense = nn.Linear(in_size, out_size, bias=True)
        self.stateful = stateful
        if not self.ann:
            self.alpha = nn.Parameter(torch.zeros(out_size).uniform_(alpha_init_gru, alpha_init_gru))
        if stateful:
            self.beta = nn.Parameter(torch.zeros(out_size).uniform_(alpha_init_gru, alpha_init_gru))
        self.args = args

    def forward(self, x):
        # X : (B, T, N)
        T = x.size(1)
        B = x.size(0)
        outputs = torch.zeros((B, T, self.out_size), device = x.device)
        potential = torch.zeros((B, self.out_size), device = x.device)
        current = torch.zeros((B, self.out_size), device = x.device)

        if self.ann:
            outputs = self.dense(x)
        else:
            if self.stateful:
                for t in range(T):
                    out = self.dense(x[:,t,:])
                    current = self.beta * current + out
                    potential = self.alpha * potential + (1 - self.alpha) * current
                    outputs[:,t,:] = potential
            else: 
                for t in range(T):
                    out = self.dense(x[:,t,:])
                    potential = self.alpha * potential + out
                    outputs[:,t,:] = potential

        return outputs

    def clamp(self):
        if not self.ann:
            self.alpha.data.clamp_(0.,1.)




class GRUlayer(nn.Module):
    """ spiking GRU layer
        ann mode if ann=True
        SpikGRU2+ if twogates=True and ternact=True
    """
    def __init__(self, args, input_size, hidden_size, ann, ternact, twogates=False):
        super(GRUlayer, self).__init__()
        self.ann = ann
        self.twogates = twogates
        self.hidden_size = hidden_size
        self.wz = nn.Linear(input_size, hidden_size, bias=True)
        self.wi = nn.Linear(input_size, hidden_size, bias=True)
        self.uz = nn.Linear(hidden_size, hidden_size, bias=False)
        self.ui = nn.Linear(hidden_size, hidden_size, bias=True)
        if self.twogates:
            self.wr = nn.Linear(input_size, hidden_size, bias=True)
            self.ur = nn.Linear(hidden_size, hidden_size, bias=False)
        if not self.ann:
            self.alpha = nn.Parameter(torch.zeros(hidden_size).uniform_(alpha_init_gru, alpha_init_gru))
        self.clamp()
        self.args = args
        if ternact:
            self.spikeact = SpikeAct_signed.apply
        else:
            self.spikeact = SpikeAct.apply

        k_ff = np.sqrt(1./hidden_size)
        k_rec = np.sqrt(1./hidden_size)
        nn.init.uniform_(self.wi.weight, a=-k_ff, b=k_ff)
        nn.init.uniform_(self.wz.weight, a=-k_ff, b=k_ff)
        nn.init.uniform_(self.ui.weight, a=-k_rec, b=k_rec)
        nn.init.uniform_(self.uz.weight, a=-k_rec, b=k_rec)
        nn.init.uniform_(self.wi.bias, a=-k_ff, b=k_ff)
        nn.init.uniform_(self.wz.bias, a=-k_ff, b=k_ff)
        if self.twogates:
            nn.init.uniform_(self.wr.weight, a=-k_ff, b=k_ff)
            nn.init.uniform_(self.ur.weight, a=-k_rec, b=k_rec)
            nn.init.uniform_(self.wr.bias, a=-k_ff, b=k_ff)


    def forward(self, x):
        T = x.size(1)
        B = x.size(0)
        outputs = torch.zeros((B, T, self.hidden_size), device = x.device)
        output_prev = torch.zeros((B, self.hidden_size), device = x.device)
        temp = torch.zeros_like(output_prev)
        tempcurrent = torch.zeros_like(output_prev)

        for t in range(T): 
            
            tempZ = torch.sigmoid(self.wz(x[:,t,:]) + self.uz(output_prev)) 
            if self.twogates:
                tempR = torch.sigmoid(self.wr(x[:,t,:]) + self.ur(output_prev))
            if self.ann:
                if self.twogates:
                    tempcurrent = torch.tanh(self.wi(x[:,t,:]) + self.ui(output_prev) * tempR)
                else:
                    tempcurrent = torch.tanh(self.wi(x[:,t,:]) + self.ui(output_prev))
            else:
                if self.twogates:
                    tempcurrent = self.alpha * tempcurrent + self.wi(x[:,t,:]) + self.ui(output_prev) * tempR
                else:
                    tempcurrent = self.alpha * tempcurrent + self.wi(x[:,t,:]) + self.ui(output_prev)
                
            if self.ann:
                temp = tempZ * temp + (1 - tempZ) * tempcurrent
                output_prev = temp
            else:
                temp = tempZ * temp + (1 - tempZ) * tempcurrent - Vth * output_prev
                output_prev = self.spikeact(temp)

            outputs[:,t,:] = output_prev

        return outputs

    def clamp(self):
        if not self.ann:
            self.alpha.data.clamp_(0.,1.)



class SAdaptiveAvgPool2d(nn.Module):
    """ spiking adaptive avg pool 2d
        ann mode if ann=True
    """
    def __init__(self, args, kernel_size, channel_in, ternact, ann=False):
        super(SAdaptiveAvgPool2d, self).__init__()
        self.ann = ann
        self.avgpool = nn.AdaptiveAvgPool2d(kernel_size)
        self.kernel_size = kernel_size
        self.args = args
        if ternact:
            self.spikeact = SpikeAct_signed.apply
        else:
            self.spikeact = SpikeAct.apply

    def forward(self, x):
        # x: (B, T, Cin, Y, X)
        T = x.size(1)
        B = x.size(0)
        Cin = x.size(2)
        out = torch.zeros((B, T, Cin, self.kernel_size[0], self.kernel_size[0]), device = x.device)
        potential = torch.zeros((B, Cin, self.kernel_size[0], self.kernel_size[1]), device = x.device)
        output_prev = torch.zeros_like(potential)

        x = x.contiguous()
        x = x.view(-1, x.size(2), x.size(3), x.size(4)) #fuse T and B dim
        pool = self.avgpool(x)
        pool = pool.view(B, T, pool.size(1), pool.size(2), pool.size(3))

        if self.ann:
            out = pool
        else:
            for t in range(T):
                potential = potential + pool[:,t,:,:,:] - Vth * output_prev #IF neuron
                output_prev = self.spikeact(potential)
                out[:,t,:, :, :] = output_prev
        return out

class SAvgPool2d(nn.Module):
    """ spiking avg pool 2d
        ann mode if ann=True
    """
    def __init__(self, args, kernel, stride, padding, out_size, channel_in, ternact, ann=False):
        super(SAvgPool2d, self).__init__()
        self.ann = ann
        self.avgpool = nn.AvgPool2d(kernel, stride=stride, padding=padding)
        self.out_size = out_size
        self.args = args
        if ternact:
            self.spikeact = SpikeAct_signed.apply
        else:
            self.spikeact = SpikeAct.apply

    def forward(self, x):
        T = x.size(1)
        B = x.size(0)
        Cin = x.size(2)
        out = torch.zeros((B, T, Cin, self.out_size, self.out_size), device = x.device)
        potential = torch.zeros((B, Cin, self.out_size, self.out_size), device = x.device)
        output_prev = torch.zeros_like(potential)

        x = x.contiguous()
        x = x.view(-1, x.size(2), x.size(3), x.size(4)) #fuse T and B dim
        pool = self.avgpool(x)
        pool = pool.view(B, T, pool.size(1), pool.size(2), pool.size(3))

        if self.ann:
            out = pool
        else:
            for t in range(T):
                potential = potential + pool[:,t,:,:,:] - Vth * output_prev #IF neuron
                output_prev = self.spikeact(potential)
                out[:,t,:, :, :] = output_prev
        return out


class LiGRU(nn.Module):
    """ 3-layer bidrectionnal GRU backend
        ann mode if ann=True
    """
    def __init__(self, args, twogates, num_layers, bidirectional, dropout, input_size, hidden_size, ann, ternact):
        super(LiGRU, self).__init__()
        self.ann = ann
        self.hidden_size = hidden_size
        if ternact:
            self.spikeact = SpikeAct_signed.apply
        else:
            self.spikeact = SpikeAct.apply
        self.bidirectional = bidirectional
        self.args = args

        if num_layers != 3:
            print("Error in LiGRU: only defined with 3 layers")
        if self.bidirectional:
            self.grulayer1 = GRUlayer(args, input_size, hidden_size, ann=ann, ternact=ternact, twogates=twogates)
            self.grulayer2 = GRUlayer(args, hidden_size * 2, hidden_size, ann=ann, ternact=ternact, twogates=twogates)
            self.grulayer3 = GRUlayer(args, hidden_size * 2, hidden_size, ann=ann, ternact=ternact, twogates=twogates)
            self.grulayer1_b = GRUlayer(args, input_size, hidden_size, ann=ann, ternact=ternact, twogates=twogates)
            self.grulayer2_b = GRUlayer(args, hidden_size * 2, hidden_size, ann=ann, ternact=ternact, twogates=twogates)
            self.grulayer3_b = GRUlayer(args, hidden_size * 2, hidden_size, ann=ann, ternact=ternact, twogates=twogates)
        else:
            self.grulayer1 = GRUlayer(args, input_size, hidden_size, ann=ann, ternact=ternact, twogates=twogates)
            self.grulayer2 = GRUlayer(args, hidden_size, hidden_size, ann=ann, ternact=ternact, twogates=twogates)
            self.grulayer3 = GRUlayer(args, hidden_size, hidden_size, ann=ann, ternact=ternact, twogates=twogates)
            
        self.dropout = nn.Dropout(p=dropout)
        self.clamp()

    def forward(self, x):
        # x: [B, T, N]
        if self.bidirectional:
            x_b = torch.flip(x, [1])
            out1 = self.grulayer1(x)
            out1_b = self.grulayer1_b(x_b)
            out2 = self.grulayer2(self.dropout(torch.cat((out1, torch.flip(out1_b, [1])), 2)))
            out2_b = self.grulayer2_b(self.dropout(torch.cat((torch.flip(out1, [1]), out1_b), 2)))
            out3 = self.grulayer3(self.dropout(torch.cat((out2, torch.flip(out2_b, [1])), 2)))
            out3_b = self.grulayer3_b(self.dropout(torch.cat((torch.flip(out2, [1]), out2_b), 2)))
            outputs = torch.cat((out3, out3_b), 2)
        else:
            out1 = self.grulayer1(x)
            out2 = self.grulayer2(self.dropout(out1))
            out3 = self.grulayer3(self.dropout(out2))
            outputs = out3
        if self.bidirectional:
            return outputs, torch.cat((out2, out2_b), 2), torch.cat((out1, out1_b), 2)
        else:
            return outputs, out2, out1

    def clamp(self):
        if not self.ann:
            self.grulayer1.clamp()
            self.grulayer2.clamp()
            self.grulayer3.clamp()
            if self.bidirectional:
                self.grulayer1_b.clamp()
                self.grulayer2_b.clamp()
                self.grulayer3_b.clamp()