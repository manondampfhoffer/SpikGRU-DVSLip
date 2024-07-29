"""
Implement neural network model
"""
import torch
import torch.nn as nn
import numpy as np

from layers import SCNNlayer, SBasicBlock, SFCLayer, GRUlayer, SAdaptiveAvgPool2d, SAvgPool2d, LiGRU


class SCNN(nn.Module): 
    """ RESNET-18 frontend + 3 BiGRU backend
        if front=True: only using frontend (with FC backend)
    """
    def __init__(self, args, num_classes, useBN, ternact, ann=False, front=False, NObidirectional=False, singlegate=False, hybridsign=False, hybridANN=False):
        super(SCNN, self).__init__()
        self.ann = ann
        self.front = front
        self.hybridANN = hybridANN

        Cin1 = 2
        Cin2 = 64
        Cin3 = 2*Cin2; Cin4 = 2*Cin3; Cin5 = 2*Cin4
        kernel_size_in = (7,7)
        kernel_size = (3,3)
        padding_in = (3,3)
        padding = (1,1)
        dilatation = (1,1)
        stride1 = (1,1)
        stride2 = (2,2)
        output_shape = Cin5 * 1 * 1
        gru_hidden_size = 1024

        kernel_size_3d = (5,7,7)
        stride3d = (1,2,2)
        padding3d = (2,3,3)
        dilatation3d = (1,1,1)

        if hybridsign:
            ternact = False
        self.layer1 = SCNNlayer(args, 44, 44, Cin1, Cin2, kernel_size_3d, dilatation3d, stride3d, padding3d, useBN=useBN, ternact=ternact, conv3d=True, ann=ann)
        self.avgpool = SAvgPool2d(args, (3,3),(2,2),(1,1), 22, Cin2, ternact=ternact, ann=ann)
        self.layer2_1 = SBasicBlock(args, 22, 22, Cin2, Cin2, kernel_size, dilatation, stride1, padding, useBN=useBN, ternact=ternact, ann=ann)
        self.layer2_2 = SBasicBlock(args, 22, 22, Cin2, Cin2, kernel_size, dilatation, stride1, padding, useBN=useBN, ternact=ternact, ann=ann)
        self.layer3_1 = SBasicBlock(args, 11, 11, Cin2, Cin3, kernel_size, dilatation, stride2, padding, useBN=useBN, ternact=ternact, ann=ann)
        self.layer3_2 = SBasicBlock(args, 11, 11, Cin3, Cin3, kernel_size, dilatation, stride1, padding, useBN=useBN, ternact=ternact, ann=ann)
        self.layer4_1 = SBasicBlock(args, 6, 6, Cin3, Cin4, kernel_size, dilatation, stride2, padding, useBN=useBN, ternact=ternact, ann=ann)
        self.layer4_2 = SBasicBlock(args, 6, 6, Cin4, Cin4, kernel_size, dilatation, stride1, padding, useBN=useBN, ternact=ternact, ann=ann)
        self.layer5_1 = SBasicBlock(args, 3, 3, Cin4, Cin5, kernel_size, dilatation, stride2, padding, useBN=useBN, ternact=ternact, ann=ann)
        self.layer5_2 = SBasicBlock(args, 3, 3, Cin5, Cin5, kernel_size, dilatation, stride1, padding, useBN=useBN, ternact=ternact, ann=ann)
        
        self.adaptavgpool = SAdaptiveAvgPool2d(args, (1,1), Cin5, ternact=ternact, ann=ann)
        self.flatten = nn.Flatten(start_dim=2, end_dim=-1) #we do not want to flatten through T (timesteps) dim
        self.dropout = nn.Dropout(p=0.5)

        if self.front: #front end only with stateful (non-spiking) synapses
            self.dense = SFCLayer(args, output_shape, num_classes, ann=False, stateful=True)       
        else:
            bidirectional = not(NObidirectional)
            twogates = not(singlegate)
            if hybridsign:
                ternact = True
            if hybridANN:
                ann = True
            if ann:
                self.gru = nn.GRU(input_size=output_shape, hidden_size=gru_hidden_size, num_layers=3, bias=True, batch_first=True, dropout=0.2, bidirectional=bidirectional)
            else:
                self.gru = LiGRU(args, twogates, 3, bidirectional, 0.2, output_shape, gru_hidden_size, ann=ann, ternact=ternact)
            
            if bidirectional:
                self.dense = SFCLayer(args, gru_hidden_size *2, num_classes, ann=True, stateful=False)       
            else:
                self.dense = SFCLayer(args, gru_hidden_size, num_classes, ann=True, stateful=False)       

    def clamp(self):
        self.layer1.clamp()
        self.layer2_1.clamp()
        self.layer2_2.clamp()
        self.layer3_1.clamp()
        self.layer3_2.clamp()
        self.layer4_1.clamp()
        self.layer4_2.clamp()
        self.layer5_1.clamp()
        self.layer5_2.clamp()
        if not self.ann and not self.front and not self.hybridANN:
            self.gru.clamp()
        self.dense.clamp()

    def forward(self, x):
        # In: (N, T, Cin, X, Y)
        out1 = self.layer1(x)
        out1pool = self.avgpool(out1)
        out2_1, out2_11 = self.layer2_1(out1pool)
        out2_2, out2_21 = self.layer2_2(out2_1)
        out3_1, out3_11 = self.layer3_1(out2_2)
        out3_2, out3_21 = self.layer3_2(out3_1)
        out4_1, out4_11 = self.layer4_1(out3_2)
        out4_2, out4_21 = self.layer4_2(out4_1)
        out5_1, out5_11 = self.layer5_1(out4_2)
        out5_2, out5_21 = self.layer5_2(out5_1)
        out5_2pool = self.adaptavgpool(out5_2)
        out5_2pool = self.flatten(out5_2pool)
        out5_2pool = self.dropout(out5_2pool)
        
        if self.front:
            out9 = self.dropout(out5_2pool)
            out = self.dense(out9)
        else:
            if self.ann or self.hybridANN: #GRU pytorch
                out8, _ = self.gru(out5_2pool)
                out7 =  out8
                out6 = out8
            else: ##LIGRU
                out8, out7, out6 = self.gru(out5_2pool)
            out9 = self.dropout(out8)
            out = self.dense(out9)

        if self.front:

            spike_act = [out1.abs().mean().detach().cpu().numpy(), out1pool.abs().mean().detach().cpu().numpy(), \
                out2_1.abs().mean().detach().cpu().numpy(), out2_11.abs().mean().detach().cpu().numpy(), \
                out2_2.abs().mean().detach().cpu().numpy(), out2_21.abs().mean().detach().cpu().numpy(), \
                out3_1.abs().mean().detach().cpu().numpy(), out3_11.abs().mean().detach().cpu().numpy(), \
                out3_2.abs().mean().detach().cpu().numpy(), out3_21.abs().mean().detach().cpu().numpy(), \
                out4_1.abs().mean().detach().cpu().numpy(), out4_11.abs().mean().detach().cpu().numpy(), \
                out4_2.abs().mean().detach().cpu().numpy(), out4_21.abs().mean().detach().cpu().numpy(), \
                out5_1.abs().mean().detach().cpu().numpy(), out5_11.abs().mean().detach().cpu().numpy(), \
                out5_2.abs().mean().detach().cpu().numpy(), out5_21.abs().mean().detach().cpu().numpy(), \
                out5_2pool.abs().mean().detach().cpu().numpy()]

            loss_act = 0.5* ((out1**2).mean() + (out2_1**2).mean() + (out2_11**2).mean() + (out2_2**2).mean() + (out2_21**2).mean() \
                + (out3_1**2).mean() + (out3_11**2).mean() + (out3_2**2).mean() + (out3_21**2).mean() \
                + (out4_1**2).mean() + (out4_11**2).mean() + (out4_2**2).mean() + (out4_21**2).mean() \
                + (out5_1**2).mean() + (out5_11**2).mean() + (out5_2**2).mean() + (out5_21**2).mean() )
        else:

            spike_act = [out1.abs().mean().detach().cpu().numpy(), out1pool.abs().mean().detach().cpu().numpy(), \
                out2_1.abs().mean().detach().cpu().numpy(), out2_11.abs().mean().detach().cpu().numpy(), \
                out2_2.abs().mean().detach().cpu().numpy(), out2_21.abs().mean().detach().cpu().numpy(), \
                out3_1.abs().mean().detach().cpu().numpy(), out3_11.abs().mean().detach().cpu().numpy(), \
                out3_2.abs().mean().detach().cpu().numpy(), out3_21.abs().mean().detach().cpu().numpy(), \
                out4_1.abs().mean().detach().cpu().numpy(), out4_11.abs().mean().detach().cpu().numpy(), \
                out4_2.abs().mean().detach().cpu().numpy(), out4_21.abs().mean().detach().cpu().numpy(), \
                out5_1.abs().mean().detach().cpu().numpy(), out5_11.abs().mean().detach().cpu().numpy(), \
                out5_2.abs().mean().detach().cpu().numpy(), out5_21.abs().mean().detach().cpu().numpy(), \
                out5_2pool.abs().mean().detach().cpu().numpy(), out6.abs().mean().detach().cpu().numpy(), \
                out7.abs().mean().detach().cpu().numpy(), out8.abs().mean().detach().cpu().numpy()]
            
            loss_act = 0.5* (((out1**2).mean() + (out2_1**2).mean() + (out2_11**2).mean() + (out2_2**2).mean() + (out2_21**2).mean() \
                + (out3_1**2).mean() + (out3_11**2).mean() + (out3_2**2).mean() + (out3_21**2).mean() \
                + (out4_1**2).mean() + (out4_11**2).mean() + (out4_2**2).mean() + (out4_21**2).mean() \
                + (out5_1**2).mean() + (out5_11**2).mean() + (out5_2**2).mean() + (out5_21**2).mean() \
                + (out6**2).mean() + (out7**2).mean() + (out8**2).mean()).mean())


        return out, loss_act, spike_act