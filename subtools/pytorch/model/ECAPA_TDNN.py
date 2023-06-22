#! /usr/bin/python
# -*- encoding: utf-8 -*-

## Here, log_input forces alternative mfcc implementation with pre-emphasis instead of actual log mfcc

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchaudio
import pdb
# from utils import PreEmphasis
import math
# import torch
import sys
sys.path.insert(0, 'subtools/pytorch')
import libs.support.utils as utils
from libs.nnet import *

class SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm1d(bottleneck),
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
            )

    def forward(self, input):
        x = self.se(input)
        return input * x

class Bottle2neck(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale = 4):

        super(Bottle2neck, self).__init__()

        width       = int(math.floor(planes / scale))
        
        self.conv1  = nn.Conv1d(inplanes, width*scale, kernel_size=1)
        self.bn1    = nn.BatchNorm1d(width*scale)
        self.scale = scale
        self.nums   = scale -1

        convs       = []
        bns         = []

        num_pad = math.floor(kernel_size/2)*dilation

        for i in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            bns.append(nn.BatchNorm1d(width))

        self.convs  = nn.ModuleList(convs)
        self.bns    = nn.ModuleList(bns)

        self.conv3  = nn.Conv1d(width*scale, planes, kernel_size=1)
        self.bn3    = nn.BatchNorm1d(planes)

        self.relu   = nn.ReLU()

        self.width  = width
        self.se     = SEModule(planes)

    def forward(self, x):
        out_list = []
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)
       
        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i==0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(sp)
            sp = self.bns[i](sp)
            out_list.append(sp)

        if self.scale != 1:
            out_list.append(spx[self.nums])
        out = torch.cat(out_list, dim=1)
        # return out

            # if i==0:
            #     out = sp
            # else:
            #     out = torch.cat((out, sp), 1)
      
        # out = torch.cat((out, spx[self.nums]),1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)
        
        out = self.se(out)
        
        out += residual

        return out 








class ECAPA(TopVirtualNnet):

    def init(self, inputs_dim,num_targets, C=1024, embd_dim=512,model_scale=8, encoder_type="ASP", context=False, summed=False,
            specaugment=False, specaugment_params={},
            aug_dropout=0., context_dropout=0., hidden_dropout=0., dropout_params={},
            tdnn_layer_params={},
            fc1=True, fc1_params={}, fc2_params={},
            margin_loss=False, margin_loss_params={},hard_pre_loss=False,
            use_step=False, step_params={},
            transfer_from="softmax_loss",
            training=True, extracted_embedding="near",
            out_bn=False, **kwargs):

        ## Params.
        default_dropout_params = {
            "type":"default", # default | random
            "start_p":0.,
            "dim":2,
            "method":"uniform", # uniform | normals
            "continuous":False,
            "inplace":True
        }

        default_tdnn_layer_params = {
            "nonlinearity":'relu', "nonlinearity_params":{"inplace":True},
            "bn-relu":False, "bn":True, "bn_params":{"momentum":0.5, "affine":False, "track_running_stats":True}
        }
        
        default_pooling_params = {
            "num_nodes":1500,
            "num_head":1,
            "share":True,
            "affine_layers":1,
            "hidden_size":64,
            "context":[0],
            "stddev":True,
            "temperature":False, 
            "fixed":True,
            "stddev":True
        }

        default_margin_loss_params = {
            "method":"am", "m":0.2, 
            "feature_normalize":True, "s":30, 
            "double":False,
            "mhe_loss":False, "mhe_w":0.01,
            "inter_loss":0.,
            "ring_loss":0.,
            "curricular":False,
            # "noise":False,
            # "total_iter":1000000,
            # "double_target":False,
            # "reg_loss":False,
            # "sqrt":False,
            # "square":False,
            # "sub_group":False
        }

        default_fc_params = {
            "nonlinearity":'relu', "nonlinearity_params":{"inplace":True},
            "bn-relu":True, 
            "bn":True, 
            "bn_params":{"momentum":0.5, "affine":True, "track_running_stats":True}
            }

        default_step_params = {
            "T":None,
            "m":False, "lambda_0":0, "lambda_b":1000, "alpha":5, "gamma":1e-4,
            "s":False, "s_tuple":(30, 12), "s_list":None,
            "t":False, "t_tuple":(0.5, 1.2), 
            "p":False, "p_tuple":(0.5, 0.1)
        }

        dropout_params = utils.assign_params_dict(default_dropout_params, dropout_params)
        tdnn_layer_params = utils.assign_params_dict(default_tdnn_layer_params, tdnn_layer_params)

        margin_loss_params = utils.assign_params_dict(default_margin_loss_params, margin_loss_params)
        step_params = utils.assign_params_dict(default_step_params, step_params)
        fc1_params = utils.assign_params_dict(default_fc_params, fc1_params)
        fc2_params = utils.assign_params_dict(default_fc_params, fc2_params)


        self.use_step = use_step
        self.step_params = step_params
        self.extracted_embedding = extracted_embedding # For extract.


        self.context    = context
        self.summed     = summed
        self.n_mfcc     = inputs_dim
        # self.log_input  = log_input
        self.encoder_type = encoder_type
        self.out_bn     = out_bn

        self.scale  = model_scale
         ## Nnet.
        # Head
        # self.mixup = Mixup(alpha=mixup_alpha) if mixup else None
        self.specaugment = SpecAugment(**specaugment_params) if specaugment else None
        self.aug_dropout = get_dropout_from_wrapper(aug_dropout, dropout_params)
        self.context_dropout = ContextDropout(p=context_dropout) if context_dropout > 0 else None
        self.hidden_dropout = get_dropout_from_wrapper(hidden_dropout, dropout_params)
        # Frame level
        self.conv1  = nn.Conv1d(self.n_mfcc, C, kernel_size=5, stride=1, padding=2)
        self.relu   = nn.ReLU()
        self.bn1    = nn.BatchNorm1d(C)
        
        self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=self.scale)
        self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=self.scale)
        self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=self.scale)
        self.layer4 = nn.Conv1d(3*C, 1536, kernel_size=1)

        # self.instancenorm   = nn.InstanceNorm1d(self.n_mfcc)
        # self.torchmfcc  = torch.nn.Sequential(
        #     PreEmphasis(),
        #     torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc = self.n_mfcc, log_mels=self.log_input, dct_type = 2, melkwargs={'n_mels': 80, 'n_fft':512, 'win_length':400, 'hop_length':160, 'f_min':20, 'f_max':7600, 'window_fn':torch.hamming_window}),
        #     )
        # pooling
        if self.context:
            attn_input = 1536*3
        else:
            attn_input = 1536

        if self.encoder_type == 'ECA':
            attn_output = 1536
        elif self.encoder_type == 'ASP':
            attn_output = 1
        else:
            raise ValueError('Undefined encoder')


        self.attention = nn.Sequential(
            nn.Conv1d(attn_input, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, attn_output, kernel_size=1),
            nn.Softmax(dim=2),
            )

        self.bn5 = nn.BatchNorm1d(3072)

        self.fc1 = ReluBatchNormTdnnLayer(3072, embd_dim, **fc1_params) if fc1 else None

        if fc1:
            fc2_in_dim = embd_dim
        else:
            fc2_in_dim = 3072
        
        self.fc2 = ReluBatchNormTdnnLayer(fc2_in_dim, embd_dim, **fc2_params)

        # self.tail_dropout = torch.nn.Dropout2d(p=tail_dropout) if tail_dropout > 0 else None

        if training :
            if margin_loss:
                self.loss = MarginSoftmaxLoss(embd_dim, num_targets, **margin_loss_params)
            
            else:
                self.loss = SoftmaxLoss(embd_dim, num_targets)

            self.transform_keys = ["conv1","layer1","layer2","layer3","layer4","fc1","fc2"]

            if margin_loss and transfer_from == "softmax_loss":
                # For softmax_loss to am_softmax_loss
                self.rename_transform_keys = {"loss.affine.weight":"loss.weight"} 

    @utils.for_device_free
    def forward(self, x):
        # x = self.auto(self.specaugment, x)
        # x = self.auto(self.aug_dropout, x)
        # x = self.auto(self.context_dropout, x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        print("1111",x.shape)
        if self.summed:
            x1 = self.layer1(x)
            x2 = self.layer2(x+x1)
            x3 = self.layer3(x+x1+x2)
        else:
            x1 = self.layer1(x)
            print("12",x1.shape)
            x2 = self.layer2(x1)
            print("13",x2.shape)
            x3 = self.layer3(x2)
            print("14",x3.shape)
        print("222")
        x = self.layer4(torch.cat((x1,x2,x3),dim=1))
        print("333",x.shape)
        x = self.relu(x)

        t = x.size()[-1]

        if self.context:
            global_x = torch.cat((x,torch.mean(x,dim=2,keepdim=True).repeat(1,1,t), torch.sqrt(torch.var(x,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t)), dim=1)
        else:
            global_x = x

        w = self.attention(global_x)
        print("444")

        # print(x.shape)
        # print(w.shape)
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-4) )

        x = torch.cat((mu,sg),1)

        x = self.bn5(x)
        print("555",x.shape)
        if len(x.shape) !=3:
            x = x.unsqueeze(dim=2)

        x = self.auto(self.fc1, x)
        
        x = self.fc2(x)
        # x = self.auto(self.tail_dropout, x)
        x = self.auto(self.hidden_dropout, x)

        # x = self.fc6(x)

        # if self.out_bn:
        #     x = self.bn6(x)
        # x = self.auto(self.hidden_dropout, x)
        # print(x.shape)
        return x

    @utils.for_device_free
    def get_loss(self, inputs, targets):
        return self.loss(inputs, targets)

    @utils.for_device_free
    def get_accuracy(self, targets):
        return self.loss.get_accuracy(targets)
    
    @for_extract_embedding(maxChunk=10000, isMatrix=True)
    def extract_embedding(self, inputs):
        x = self.conv1(inputs)
        x = self.relu(x)
        x = self.bn1(x)
        if self.summed:
            x1 = self.layer1(x)
            x2 = self.layer2(x+x1)
            x3 = self.layer3(x+x1+x2)
        else:
            x1 = self.layer1(x)
            x2 = self.layer2(x1)
            x3 = self.layer3(x2)
        x = self.layer4(torch.cat((x1,x2,x3),dim=1))
        x = self.relu(x)
        t = x.size()[-1]
        if self.context:
            global_x = torch.cat((x,torch.mean(x,dim=2,keepdim=True).repeat(1,1,t), torch.sqrt(torch.var(x,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t)), dim=1)
        else:
            global_x = x
        w = self.attention(global_x)
        # print(x.shape)
        # print(w.shape)
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-4) )
        x = torch.cat((mu,sg),1)
        x = self.bn5(x)
        if len(x.shape) !=3:
            x = x.unsqueeze(dim=2)
        if self.extracted_embedding == "far":
            assert self.fc1 is not None
            x = self.fc1.affine(x)

        elif self.extracted_embedding == "near":
            x = self.auto(self.fc1, x)
            x = self.fc2(x)

        return x
    

    def get_warmR_T(T_0, T_mult, epoch):
        n = int(math.log(max(0.05, (epoch / T_0 * (T_mult - 1) + 1)), T_mult))
        T_cur = epoch - T_0 * (T_mult ** n - 1) / (T_mult - 1)
        T_i = T_0 * T_mult ** (n)
        return T_cur, T_i

    def compute_decay_value(self, start, end, T_cur, T_i):
        # Linear decay in every cycle time.
        return start - (start - end)/(T_i-1) * (T_cur%T_i)

    def step(self, epoch, this_iter, epoch_batchs):
        # Heated up for t and s.
        # Decay for margin and dropout p.
        if self.use_step:
            if self.step_params["m"]:
                current_postion = epoch*epoch_batchs + this_iter
                lambda_factor = max(self.step_params["lambda_0"], 
                                 self.step_params["lambda_b"]*(1+self.step_params["gamma"]*current_postion)**(-self.step_params["alpha"]))
                self.loss.step(lambda_factor)

            if self.step_params["T"] is not None and (self.step_params["t"] or self.step_params["p"]):
                T_cur, T_i = get_warmR_T(*self.step_params["T"], epoch)
                T_cur = T_cur*epoch_batchs + this_iter
                T_i = T_i * epoch_batchs

            if self.step_params["t"]:
                self.loss.t = self.compute_decay_value(*self.step_params["t_tuple"], T_cur, T_i)

            if self.step_params["p"]:
                self.aug_dropout.p = self.compute_decay_value(*self.step_params["p_tuple"], T_cur, T_i)

            if self.step_params["s"]:
                self.loss.s = self.step_params["s_tuple"][self.step_params["s_list"][epoch]]
        

if __name__ == '__main__':
    ecapa= ECAPA(80,1211,encoder_type="ASP",context=True)

    a = torch.randn((512,80,200))
    out = ecapa(a)


    print(out.shape)
    