from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
sys.path.insert(0, 'subtools/pytorch')
import libs.support.utils as utils
from libs.nnet import *
import torch.utils.checkpoint as cp
from torch.nn.modules.utils import _single, _pair

# from layers import TimeDelay, TDNNLayer, MultiBranchDenseTDNNBlock, TransitLayer, DenseLayer, StatsPool



def get_nonlinear(config_str, channels):
    nonlinear = nn.Sequential()
    for i, name in enumerate(config_str.split('-')):
        if name == 'relu':
            nonlinear.add_module('relu', nn.ReLU(inplace=True))
        elif name == 'prelu':
            nonlinear.add_module('prelu', nn.PReLU(channels))
        elif name == 'batchnorm':
            nonlinear.add_module('batchnorm', nn.BatchNorm1d(channels))
        elif name == 'batchnorm_':
            nonlinear.add_module('batchnorm', nn.BatchNorm1d(channels, affine=False))
        else:
            raise ValueError('Unexpected module ({}).'.format(name))
    return nonlinear


def statistics_pooling(x, dim=-1, keepdim=False, unbiased=True, eps=1e-2):
    mean = x.mean(dim=dim)
    std = x.std(dim=dim, unbiased=unbiased)
    stats = torch.cat([mean, std], dim=-1)
    if keepdim:
        stats = stats.unsqueeze(dim=dim)
    return stats


def high_order_statistics_pooling(x, dim=-1, keepdim=False, unbiased=True, eps=1e-2):
    mean = x.mean(dim=dim)
    std = x.std(dim=dim, unbiased=unbiased)
    norm = (x - mean.unsqueeze(dim=dim)) / std.clamp(min=eps).unsqueeze(dim=dim)
    skewness = norm.pow(3).mean(dim=dim)
    kurtosis = norm.pow(4).mean(dim=dim)
    stats = torch.cat([mean, std, skewness, kurtosis], dim=-1)
    if keepdim:
        stats = stats.unsqueeze(dim=dim)
    return stats


class StatsPool(nn.Module):

    def forward(self, x):
        return statistics_pooling(x)


class HighOrderStatsPool(nn.Module):

    def forward(self, x):
        return high_order_statistics_pooling(x)


class TimeDelay(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=True):
        super(TimeDelay, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _single(kernel_size)
        self.stride = _single(stride)
        self.padding = _pair(padding)
        self.dilation = _single(dilation)
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels * kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            std = 1 / math.sqrt(self.out_channels)
            self.weight.normal_(0, std)
            if self.bias is not None:
                self.bias.normal_(0, std)

    def forward(self, x):
        x = F.pad(x, self.padding).unsqueeze(1)
        x = F.unfold(x, (self.in_channels,)+self.kernel_size, dilation=(1,)+self.dilation, stride=(1,)+self.stride)
        return F.linear(x.transpose(1, 2), self.weight, self.bias).transpose(1, 2)


class TDNNLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, bias=True, config_str='batchnorm-relu'):
        super(TDNNLayer, self).__init__()
        if padding < 0:
            assert kernel_size % 2 == 1, 'Expect equal paddings, but got even kernel size ({})'.format(kernel_size)
            padding = (kernel_size - 1) // 2 * dilation
        self.linear = TimeDelay(in_channels, out_channels, kernel_size, stride=stride,
                                padding=padding, dilation=dilation, bias=bias)
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x):
        x = self.linear(x)
        x = self.nonlinear(x)
        return x


class DenseTDNNLayer(nn.Module):

    def __init__(self, in_channels, out_channels, bn_channels, kernel_size, stride=1,
                 dilation=1, bias=False, config_str='batchnorm-relu', memory_efficient=False):
        super(DenseTDNNLayer, self).__init__()
        assert kernel_size % 2 == 1, 'Expect equal paddings, but got even kernel size ({})'.format(kernel_size)
        padding = (kernel_size - 1) // 2 * dilation
        self.memory_efficient = memory_efficient
        self.nonlinear1 = get_nonlinear(config_str, in_channels)
        self.linear1 = nn.Linear(in_channels, bn_channels, bias=False)
        self.nonlinear2 = get_nonlinear(config_str, bn_channels)
        self.linear2 = TimeDelay(bn_channels, out_channels, kernel_size, stride=stride,
                                 padding=padding, dilation=dilation, bias=bias)

    def bn_function(self, x):
        return self.linear1(self.nonlinear1(x).transpose(1, 2)).transpose(1, 2)

    def forward(self, x):
        if self.training and self.memory_efficient:
            x = cp.checkpoint(self.bn_function, x)
        else:
            x = self.bn_function(x)
        x = self.linear2(self.nonlinear2(x))
        return x


class DenseTDNNBlock(nn.ModuleList):

    def __init__(self, num_layers, in_channels, out_channels, bn_channels, kernel_size,
                 stride=1, dilation=1, bias=False, config_str='batchnorm-relu', memory_efficient=False):
        super(DenseTDNNBlock, self).__init__()
        for i in range(num_layers):
            layer = DenseTDNNLayer(
                in_channels=in_channels + i * out_channels,
                out_channels=out_channels,
                bn_channels=bn_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                bias=bias,
                config_str=config_str,
                memory_efficient=memory_efficient
            )
            self.add_module('tdnnd%d' % (i + 1), layer)

    def forward(self, x):
        for layer in self:
            x = torch.cat([x, layer(x)], 1)
        return x


class StatsSelect(nn.Module):

    def __init__(self, channels, branches, null=False, reduction=1):
        super(StatsSelect, self).__init__()
        self.gather = HighOrderStatsPool()
        self.linear1 = nn.Linear(channels * 4, channels // reduction)
        self.linear2 = nn.ModuleList()
        if null:
            branches += 1
        for _ in range(branches):
            self.linear2.append(nn.Linear(channels // reduction, channels))
        self.channels = channels
        self.branches = branches
        self.null = null
        self.reduction = reduction

    def forward(self, x):
        f = torch.cat([_x.unsqueeze(dim=1) for _x in x], dim=1)
        x = torch.sum(f, dim=1)
        x = self.linear1(self.gather(x))
        s = []
        for linear in self.linear2:
            s.append(linear(x).unsqueeze(dim=1))
        s = torch.cat(s, dim=1)
        s = F.softmax(s, dim=1).unsqueeze(dim=-1)
        if self.null:
            s = s[:, :-1, :, :]
        return torch.sum(f * s, dim=1)

    def extra_repr(self):
        return 'channels={}, branches={}, reduction={}'.format(
            self.channels, self.branches, self.reduction
        )


class MultiBranchDenseTDNNLayer(DenseTDNNLayer):

    def __init__(self, in_channels, out_channels, bn_channels, kernel_size, stride=1,
                 dilation=(1,), bias=False, null=False, reduction=1,
                 config_str='batchnorm-relu', memory_efficient=False):
        super(DenseTDNNLayer, self).__init__()
        assert kernel_size % 2 == 1, 'Expect equal paddings, but got even kernel size ({})'.format(kernel_size)
        padding = (kernel_size - 1) // 2
        if not isinstance(dilation, (tuple, list)):
            dilation = (dilation,)
        self.memory_efficient = memory_efficient
        self.nonlinear1 = get_nonlinear(config_str, in_channels)
        self.linear1 = nn.Linear(in_channels, bn_channels, bias=False)
        self.nonlinear2 = get_nonlinear(config_str, bn_channels)
        self.linear2 = nn.ModuleList()
        for _dilation in dilation:
            self.linear2.append(TimeDelay(bn_channels, out_channels, kernel_size, stride=stride,
                                          padding=padding * _dilation, dilation=_dilation, bias=bias))
        self.select = StatsSelect(out_channels, len(dilation), null=null, reduction=reduction)

    def forward(self, x):
        if self.training and self.memory_efficient:
            x = cp.checkpoint(self.bn_function, x)
        else:
            x = self.bn_function(x)
        x = self.nonlinear2(x)
        x = self.select([linear(x) for linear in self.linear2])
        return x


class MultiBranchDenseTDNNBlock(DenseTDNNBlock):

    def __init__(self, num_layers, in_channels, out_channels, bn_channels, kernel_size,
                 stride=1, dilation=1, bias=False, null=False, reduction=1,
                 config_str='batchnorm-relu', memory_efficient=False):
        super(DenseTDNNBlock, self).__init__()
        for i in range(num_layers):
            layer = MultiBranchDenseTDNNLayer(
                in_channels=in_channels + i * out_channels,
                out_channels=out_channels,
                bn_channels=bn_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                bias=bias,
                null=null,
                reduction=reduction,
                config_str=config_str,
                memory_efficient=memory_efficient
            )
            self.add_module('tdnnd%d' % (i + 1), layer)


class TransitLayer(nn.Module):

    def __init__(self, in_channels, out_channels, bias=True,
                 config_str='batchnorm-relu'):
        super(TransitLayer, self).__init__()
        self.nonlinear = get_nonlinear(config_str, in_channels)
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x):
        x = self.nonlinear(x)
        x = self.linear(x.transpose(1, 2)).transpose(1, 2)
        return x


class DenseLayer(nn.Module):

    def __init__(self, in_channels, out_channels, bias=True,
                 config_str='batchnorm-relu'):
        super(DenseLayer, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x):
        if len(x.shape) == 2:
            x = self.linear(x)
        else:
            x = self.linear(x.transpose(1, 2)).transpose(1, 2)
        x = self.nonlinear(x)
        return x



class DTDNNSS(TopVirtualNnet):

    def init(self, inputs_dim, num_targets, embd_dim=512, 
                 growth_rate=64, bn_size=2, init_channels=128, null=False, reduction=2,
                 config_str='batchnorm-relu', memory_efficient=True,
                 mixup=False, mixup_alpha=1.0,
                 margin_loss=False, margin_loss_params={},
                 use_step=False, step_params={},
                 training=True, extracted_embedding="far" ):

        default_margin_loss_params = {
            "method":"am", "m":0.2, 
            "feature_normalize":True, "s":30, 
            "double":False,
            "mhe_loss":False, "mhe_w":0.01,
            "inter_loss":0.,
            "ring_loss":0.,
            "curricular":False}

        default_step_params = {
            "T":None,
            "m":False, "lambda_0":0, "lambda_b":1000, "alpha":5, "gamma":1e-4,
            "s":False, "s_tuple":(30, 12), "s_list":None,
            "t":False, "t_tuple":(0.5, 1.2), 
            "p":False, "p_tuple":(0.5, 0.1)
        }
        
        self.use_step = use_step
        self.step_params = step_params

        self.extracted_embedding = extracted_embedding 

        margin_loss_params = utils.assign_params_dict(default_margin_loss_params, margin_loss_params)
        step_params = utils.assign_params_dict(default_step_params, step_params)
        self.mixup = Mixup(alpha=mixup_alpha) if mixup else None

        self.xvector = nn.Sequential(OrderedDict([
            ('tdnn', TDNNLayer(inputs_dim, init_channels, 5, dilation=1, padding=-1,
                               config_str=config_str)),
        ]))

        channels = init_channels
        for i, (num_layers, kernel_size, dilation) in enumerate(zip((6, 12), (3, 3), ((1, 3), (1, 3)))):
            block = MultiBranchDenseTDNNBlock(
                num_layers=num_layers,
                in_channels=channels,
                out_channels=growth_rate,
                bn_channels=bn_size * growth_rate,
                kernel_size=kernel_size,
                dilation=dilation,
                null=null,
                reduction=reduction,
                config_str=config_str,
                memory_efficient=memory_efficient
            )
            self.xvector.add_module('block%d' % (i + 1), block)
            channels = channels + num_layers * growth_rate
            self.xvector.add_module(
                'transit%d' % (i + 1), TransitLayer(channels, channels // 2, bias=False,
                                                    config_str=config_str))
            channels //= 2
        self.xvector.add_module('stats', StatsPool())
        self.xvector.add_module('dense', DenseLayer(channels * 2, embd_dim, config_str='batchnorm_'))

        if training :
            if margin_loss:
                self.loss = MarginSoftmaxLoss(512, num_targets, **margin_loss_params)
            else:
                self.loss = SoftmaxLoss(512, num_targets)

            self.wrapper_loss = MixupLoss(self.loss, self.mixup) if mixup else None

        for m in self.modules():
            if isinstance(m, TimeDelay):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    @utils.for_device_free
    def forward(self, x):
        x = self.xvector(x)
        # if self.training:
        #     x = self.classifier(x)
        return x.unsqueeze(dim=2)

    @utils.for_device_free
    def get_loss(self, inputs, targets):
        """Should call get_loss() after forward() with using Xvector model function.
        e.g.:
            m=Xvector(20,10)
            loss=m.get_loss(m(inputs),targets)

        model.get_loss [custom] -> loss.forward [custom]
          |
          v
        model.get_accuracy [custom] -> loss.get_accuracy [custom] -> loss.compute_accuracy [static] -> loss.predict [static]
        """
        if self.wrapper_loss is not None:
            return self.wrapper_loss(inputs, targets)
        else:
            return self.loss(inputs, targets)

    @utils.for_device_free
    def get_accuracy(self, targets):
        """Should call get_accuracy() after get_loss().
        @return: return accuracy
        """
        if self.wrapper_loss is not None:
            return self.wrapper_loss.get_accuracy(targets)
        else:
            return self.loss.get_accuracy(targets)

    @for_extract_embedding(maxChunk=10000, isMatrix=True)
    def extract_embedding(self, inputs):
        if self.extracted_embedding == "far":
            x = self.xvector(x)
            return x.unsqueeze(dim=2)
    
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
    
        

# if __name__ == '__main__':
#     # Input size: batch_size * seq_len * inputs_dim
#     x = torch.zeros(2, 80, 200)
#     model = DTDNNSS(80,512,1211)
#     out = model(x)
#     print(model)
#     print(out.shape)