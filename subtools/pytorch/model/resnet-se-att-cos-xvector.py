# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Snowdar 2020-02-28)

import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, "subtools/pytorch")

import libs.support.utils as utils
from libs.nnet import *



def conv3x3(in_planes, out_planes, Conv=nn.Conv2d, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return Conv(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, Conv=nn.Conv2d, stride=1):
    """1x1 convolution"""
    return Conv(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class SEBlock(torch.nn.Module):
    
    def __init__(self, input_dim, ratio=8, inplace=True):
        '''
        @ratio: a reduction ratio which allows us to vary the capacity and computational cost of the SE blocks 
        in the network.
        '''
        super(SEBlock, self).__init__()

        self.input_dim = input_dim

        self.fc_1 = TdnnAffine(input_dim, input_dim//ratio)
        self.relu = torch.nn.ReLU(inplace=inplace)
        self.fc_2 = TdnnAffine(input_dim//ratio, input_dim)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 3
        assert inputs.shape[1] == self.input_dim

        x = inputs.mean(dim=2, keepdim=True)
        x = self.relu(self.fc_1(x))
        scale = self.sigmoid(self.fc_2(x))

        return scale


class RFM(TopVirtualNnet):
    
    def init(self, input_dim, gradient_clipping_bounds=0.25, scale_factor=1.0):
        tdnn_layer_params={"nonlinearity":'relu', "nonlinearity_params":{"inplace":True},
                         "bn-relu":False, 
                         "bn":True, 
                         "bn_params":{"momentum":0.5, "affine":False, "track_running_stats":True}}
        tdnn_params={"nonlinearity":"", "bn":False}

        self.layer1 = ReluBatchNormTdnnLayer(input_dim,256,**tdnn_params)
        # self.layer2 = ReluBatchNormTdnnLayer(512,512,**tdnn_params)
        self._lambda = scale_factor
        self._clipping = gradient_clipping_bounds

    @utils.for_device_free
    def forward(self, inputs):
        # inputs = GradientReversalFunction.apply(inputs, self._lambda, self._clipping)
        x = self.layer1(inputs)
        # x = self.layer2(x)
        return x





class DAL_regularizer(TopVirtualNnet):
    
    def init(self, n_in):
        self.w_id = TdnnAffine(n_in, n_in, bias=True)
        self.w_noise = TdnnAffine(n_in, n_in, bias=True)
    
    def forward(self, features_noise, features_id):
        features_noise = self.w_noise(features_noise)
        features_id = self.w_id(features_id)
        normalized_id = F.normalize(features_id.squeeze(dim=2)-features_id.mean(dim=1), dim=1)
        normalized_noise = F.normalize(features_noise.squeeze(dim=2)-features_noise.mean(dim=1), dim=1)
        cosine = torch.sum(normalized_id * normalized_noise, dim=1).mean()
        # features_id = features_id.squeeze()
        # features_noise = features_noise.squeeze()
        # vs_id = self.w_id(features_id)
        # vs_noise = self.w_noise(features_noise)
        # rho = ((vs_noise - vs_noise.mean(dim=0)) * (vs_id - vs_id.mean(dim=0))).mean(dim=0).pow(2) \
        #         / ( (vs_noise.var(dim=0) + 1e-6) * (vs_id.var(dim=0) + 1e-6))
        return cosine.pow(2)



class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, Conv=nn.Conv2d, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, norm_layer_params={}, full_pre_activation=True):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.downsample = downsample
        self.stride = stride
        self.full_pre_activation = full_pre_activation

        if self.full_pre_activation:
            self._full_pre_activation(inplanes, planes, Conv, stride, norm_layer, norm_layer_params)
        else:
            self._original(inplanes, planes, Conv, stride, norm_layer, norm_layer_params)

    def _original(self, inplanes, planes, Conv, stride, norm_layer, norm_layer_params):
        self.conv1 = conv3x3(inplanes, planes, Conv, stride)
        self.bn1 = norm_layer(planes, **norm_layer_params)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, Conv)
        self.bn2 = norm_layer(planes, **norm_layer_params)
        self.relu2 = nn.ReLU(inplace=True)
        self.se = SEBlock_2D(planes,16)

    def _full_pre_activation(self, inplanes, planes, Conv, stride, norm_layer, norm_layer_params):
        self.bn1 = norm_layer(inplanes, **norm_layer_params)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, Conv, stride)
        self.bn2 = norm_layer(planes, **norm_layer_params)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, Conv)
        self.se = SEBlock_2D(planes,16)

    def _original_forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu2(out)

        return out

    def _full_pre_activation_forward(self, x):
        """Reference: He, K., Zhang, X., Ren, S., & Sun, J. (2016). Identity mappings in deep residual 
                      networks. Paper presented at the European conference on computer vision.
        """
        identity = x

        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        out = self.se(out)
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity

        return out

    def forward(self, x):
        if self.full_pre_activation:
            return self._full_pre_activation_forward(x)
        else:
            return self._original_forward(x)

class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, Conv=nn.Conv2d, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, norm_layer_params={}, full_pre_activation=False):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        
        self.downsample = downsample
        self.stride = stride
        self.full_pre_activation = full_pre_activation

        if self.full_pre_activation:
            self._full_pre_activation(inplanes, planes, Conv, groups, width, stride, norm_layer, norm_layer_params, dilation)
        else:
            self._original(inplanes, planes, Conv, groups, width, stride, norm_layer, norm_layer_params, dilation)

    def _original(self, inplanes, planes, Conv, groups, width, stride, norm_layer, norm_layer_params, dilation):
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width, Conv)
        self.bn1 = norm_layer(width, **norm_layer_params)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(width, width, Conv, stride, groups, dilation)
        self.bn2 = norm_layer(width, **norm_layer_params)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv1x1(width, planes * self.expansion, Conv)
        self.bn3 = norm_layer(planes * self.expansion, **norm_layer_params)
        self.relu3 = nn.ReLU(inplace=True)

    def _full_pre_activation(self, inplanes, planes, Conv, groups, width, stride, norm_layer, norm_layer_params, dilation):
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.bn1 = norm_layer(inplanes, **norm_layer_params)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv1x1(inplanes, width, Conv)
        self.bn2 = norm_layer(width, **norm_layer_params)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(width, width, Conv, stride, groups, dilation)
        self.bn3 = norm_layer(width, **norm_layer_params)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = conv1x1(width, planes * self.expansion, Conv)

    def _original_forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)

        return out

    def _full_pre_activation_forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu3(out)
        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        return out

    def forward(self, x):
        if self.full_pre_activation:
            return self._full_pre_activation_forward(x)
        else:
            return self._original_forward(x)



class ResNet_SE(nn.Module):
    """Just return a structure (preconv + resnet) without avgpool and final linear.
    """
    def __init__(self, head_inplanes, block="BasicBlock", layers=[3, 4, 6, 3], planes=[32, 64, 128, 256], convXd=2, 
                 full_pre_activation=True,
                 head_conv=True, head_conv_params={"kernel_size":3, "stride":1, "padding":1},
                 head_maxpool=True, head_maxpool_params={"kernel_size":3, "stride":1, "padding":1},
                 zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, norm_layer_params={}):
        super(ResNet_SE, self).__init__()

        if convXd != 1 and convXd != 2:
            raise TypeError("Expected 1d or 2d conv, but got {}.".format(convXd))

        if norm_layer is None:
            if convXd == 2:
                norm_layer = nn.BatchNorm2d
            else:
                norm_layer = nn.BatchNorm1d

        self._norm_layer = norm_layer

        self.inplanes = planes[0]
        if not head_conv and self.in_planes != head_inplanes:
            raise ValueError("The inplanes is not equal to resnet first block" \
                             "inplanes without head conv({} vs. {}).".format(head_inplanes, self.inplanes))
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        if block == "BasicBlock":
            used_block = BasicBlock
        elif block == "Bottleneck":
            used_block = Bottleneck
        else:
            raise TypeError("Do not support {} block.".format(block))

        res_se_block="RES_SE_Block"
        self.groups = groups
        self.base_width = width_per_group
        self.head_conv = head_conv
        self.head_maxpool = head_maxpool

        self.downsample_multiple = 1
        self.full_pre_activation = full_pre_activation
        self.norm_layer_params = norm_layer_params

        self.Conv = nn.Conv2d if convXd == 2 else nn.Conv1d

        if self.head_conv:
            # Keep conv1.outplanes == layer1.inplanes
            self.conv1 = self.Conv(head_inplanes, self.inplanes, **head_conv_params, bias=False)
            self.bn1 = norm_layer(self.inplanes, **norm_layer_params)
            self.relu = nn.ReLU(inplace=True)
            self.downsample_multiple *= head_conv_params["stride"]

        if self.head_maxpool:
            Maxpool = nn.MaxPool2d if convXd == 2 else nn.MaxPool1d
            self.maxpool = Maxpool(**head_maxpool_params)
            self.downsample_multiple *= head_maxpool_params["stride"]

        self.layer1 = self._make_layer(used_block, planes[0], layers[0])
        self.layer2 = self._make_layer(used_block, planes[1], layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(used_block, planes[2], layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(used_block, planes[3], layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        self.downsample_multiple *= 8
        self.output_planes = planes[3] * used_block.expansion

        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.fc = nn.Linear(512 * block.expansion, num_classes)

        if "affine" in norm_layer_params.keys():
            norm_layer_affine = norm_layer_params["affine"]
        else:
            norm_layer_affine = True # torch.nn default it True

        for m in self.modules():
            if isinstance(m, self.Conv):
                torch.nn.init.normal_(m.weight, 0., 0.01)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)) and norm_layer_affine:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual and norm_layer_affine:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def get_downsample_multiple(self):
        return self.downsample_multiple

    def get_output_planes(self):
        return self.output_planes

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, self.Conv, stride),
                norm_layer(planes * block.expansion, **self.norm_layer_params),
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.Conv, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, 
                            norm_layer_params=self.norm_layer_params,
                            full_pre_activation=self.full_pre_activation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.Conv, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, norm_layer_params=self.norm_layer_params,
                                full_pre_activation=self.full_pre_activation))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        if self.head_conv:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        
        if self.head_maxpool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def forward(self, x): 
        return self._forward_impl(x)


class ResNetXvector(TopVirtualNnet):
    """ A resnet x-vector framework """
    
    def init(self, inputs_dim, num_targets, aug_dropout=0., tail_dropout=0., training=True, extracted_embedding="near", 
             resnet_params={}, pooling="statistics", pooling_params={}, fc1=False, fc1_params={}, fc2_params={}, margin_loss=False, margin_loss_params={},
             use_step=False, step_params={}, adacos=False,transfer_from="softmax_loss"):

        ## Params.
        default_resnet_params = {
            "head_conv":True, "head_conv_params":{"kernel_size":3, "stride":1, "padding":1},
            "head_maxpool":False, "head_maxpool_params":{"kernel_size":3, "stride":1, "padding":1},
            "block":"BasicBlock",
            "layers":[3, 4, 6, 3],
            "planes":[32, 64, 128, 256], # a.k.a channels.
            "convXd":2,
            "norm_layer_params":{"momentum":0.5, "affine":True},
            "full_pre_activation":True,
            "zero_init_residual":False
            }

        default_pooling_params = {
            "num_head":1,
            "hidden_size":64,
            "share":True,
            "affine_layers":1,
            "context":[0],
            "stddev":True,
            "temperature":False, 
            "fixed":True
        }
        
        default_fc_params = {
            "nonlinearity":'relu', "nonlinearity_params":{"inplace":True},
            "bn-relu":False, 
            "bn":True, 
            "bn_params":{"momentum":0.5, "affine":True, "track_running_stats":True}
            }

        default_margin_loss_params = {
            "method":"am", "m":0.2, "feature_normalize":True, 
            "s":30, "mhe_loss":False, "mhe_w":0.01
            }
        
        default_step_params = {
            "T":None,
            "m":False, "lambda_0":0, "lambda_b":1000, "alpha":5, "gamma":1e-4,
            "s":False, "s_tuple":(30, 12), "s_list":None,
            "t":False, "t_tuple":(0.5, 1.2), 
            "p":False, "p_tuple":(0.5, 0.1)
            }

        resnet_params = utils.assign_params_dict(default_resnet_params, resnet_params)
        pooling_params = utils.assign_params_dict(default_pooling_params, pooling_params)
        fc1_params = utils.assign_params_dict(default_fc_params, fc1_params)
        fc2_params = utils.assign_params_dict(default_fc_params, fc2_params)
        margin_loss_params = utils.assign_params_dict(default_margin_loss_params, margin_loss_params)
        step_params = utils.assign_params_dict(default_step_params, step_params)

        ## Var.
        self.extracted_embedding = extracted_embedding # only near here.
        self.use_step = use_step
        self.step_params = step_params
        self.convXd = resnet_params["convXd"]
        
        ## Nnet.
        self.aug_dropout = torch.nn.Dropout2d(p=aug_dropout) if aug_dropout > 0 else None

        # [batch, 1, feats-dim, frames] for 2d and  [batch, feats-dim, frames] for 1d.
        # Should keep the channel/plane is always in 1-dim of tensor (index-0 based).
        inplanes = 1 if self.convXd == 2 else inputs_dim
        self.resnet = ResNet_SE(inplanes, **resnet_params)

        

        # It is just equal to Ceil function.
        resnet_output_dim = (inputs_dim + self.resnet.get_downsample_multiple() - 1) // self.resnet.get_downsample_multiple() \
                            * self.resnet.get_output_planes() if self.convXd == 2 else self.resnet.get_output_planes()


        # Pooling
        stddev = pooling_params.pop("stddev")
        if pooling == "lde":
            self.stats = LDEPooling(resnet_output_dim, c_num=pooling_params["num_head"])
        elif pooling == "attentive":
            self.stats = AttentiveStatisticsPooling(resnet_output_dim, hidden_size=pooling_params["hidden_size"], 
                                                    context=pooling_params["context"], stddev=stddev)
        elif pooling == "multi-head":
            self.stats = MultiHeadAttentionPooling(resnet_output_dim, stddev=stddev, **pooling_params)
        elif pooling == "multi-resolution":
            self.stats = MultiResolutionMultiHeadAttentionPooling(resnet_output_dim, **pooling_params)
        else:
            self.stats = StatisticsPooling(resnet_output_dim, stddev=stddev)

        self.fc1 = ReluBatchNormTdnnLayer(self.stats.get_output_dim(), resnet_params["planes"][3], **fc1_params) if fc1 else None

        if fc1:
            fc2_in_dim = resnet_params["planes"][3]
        else:
            fc2_in_dim = self.stats.get_output_dim()

        self.fc2 = ReluBatchNormTdnnLayer(fc2_in_dim, resnet_params["planes"][3], **fc2_params)

        self.RFM = RFM(resnet_params["planes"][3])
        self.DAL = DAL_regularizer(resnet_params["planes"][3])


        # self.RFM = RFM(512)
        # self.DAL = DAL_regularizer(512)
        self.att = SEBlock(resnet_params["planes"][3])

        self.tail_dropout = torch.nn.Dropout2d(p=tail_dropout) if tail_dropout > 0 else None

        ## Do not need when extracting embedding.
        if training :
            if margin_loss:
                self.loss = MarginSoftmaxLoss(resnet_params["planes"][3], num_targets, **margin_loss_params)
                self.loss2 = SoftmaxLoss(resnet_params["planes"][3], 2)
            elif adacos:
                self.loss = AdaCos(resnet_params["planes"][3],num_targets)
            else:
                self.loss = SoftmaxLoss(resnet_params["planes"][3], num_targets)

            # An example to using transform-learning without initializing loss.affine parameters
            self.transform_keys = ["resnet", "stats", "fc1", "fc2"]
            # self.transform_keys = ["resnet"]

            # if margin_loss and transfer_from == "softmax_loss":
            #     # For softmax_loss to am_softmax_loss
            #     self.rename_transform_keys = {"loss.affine.weight":"loss.weight"} 

    @utils.for_device_free
    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        x = inputs
        x = self.auto(self.aug_dropout, x) # This auto function is equal to "x = layer(x) if layer is not None else x" for convenience.
        # [samples-index, frames-dim-index, frames-index] -> [samples-index, 1, frames-dim-index, frames-index]
        x = x.unsqueeze(1) if self.convXd == 2 else x
        # print("unsqueeze",x.shape)
        x = self.resnet(x)

        # print("resnet",x.shape)


        # [samples-index, channel, frames-dim-index, frames-index] -> [samples-index, channel*frames-dim-index, frames-index]
        x = x.reshape(x.shape[0], x.shape[1]*x.shape[2], x.shape[3]) if self.convXd == 2 else x
        x = self.stats(x) 
        x = self.auto(self.fc1, x)
        x = self.fc2(x)
        y = self.auto(self.RFM, x)
        # x = x+y
        scale = self.att(y)
        y = y*scale
        x = x*(1-scale)

        cc = self.DAL(y,x)
        x = self.auto(self.tail_dropout, x)

        return x,y,cc


    @utils.for_device_free
    def get_loss(self, inputs, targets,inputs2,targets2):
        """Should call get_loss() after forward() with using Xvector model function.
        e.g.:
            m=Xvector(20,10)
            loss=m.get_loss(m(inputs),targets)
        model.get_loss [custom] -> loss.forward [custom]
          |
          v
        model.get_accuracy [custom] -> loss.get_accuracy [custom] -> loss.compute_accuracy [static] -> loss.predict [static]
        """
        
        
        return self.loss(inputs, targets) + 0.1*self.loss2(inputs2,targets2)

    def get_posterior(self):
        """Should call get_posterior after get_loss. This function is to get outputs from loss component.
        @return: return posterior
        """
        return self.loss.get_posterior()

    @for_extract_embedding(maxChunk=10000, isMatrix=True)
    def extract_embedding(self, inputs):
        """
        inputs: a 3-dimensional tensor with batch-dim = 1 or normal features matrix
        return: an 1-dimensional vector after processed by decorator
        """

        x = inputs
        # Tensor shape is not modified in libs.nnet.resnet.py for calling free, such as using this framework in cv.
        x = x.unsqueeze(1) if self.convXd == 2 else x
        x = self.resnet(x)
        x = x.reshape(x.shape[0], x.shape[1]*x.shape[2], x.shape[3]) if self.convXd == 2 else x
        x = self.stats(x)

        if self.extracted_embedding == "far":
            assert self.fc1 is not None
            xvector = self.fc1.affine(x)
        elif self.extracted_embedding == "near_affine":
            x = self.auto(self.fc1, x)
            xvector = self.fc2.affine(x)
        elif self.extracted_embedding == "near":
            x = self.auto(self.fc1, x)
            xvector = self.fc2(x)
            y = self.auto(self.RFM, xvector)

            scale = self.att(y)
            # y = y*scale
            xvector = xvector*(1-scale)
            # xvector = xvector + y 
            # xvector = F.normalize(xvector)

        else:
            raise TypeError("Expected far or near position, but got {}".format(self.extracted_embedding))

        return xvector


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


# Test.
if __name__ == "__main__":
    # Let bach-size:128, fbank:40, frames:200.
    tensor = torch.randn(128, 40, 200)
    print("Test resnet2d ...")
    resnet2d = ResNetXvector(40, 1211, resnet_params={"convXd":2})
    print(resnet2d)
    # print(resnet2d(tensor).shape)
    # print("\n")
    # print("Test resnet1d ...")
    # resnet1d = ResNetXvector(40, 1211, resnet_params={"convXd":1})
    # print(resnet1d)
    # print(resnet1d(tensor).shape)

    print("Test done.")
