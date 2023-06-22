import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
sys.path.insert(0, 'subtools/pytorch')
import libs.support.utils as utils
from libs.nnet import *



''' Res2Conv1d + BatchNorm1d + ReLU
'''
class SE_RES2Block(nn.Module):
    
    def __init__(self, channel,scale=8, k=3, d=2):
        super(SE_RES2Block, self).__init__()
        self.channel = channel
        self.scale = scale
        self.k = k
        self.d = d
        self.padding = int(self.d * (self.k - 1) / 2)
        self.res2net_hiddens = [int(self.channel / (2 ** i)) for i in range(self.scale)]
        # self.conv1d_expand = nn.Conv1d(self.channel, self.channel * self.scale, 1)
        # self.batchnorm_expand = nn.BatchNorm1d(self.channel * self.scale)

        self.res2conv1d_list = nn.ModuleList([nn.Conv1d(self.channel//scale, self.channel//scale, self.k, dilation=self.d, padding=self.padding) for i in range(self.scale-1)])
        self.res2batch_norm_list = nn.ModuleList([nn.BatchNorm1d(self.channel//scale) for i in range(self.scale-1)])

        # self.conv1d_collapse = nn.Conv1d(self.channel * self.scale, self.channel, 1)
        # self.batchnorm_collapse = nn.BatchNorm1d(self.channel)

        self.fc_1 = nn.Linear(self.channel, 128)
        self.fc_2 = nn.Linear(128, self.channel)

    def forward(self, input_tensor):

        # tensor = self.conv1d_expand(input_tensor)  # (B, C, T) -> (B, 8C, T)
        # tensor = F.relu(tensor)
        # tensor = self.batchnorm_expand(tensor)

        tensors = torch.split(input_tensor, input_tensor.size()[1]//self.scale, dim=1) # (B, 8C, T) -> (B, C, T)

        tensor_list = []

        assert len(tensors) == len(self.res2conv1d_list) + 1, f'{len(tensors)} != {len(self.res2conv1d_list)} + 1'

        for i, tensor in enumerate(tensors):
            if i > 1:
                tensor = tensor + last_tensor
            if i > 0:
                tensor = self.res2conv1d_list[i-1](tensor)
                tensor = F.relu(tensor)
                tensor = self.res2batch_norm_list[i-1](tensor)
            tensor_list.append(tensor)
            last_tensor = tensor 

        tensor = torch.cat(tensor_list, axis=1)

        # tensor = self.conv1d_collapse(tensor)  # (B, 8C, T) -> (B, C, T)
        # tensor = F.relu(tensor)
        # tensor = self.batchnorm_collapse(tensor)

        '''
        The dimension of the bottleneck in the SE-Block and the attention module is set to 128.
        '''

        z = torch.mean(tensor, dim=2)

        s = torch.sigmoid(self.fc_2(F.relu(self.fc_1(z)))) # (B, C)

        s = torch.unsqueeze(s, dim=2) # (B, C, 1)

        # s = torch.unsqueeze(s, dim=-1)

        se = s * tensor # (B, C, 1) * (B, C, T) = (B, C, T)

        # print(z.shape)
        # print(s.shape)

        # tensor = tensor + se # Gradient Explodes!!!
        
        # tensor = se + input_tensor 

        tensor = se

        # tensor += se 
        # RuntimeError: one of the variables needed for gradient computation 
        # has been modified by an inplace operation

        return tensor



# ''' Conv1d + BatchNorm1d + ReLU
# '''
# class Conv1dReluBn(nn.Module):
#     def __init__(self, inputs_dim, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
#         super().__init__()
#         self.conv = nn.Conv1d(inputs_dim, out_channels, kernel_size, stride, padding, dilation, bias=bias)
#         self.bn = nn.BatchNorm1d(out_channels)

#     def forward(self, x):
#         return self.bn(F.relu(self.conv(x)))



''' The SE connection of 1D case.
'''
# class SE_Connect(nn.Module):
#     def __init__(self, channels, s=4):
#         super().__init__()
#         assert channels % s == 0, "{} % {} != 0".format(channesl, s)
#         self.linear1 = nn.Linear(channels, channels // s)
#         self.linear2 = nn.Linear(channels // s, channels)

#     def forward(self, x):
#         out = x.mean(dim=2)
#         out = F.relu(self.linear1(out))
#         out = torch.sigmoid(self.linear2(out))
#         out = x * out.unsqueeze(2)
#         return out

# class SE_Connect(nn.Module):
#     def __init__(self, channels, bottleneck=128):
#         super(SE_Connect, self).__init__()
#         self.se = nn.Sequential(
#             nn.AdaptiveAvgPool1d(1),
#             nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
#             nn.ReLU(),
#             # nn.BatchNorm1d(bottleneck),
#             nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
#             nn.Sigmoid(),
#             )

#     def forward(self, input):
#         x = self.se(input)
#         return input * x


''' SE-Res2Block.
    Note: residual connection is implemented in the ECAPA_TDNN model, not here.
# '''
# def SE_Res2Block(channels, kernel_size, stride, padding, dilation, scale):
#     return nn.Sequential(
#         Conv1dReluBn(channels, channels, kernel_size=1, stride=1, padding=0),
#         Res2Conv1dReluBn(channels, kernel_size, stride, padding, dilation, scale=scale),
#         Conv1dReluBn(channels, channels, kernel_size=1, stride=1, padding=0),
#         SE_Connect(channels)
#     )



''' Attentive weighted mean and standard deviation pooling.
'''
class AttentiveStatPooling(nn.Module):
    
    def __init__(self, channel):
        super(AttentiveStatPooling, self).__init__()
        self.linear1 = nn.Linear(3 * channel, 128)
        self.linear2 = nn.Linear(128, 3 * channel)

    def forward(self, input_tensor, mask_tensor=None):
        h = input_tensor.transpose(1, 2) # (B, H, L) =>  (B, L, H) 
        tensor = F.relu(self.linear1(h)) # (B, L, 128) 
        e_tensor = self.linear2(tensor) # (B, L, 1536) 
        # a_tensor = F.softmax(e_tensor, dim=-1) # (B, L, 1536)
        a_tensor = F.softmax(e_tensor, dim=1) # (B, L, 1536)

        a_h_tensor = torch.mul(a_tensor, h) # (B, L, 1536)

        h_mean = torch.sum(a_h_tensor, dim=1, keepdim=False) # (B, H)
        h_mean_square = torch.mul(h_mean, h_mean) # (B, H)

        h_square = torch.mul(h, h) # (B, L, H)
        weighted_h_mean_square = torch.mul(a_tensor, h_square) # (B, L, H)

        weighted_square = torch.sum(weighted_h_mean_square, dim=1, keepdim=False) # (B, H)

        # neg_tensor = weighted_square - h_mean_square
        neg_tensor = (weighted_square - h_mean_square).clamp(min=1e-4)  
        
        if (neg_tensor < 0).any(): print("########## Negative value in Negative Tensor")
        sigma = torch.sqrt(neg_tensor) # (B, H) - (B, H)
        if torch.isnan(sigma).any(): 
            print("########## NaN in Sigma Tensor")
            print(torch.min(neg_tensor))

        tensor = torch.cat((h_mean, sigma), axis=1)

        return tensor, a_tensor



''' Implementation of
    "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification".

    Note that we DON'T concatenate the last frame-wise layer with non-weighted mean and standard deviation, 
    because it brings little improvment but significantly increases model parameters. 
    As a result, this implementation basically equals the A.2 of Table 2 in the paper.
'''
class XMU_TDNN(TopVirtualNnet):
    def init(self, inputs_dim, num_targets, channels=512, embd_dim=256, 
             aug_dropout=0., tail_dropout=0., chunk_dropout=False,chunk_dropout_params={}, training=True,
             extracted_embedding="near", mixup=False, mixup_alpha=1.0,
             pooling="ecpa-attentive", pooling_params={}, fc1=False, fc1_params={}, fc2_params={},
             margin_loss= True, margin_loss_params={}, use_step=False, step_params={}, adacos=False,Arc_adaptive=False,
             transfer_from="softmax_loss" ):
        

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

        pooling_params = utils.assign_params_dict(default_pooling_params, pooling_params)
        fc1_params = utils.assign_params_dict(default_fc_params, fc1_params)
        fc2_params = utils.assign_params_dict(default_fc_params, fc2_params)
        margin_loss_params = utils.assign_params_dict(default_margin_loss_params, margin_loss_params)
        step_params = utils.assign_params_dict(default_step_params, step_params)

        self.chunk_dropout = ChunkDropout(**chunk_dropout_params) if chunk_dropout else None
        self.mixup = Mixup(alpha=mixup_alpha) if mixup else None

        self.conv1d_in = nn.Conv1d(inputs_dim, channels, 5, padding=2)
        self.batchnorm_in = nn.BatchNorm1d(channels)

        self.se_res2block_1 = SE_RES2Block(channels,8, 3, 2)
        self.se_res2block_2 = SE_RES2Block(channels,8, 3, 3)
        self.se_res2block_3 = SE_RES2Block(channels,8, 3, 4)

        
        cat_channels = channels * 3
        self.conv1d_out = nn.Conv1d(cat_channels, cat_channels, 1)
        if pooling == "ecpa-attentive":
            self.stats = AttentiveStatPooling(channels)
        else:
            print("Using  StatisticsPooling!")
            self.stats = StatisticsPooling(cat_channels, stddev=stddev)

        # self.attentive_stat_pooling = AttentiveStatPooling()

        self.batchnorm_2 = nn.BatchNorm1d(channels * 3 * 2)

        self.fc = nn.Linear(channels * 3 * 2, embd_dim)

        self.batchnorm_3 = nn.BatchNorm1d(embd_dim)

        # self.speaker_embedding = nn.utils.weight_norm(nn.Linear(embd_dim, NUM_SPEAKERS, bias=False), dim=0)

        # self.scale = HYPER_RADIUS
        
        # self.m = torch.tensor(AMM_MARGIN, requires_grad=False).to(device)

        # self.num_speakers = NUM_SPEAKERS

        # self.device = device


        # Pooling
        # stddev = pooling_params.pop("stddev")
        # if pooling == "lde":
        #     self.stats = LDEPooling(cat_channels, c_num=pooling_params["num_head"])

        # elif pooling == "attentive":
        #     self.stats = AttentiveStatisticsPooling(cat_channels, hidden_size=pooling_params["hidden_size"], 
        #                                             context=pooling_params["context"], stddev=stddev)
        #     # self.bn_stats = nn.BatchNorm1d(cat_channels * 2)
        # elif pooling == "ecpa-attentive":
        #     self.stats = AttentiveStatsPool(cat_channels,128)
        #     self.bn_stats = nn.BatchNorm1d(cat_channels * 2)
        #     self.fc1 = ReluBatchNormTdnnLayer(cat_channels * 2, embd_dim, **fc1_params) if fc1 else None

        # elif pooling == "multi-head":
        #     self.stats = MultiHeadAttentionPooling(cat_channels, stddev=stddev, **pooling_params)
        #     self.bn_stats = nn.BatchNorm1d(cat_channels * 2)
        #     self.fc1 = ReluBatchNormTdnnLayer(cat_channels * 2, embd_dim, **fc1_params) if fc1 else None
        # elif pooling == "global-multi":
        #     self.stats = GlobalMultiHeadAttentionPooling(cat_channels,stddev=stddev, **pooling_params)
        #     self.bn_stats = nn.BatchNorm1d(cat_channels * 2* pooling_params["num_head"])
        #     self.fc1 = ReluBatchNormTdnnLayer(cat_channels * 2* pooling_params["num_head"], embd_dim, **fc1_params) if fc1 else None
        # elif pooling == "multi-resolution":
        #     self.stats = MultiResolutionMultiHeadAttentionPooling(cat_channels, **pooling_params)
        #     self.bn_stats = nn.BatchNorm1d(cat_channels * 2* pooling_params["num_head"])
        #     self.fc1 = ReluBatchNormTdnnLayer(cat_channels * 2* pooling_params["num_head"], embd_dim, **fc1_params) if fc1 else None

        # else:
        #     print("Using  StatisticsPooling!")
        #     self.stats = StatisticsPooling(cat_channels, stddev=stddev)
        #     self.bn_stats = nn.BatchNorm1d(cat_channels * 2)
        #     self.fc1 = ReluBatchNormTdnnLayer(cat_channels * 2, embd_dim, **fc1_params) if fc1 else None

        # self.tail_dropout = torch.nn.Dropout2d(p=tail_dropout) if tail_dropout > 0 else None

        # if fc1:
        #     fc2_in_dim = embd_dim
        # else:
        #     fc2_in_dim = cat_channels * 2
        # self.fc2 = ReluBatchNormTdnnLayer(fc2_in_dim, embd_dim, **fc2_params)
        # self.tail_dropout = torch.nn.Dropout2d(p=tail_dropout) if tail_dropout > 0 else None

         # Loss
        # Do not need when extracting embedding.
        if training :
            if margin_loss:
                self.loss = MarginSoftmaxLoss(embd_dim, num_targets, **margin_loss_params)
            # elif adacos:
            #     self.loss = AdaCos(embd_dim,num_targets)
            # # if sub_Arc_adaptive:
            # #     self.loss = sub_ArcFaceLossAdaptiveMargin(embd_dim,num_targets,0.2)
            # elif Arc_adaptive:
            #     self.loss = ArcFaceLossAdaptiveMargin(embd_dim,num_targets,0.2)
            else:
                self.loss = SoftmaxLoss(embd_dim, num_targets)
                # self.loss = AngleLoss(embd_dim,num_targets)
            self.wrapper_loss = MixupLoss(self.loss, self.mixup) if mixup else None
            # An example to using transform-learning without initializing loss.affine parameters
            self.transform_keys = ["conv1d_in","batchnorm_in","se_res2block_1","se_res2block_2","se_res2block_3","conv1d_out",
                                    "stats","batchnorm_2","fc","batchnorm_3"]

            if margin_loss and transfer_from == "softmax_loss":
                # For softmax_loss to am_softmax_loss
                self.rename_transform_keys = {"loss.affine.weight":"loss.weight"}

    @utils.for_device_free
    def forward(self, x,mask_tensor=None):
        tensor = self.conv1d_in(x) # (B, M, T) -> (B, C, T)
        tensor = F.relu(tensor)
        tensor = self.batchnorm_in(tensor)
        if torch.isnan(tensor).any(): print("NaN in first batch norm")

        tensor_1 = self.se_res2block_1(tensor)   # (B, C, T)
        tensor_1 = tensor_1 + tensor
        if torch.isnan(tensor_1).any(): print("NaN in tensor_1")
        tensor_2 = self.se_res2block_2(tensor_1) # (B, C, T)
        tensor_2 = tensor_2 + tensor_1 + tensor
        if torch.isnan(tensor_2).any(): print("NaN in tensor_2")
        tensor_3 = self.se_res2block_3(tensor_2) # (B, C, T)
        tensor_3 = tensor_3 + tensor_2 + tensor_1 + tensor
        if torch.isnan(tensor_3).any(): print("NaN in tensor_3")

        tensor = torch.cat([tensor_1, tensor_2, tensor_3], axis=1) # (B, 3C, T)
        # tensor = torch.cat([tensor_1, tensor_1, tensor_1], axis=1) # (B, 3C, T)
        tensor = self.conv1d_out(tensor) # (B, M, T) -> (B, C, T)
        tensor = F.relu(tensor)

        '''
        The dimension of the bottleneck in the SE-Block and the attention module is set to 128.
        '''

        tensor, a_tensor = self.stats(tensor, mask_tensor) # (B, H, L) =>  (B, H) 
        if torch.isnan(tensor).any(): print("NaN in stats")

        tensor = self.batchnorm_2(tensor)

        tensor = self.fc(tensor)

        tensor = self.batchnorm_3(tensor) # (B, H)

        tensor_g = torch.norm(tensor, dim=1, keepdim=True)
        normalized_tensor = tensor / tensor_g
        if torch.isnan(normalized_tensor).any(): print("NaN in normalization")
        return normalized_tensor.unsqueeze(dim=2)

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
    def extract_embedding(self, inputs,mask_tensor=None):
        tensor = self.conv1d_in(inputs) # (B, M, T) -> (B, C, T)
        tensor = F.relu(tensor)
        tensor = self.batchnorm_in(tensor)
        tensor_1 = self.se_res2block_1(tensor)   # (B, C, T)
        tensor_1 = tensor_1 + tensor
        tensor_2 = self.se_res2block_2(tensor_1) # (B, C, T)
        tensor_2 = tensor_2 + tensor_1 + tensor
        tensor_3 = self.se_res2block_3(tensor_2) # (B, C, T)
        tensor_3 = tensor_3 + tensor_2 + tensor_1 + tensor
        tensor = torch.cat([tensor_1, tensor_2, tensor_3], axis=1) # (B, 3C, T)
        tensor = self.conv1d_out(tensor) # (B, M, T) -> (B, C, T)
        tensor = F.relu(tensor)
        tensor, a_tensor = self.stats(tensor, mask_tensor) # (B, H, L) =>  (B, H) 
        tensor = self.batchnorm_2(tensor)

        if len(x.shape) !=3:
            x = x.unsqueeze(dim=2)
        if self.extracted_embedding == "far":
            assert self.fc1 is not None
            xvector = self.fc1.affine(x)
            normalized_tensor = F.normalize(xvector)
        elif self.extracted_embedding == "near_affine":
            x = self.auto(self.fc1, x)
            xvector = self.fc2.affine(x)
            xvector = F.normalize(xvector)
        elif self.extracted_embedding == "near":
            tensor = self.fc(tensor)
            tensor = self.batchnorm_3(tensor) # (B, H)
            tensor_g = torch.norm(tensor, dim=1, keepdim=True)
            normalized_tensor = tensor / tensor_g
        else:
            raise TypeError("Expected far or near position, but got {}".format(self.extracted_embedding))

        return normalized_tensor.unsqueeze(dim=2)



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
    # Input size: batch_size * seq_len * feat_dim
    x = torch.zeros(2, 26, 200)
    model = XMU_TDNN(inputs_dim=26,num_targets=1211, channels=512, embd_dim=512)
    out = model(x)
    print(model)
    print(out.shape)    # should be [2, 192]
