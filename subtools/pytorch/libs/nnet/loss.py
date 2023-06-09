# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Snowdar 2019-05-29)

import numpy as np
import math
import torch
import torch.nn.functional as F

from libs.support.utils import to_device
from .components import *


## TopVirtualLoss ✿
class TopVirtualLoss(torch.nn.Module):
    """ This is a virtual loss class to be suitable for pipline scripts, such as train.py. And it requires
    to implement the function get_posterior to compute accuracy. But just using self.posterior to record the outputs
    before computing loss in forward is more convenient.
    For example,
        def forward(self, inputs, targets):
            outputs = softmax(inputs)
            self.posterior = outputs
            loss = CrossEntropy(outputs, targets)
        return loss
    It means that get_posterior should be called after forward.
    """

    def __init__(self, *args, **kwargs):
        super(TopVirtualLoss, self).__init__()
        self.posterior = None
        self.init(*args, **kwargs)

    def init(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, *inputs):
        raise NotImplementedError

    def get_posterior(self):
        assert self.posterior is not None
        return self.posterior

    @utils.for_device_free
    def get_accuracy(self, targets):
        """
        @return: return accuracy
        """
        return self.compute_accuracy(self.get_posterior(), targets)

    @utils.for_device_free
    def predict(self, outputs):
        """
        @outputs: the outputs tensor with [batch-size,n,1] shape comes from affine before computing softmax or 
                  just softmax for n classes
        @return: an 1-dimensional vector including class-id (0-based) for prediction
        """
        with torch.no_grad():
            prediction = torch.squeeze(torch.argmax(outputs, dim=1))

        return prediction

    @utils.for_device_free
    def compute_accuracy(self, outputs, targets):
        """
        @outputs: the outputs tensor with [batch-size,n,1] shape comes from affine before computing softmax or 
                 just softmax for n classes
        @return: the float accuracy
        """
        assert outputs.shape[0] == len(targets)

        with torch.no_grad():
            prediction = self.predict(outputs)
            num_correct = (targets==prediction).sum()

        return num_correct.item()/len(targets)

#############################################

## Loss ✿
"""
Note, there are some principles about loss implements:
    In process: torch.nn.CrossEntropyLoss = softmax + log + torch.nn.NLLLoss()
    In function: torch.nn.NLLLoss() <-> - (sum(torch.tensor.gather())
so, in order to keep codes simple and efficient, do not using 'for' or any other complex grammar to implement what could be replaced by above.
"""

class SoftmaxLoss(TopVirtualLoss):
    """ An usual log-softmax loss with affine component.
    """
    def init(self, input_dim, num_targets, t=1, reduction='mean', special_init=False):
        self.affine = TdnnAffine(input_dim, num_targets)
        self.t = t # temperature
        # CrossEntropyLoss() has included the LogSoftmax, so do not add this function extra.
        self.loss_function = torch.nn.CrossEntropyLoss(reduction=reduction)

        # The special_init is not recommended in this loss component
        if special_init :
            torch.nn.init.xavier_uniform_(self.affine.weight, gain=torch.nn.init.calculate_gain('sigmoid'))

    def forward(self, inputs, targets):
        """Final outputs should be a (N, C) matrix and targets is a (1,N) matrix where there are 
        N targets-indexes (index value belongs to 0~9 when target-class C = 10) for N examples rather than 
        using one-hot format directly.
        One example, one target.
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 3
        assert inputs.shape[2] == 1

        posterior = self.affine(inputs)
        self.posterior = posterior.detach()

        # The frames-index is 1 now.
        outputs = torch.squeeze(posterior, dim=2)
        return self.loss_function(outputs/self.t, targets)

# class AngleLoss(TopVirtualLoss):
#     def init(self, input_dim, num_targets,init_w=10.0, init_b=-5.0, **kwargs):):
#         self.affine = TdnnAffine(input_dim, num_targets)
#         self.test_normalize = True
        
#         self.w = nn.Parameter(torch.tensor(init_w))
#         self.b = nn.Parameter(torch.tensor(init_b))
#         self.criterion  = torch.nn.CrossEntropyLoss()
#         print('Initialised AngleProto')
        
#     def forward(self, x, targets=None):
    
#         assert len(inputs.shape) == 3
#         assert inputs.shape[2] == 1

#         out_anchor      = torch.mean(x[:,1:,:],1)
#         out_positive    = x[:,0,:]
#         stepsize        = out_anchor.size()[0]

#         cos_sim_matrix  = F.cosine_similarity(out_positive.unsqueeze(-1),out_anchor.unsqueeze(-1).transpose(0,2))
#         torch.clamp(self.w, 1e-6)
#         cos_sim_matrix = cos_sim_matrix * self.w + self.b
        
#         targets       = torch.from_numpy(numpy.asarray(range(0,stepsize))).cuda()
#         nloss       = self.criterion(cos_sim_matrix, targets)
#         prec1, _    = accuracy(cos_sim_matrix.detach().cpu(), targets.detach().cpu(), topk=(1, 5))

#         return nloss, prec1



class FocalLoss(TopVirtualLoss):
    """Implement focal loss according to [Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. 
    "Focal loss for dense object detection", IEEE international conference on computer vision, 2017.]
    """
    def init(self, input_dim, num_targets, gamma=2, reduction='sum', eps=1.0e-10):

        self.softmax_affine = SoftmaxAffineLayer(input_dim, num_targets, dim=1, log=False, bias=True)
        self.loss_function = torch.nn.NLLLoss(reduction=reduction)

        self.gamma = gamma
        # self.alpha = alpha
        self.eps = eps

    def forward(self, inputs, targets):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 3
        assert inputs.shape[2] == 1

        posterior = self.softmax_affine(inputs)
        self.posterior = posterior.detach()

        focal_posterior = (1 - posterior)**self.gamma * torch.log(posterior.clamp(min=self.eps))
        outputs = torch.squeeze(focal_posterior, dim=2)
        return self.loss_function(outputs, targets)


class MarginSoftmaxLoss(TopVirtualLoss):
    """Margin softmax loss.
    There are AM, AAM, Double-AM, SM1 (Snowdar Margin softmax loss), SM2 and SM3. 
    Do not provide A-softmax loss again for its complex implementation and margin limitation.
    Reference:
            [1] Liu, W., Wen, Y., Yu, Z., & Yang, M. (2016). Large-margin softmax loss for convolutional neural networks. 
                Paper presented at the ICML. 

            [2] Liu, W., Wen, Y., Yu, Z., Li, M., Raj, B., & Song, L. (2017). Sphereface: Deep hypersphere embedding for 
                face recognition. Paper presented at the Proceedings of the IEEE conference on computer vision and pattern 
                recognition.  # a-softmax

            [3] Wang, F., Xiang, X., Cheng, J., & Yuille, A. L. (2017). Normface: l2 hypersphere embedding for face 
                verification. Paper presented at the Proceedings of the 25th ACM international conference on Multimedia.

            [4] Wang, F., Cheng, J., Liu, W., & Liu, H. (2018). Additive margin softmax for face verification. IEEE Signal 
                Processing Letters, 25(7), 926-930.  #am

            [5] Wang, H., Wang, Y., Zhou, Z., Ji, X., Gong, D., Zhou, J., . . . Liu, W. (2018). Cosface: Large margin cosine 
                loss for deep face recognition. Paper presented at the Proceedings of the IEEE Conference on Computer Vision 
                and Pattern Recognition.

            [6] Deng, J., Guo, J., Xue, N., & Zafeiriou, S. (2019). Arcface: Additive angular margin loss for deep face 
                recognition. Paper presented at the Proceedings of the IEEE Conference on Computer Vision and Pattern 
                Recognition. #aam

            [7] Zhou, S., Chen, C., Han, G., & Hou, X. (2020). Double Additive Margin Softmax Loss for Face Recognition. 
                Applied Sciences, 10(1), 60. 
    """
    def init(self, input_dim, num_targets,
             m=0.2, s=30., t=1.,
             feature_normalize=True,
             method="am",
             double=False,
             mhe_loss=False, mhe_w=0.01,
             inter_loss=0.,
             ring_loss=0.,
             curricular=False,
             reduction='mean', eps=1.0e-10, init=True,
             noise=False,total_iter=1000000,double_target=False,
             **args):

        self.input_dim = input_dim
        self.num_targets = num_targets
        self.weight = torch.nn.Parameter(torch.randn(num_targets, input_dim, 1))
        self.s = s # scale factor with feature normalization
        self.m = m # margin
        self.t = t # temperature
        # self.feature_normalize = feature_normalize
        self.method = method # am | aam | sm1 | sm2 | sm3
        self.double = double
        self.feature_normalize = feature_normalize
        self.mhe_loss = mhe_loss
        self.mhe_w = mhe_w
        self.inter_loss = inter_loss
        self.ring_loss = ring_loss
        self.lambda_factor = 0

        self.noise = noise
        self.postion = list(range(0,total_iter))
        self.total_iter=total_iter
        self.double_target=double_target
        # print(self.postion)

        self.curricular = CurricularMarginComponent() if curricular else None

        if self.ring_loss > 0:
            self.r = torch.nn.Parameter(torch.tensor(20.))
            self.feature_normalize = False

        self.eps = eps

        if feature_normalize :
            p_target = [0.9, 0.95, 0.99]
            suggested_s = [ (num_targets-1)/num_targets*np.log((num_targets-1)*x/(1-x)) for x in p_target ]

            if self.s < suggested_s[0]:
                print("Warning : using feature noamlization with small scalar s={s} could result in bad convergence. \
                There are some suggested s : {suggested_s} w.r.t p_target {p_target}.".format(
                s=self.s, suggested_s=suggested_s, p_target=p_target))

        self.loss_function = torch.nn.CrossEntropyLoss(reduction=reduction)

        # Init weight.
        if init:
             # torch.nn.init.xavier_normal_(self.weight, gain=1.0)
            torch.nn.init.normal_(self.weight, 0., 0.01) # It seems better.

    def forward(self, inputs, targets):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 3
        assert inputs.shape[2] == 1

        ## Normalize
        normalized_x = F.normalize(inputs.squeeze(dim=2), dim=1)
        normalized_weight = F.normalize(self.weight.squeeze(dim=2), dim=1)
        cosine_theta = F.linear(normalized_x, normalized_weight) # Y = x * W.t [batch * N]

    
        if not self.feature_normalize :
            self.s = inputs.norm(2, dim=1) # [batch-size, l2-norm]
            # print(self.s)
            # The accuracy must be reported before margin penalty added
            self.posterior = (self.s.detach() * cosine_theta.detach()).unsqueeze(2)
        if self.method == "ada":
            self.s = math.sqrt(2) * math.log(self.num_targets - 1)
            self.posterior = (self.s * cosine_theta.detach()).unsqueeze(2) 
        else:
            self.posterior = (self.s * cosine_theta.detach()).unsqueeze(2)

        

        if not self.training:
            # For valid set.
            outputs = self.s * cosine_theta
            return self.loss_function(outputs, targets)

        ## Margin Penalty
        # cosine_theta [batch_size, num_class]
        # targets.unsqueeze(1) [batch_size, 1]
        #################################
        if self.noise:
            current_postion=self.postion.pop(0)
            p = (1-current_postion/self.total_iter)  #t 当前迭代   T 总的迭代  ？ 怎么得到
            p = np.sqrt(p)
            x = np.random.binomial(1,p)
            targets1 = targets
            if x==1:
                pass
            else:
                targets = torch.argmax(cosine_theta,1)
         
            
        #################################
        cosine_theta_target = cosine_theta.gather(1, targets.unsqueeze(1)) #batch*1



        if self.inter_loss > 0:
            inter_cosine_theta = torch.softmax(self.s * cosine_theta, dim=1)
            inter_cosine_theta_target = inter_cosine_theta.gather(1, targets.unsqueeze(1))
            inter_loss = torch.log((inter_cosine_theta.sum(dim=1) - inter_cosine_theta_target)/(self.num_targets - 1) + self.eps).mean()

        if self.method == "am":
            penalty_cosine_theta = cosine_theta_target - self.m
            if self.double:
                double_cosine_theta = cosine_theta + self.m
        elif self.method == "aam":
            # Another implementation w.r.t cosine(theta+m) = cosine_theta * cos_m - sin_theta * sin_m
            # penalty_cosine_theta = self.cos_m * cosine_theta_target - self.sin_m * torch.sqrt((1-cosine_theta_target**2).clamp(min=0.))
            penalty_cosine_theta = torch.cos(torch.acos(cosine_theta_target) + self.m)
            if self.double:
                double_cosine_theta = torch.cos(torch.acos(cosine_theta).add(-self.m))

        elif self.method == "ada":
            # self.s = math.sqrt(2) * math.log(num_targets - 1)
            # penalty_cosine_theta = torch.cos(torch.acos(cosine_theta_target) + self.m)
            theta = torch.acos(torch.clamp(cosine_theta, -1.0 + 1e-7, 1.0 - 1e-7))
            one_hot = torch.zeros_like(cosine_theta)
            one_hot.scatter_(1, targets.view(-1, 1).long(), 1)
            with torch.no_grad():
                B_avg = torch.where(one_hot < 1, torch.exp(self.s * cosine_theta), torch.zeros_like(cosine_theta))
                B_avg = torch.sum(B_avg) / inputs.size(0)
                theta_med = torch.median(theta[one_hot == 1])
                # print("torch.min(math.pi/4 * torch.ones_like(theta_med)",torch.min(math.pi/4 * torch.ones_like(theta_med)))
                # print("theta_med",theta_med)
                self.s = torch.log(B_avg) / torch.cos(torch.min(math.pi/4 * torch.ones_like(theta_med), theta_med))
            # penalty_cosine_theta = torch.cos(torch.acos(cosine_theta_target) + self.m)
            
            outputs = self.s * cosine_theta

        elif self.method == "sm1":
            # penalty_cosine_theta = cosine_theta_target - (1 - cosine_theta_target) * self.m
            penalty_cosine_theta = (1 + self.m) * cosine_theta_target - self.m
        elif self.method == "sm2":
            penalty_cosine_theta = cosine_theta_target - (1 - cosine_theta_target**2) * self.m
        elif self.method == "sm3":
            penalty_cosine_theta = cosine_theta_target - (1 - cosine_theta_target)**2 * self.m
        else:
            raise ValueError("Do not support this {0} margin w.r.t [ am | aam | sm1 | sm2 | sm3 ]".format(self.method))

        if self.method != "ada":
            penalty_cosine_theta = 1 / (1 + self.lambda_factor) * penalty_cosine_theta + \
                               self.lambda_factor / (1 + self.lambda_factor) * cosine_theta_target

        if self.double:
            cosine_theta = 1/(1+self.lambda_factor) * double_cosine_theta + self.lambda_factor/(1+self.lambda_factor) * cosine_theta

        if self.curricular is not None:
            cosine_theta = self.curricular(cosine_theta, cosine_theta_target, penalty_cosine_theta)

        if self.method != "ada":
            outputs = self.s * cosine_theta.scatter(1, targets.unsqueeze(1), penalty_cosine_theta)

        ## Other extra loss
        # Final reported loss will be always higher than softmax loss for the absolute margin penalty and 
        # it is a lie about why we can not decrease the loss to a mininum value. We should not report the 
        # loss after margin penalty did but we really report this invalid loss to avoid computing the 
        # training loss twice.

        if self.ring_loss > 0:
            ring_loss = torch.mean((self.s - self.r)**2)/2
        else:
            ring_loss = 0.

        if self.mhe_loss:
            sub_weight = normalized_weight - torch.index_select(normalized_weight, 0, targets).unsqueeze(dim=1)
            # [N, C]
            normed_sub_weight = sub_weight.norm(2, dim=2)
            mask = torch.full_like(normed_sub_weight, True, dtype=torch.bool).scatter_(1, targets.unsqueeze(dim=1), False)
            # [N, C-1]
            normed_sub_weight_clean = torch.masked_select(normed_sub_weight, mask).reshape(targets.size()[0], -1)
            # torch.mean means 1/(N*(C-1))
            the_mhe_loss = self.mhe_w * torch.mean((normed_sub_weight_clean**2).clamp(min=self.eps)**-1)

            return self.loss_function(outputs/self.t, targets) + the_mhe_loss + self.ring_loss * ring_loss
        elif self.inter_loss > 0:
            return self.loss_function(outputs/self.t, targets) + self.inter_loss * inter_loss + self.ring_loss * ring_loss
        
        elif self.noise and self.double_target: # (outputs1,targets1) means ground truth label (ori), (outputs,targets) means predicte label 
            cosine_theta_target = cosine_theta.gather(1, targets1.unsqueeze(1))
            penalty_cosine_theta = cosine_theta_target - self.m
            outputs1 = self.s * cosine_theta.scatter(1, targets1.unsqueeze(1), penalty_cosine_theta)

            return 0.5 * self.loss_function(outputs1/self.t, targets1) + 0.5*self.loss_function(outputs/self.t, targets) + self.ring_loss * ring_loss

        else:
            return self.loss_function(outputs/self.t, targets) + self.ring_loss * ring_loss
    
    def step(self, lambda_factor):
        self.lambda_factor = lambda_factor

    def extra_repr(self):
        return '(~affine): (input_dim={input_dim}, num_targets={num_targets}, method={method}, double={double}, ' \
               'margin={m}, s={s}, t={t}, feature_normalize={feature_normalize}, mhe_loss={mhe_loss}, mhe_w={mhe_w}, ' \
               'eps={eps})'.format(**self.__dict__)


class CurricularMarginComponent(torch.nn.Module):
    """CurricularFace is implemented as a called component for MarginSoftmaxLoss.
    Reference: Huang, Yuge, Yuhan Wang, Ying Tai, Xiaoming Liu, Pengcheng Shen, Shaoxin Li, Jilin Li, 
               and Feiyue Huang. 2020. “CurricularFace: Adaptive Curriculum Learning Loss for Deep Face 
               Recognition.” ArXiv E-Prints arXiv:2004.00288.
    Github: https://github.com/HuangYG123/CurricularFace. Note, the momentum of this github is a wrong value w.r.t
            the above paper. The momentum 't' should not increase so fast and I have corrected it as follow.

    By the way, it does not work in my experiments.
    """
    def __init__(self, momentum=0.01):
        super(CurricularMarginComponent, self).__init__()
        self.momentum = momentum
        self.register_buffer('t', torch.zeros(1))

    def forward(self, cosine_theta, cosine_theta_target, penalty_cosine_theta):
        with torch.no_grad():
            self.t = (1 - self.momentum) * cosine_theta_target.mean() + self.momentum * self.t

        mask = cosine_theta > penalty_cosine_theta
        hard_example = cosine_theta[mask]
        # Use clone to avoid problem "RuntimeError: one of the variables needed for gradient computation 
        # has been modified by an inplace operation"
        cosine_theta_clone = cosine_theta.clone()
        cosine_theta_clone[mask] = hard_example * (self.t + hard_example)

        return cosine_theta_clone


class LogisticAffinityLoss(TopVirtualLoss):
    """LogisticAffinityLoss.
    Reference: Peng, J., Gu, R., & Zou, Y. (2019). 
               LOGISTIC SIMILARITY METRIC LEARNING VIA AFFINITY MATRIX FOR TEXT-INDEPENDENT SPEAKER VERIFICATION. 
    """
    def init(self, init_w=5., init_b=-1., reduction='mean'):
        self.reduction = reduction

        self.w = torch.nn.Parameter(torch.tensor(init_w))
        self.b = torch.nn.Parameter(torch.tensor(init_b))

    def forward(self, inputs, targets):
        # This loss has no way to compute accuracy
        S = F.normalize(inputs.squeeze(dim=2), dim=1)
        A = torch.sigmoid(self.w * torch.mm(S, S.t()) + self.b) # This can not keep the diag-value equal to 1 and it maybe a question.

        targets_matrix = targets + torch.zeros_like(A)
        condition = targets_matrix - targets_matrix.t()
        outputs = -torch.log(torch.where(condition==0, A, 1-A))

        if self.reduction == 'sum':
            return outputs.sum()
        elif self.reduction == 'mean':
            return outputs.sum() / targets.shape[0]
        else:
            raise ValueError("Do not support this reduction {0}".format(self.reduction))


class MixupLoss(TopVirtualLoss):
    """Implement a mixup component to augment data and increase the generalization of model training.
    Reference: 
        [1] Zhang, Hongyi, Moustapha Cisse, Yann N. Dauphin, and David Lopez-Paz. n.d. Mixup: BEYOND EMPIRICAL RISK MINIMIZATION.
        [2] Zhu, Yingke, Tom Ko, and Brian Mak. 2019. “Mixup Learning Strategies for Text-Independent Speaker Verification.”

    Github: https://github.com/hongyi-zhang/mixup/blob/master/cifar/utils.py
    """
    def init(self, base_loss, mixup_component):
        
        self.base_loss = base_loss
        self.mixup_component = mixup_component

    def forward(self, inputs, targets):
        if self.training:
            lam = self.mixup_component.lam
            index = self.mixup_component.index

            loss = lam * self.base_loss(inputs, targets) + \
                (1 - lam) * self.base_loss(inputs, targets[index])
        else:
            loss = self.base_loss(inputs, targets)

        return loss

    def get_accuracy(self, targets):
        if self.training:
            # It is not very clear to compute accuracy for mixed data.
            lam = self.mixup_component.lam
            index = self.mixup_component.index
            return lam * self.compute_accuracy(self.base_loss.get_posterior(), targets) + \
                   (1 - lam) * self.compute_accuracy(self.base_loss.get_posterior(), targets[index])
        else:
            return self.compute_accuracy(self.base_loss.get_posterior(), targets)

class AdaCos(TopVirtualLoss):
    def init(self, input_dim, num_targets, m=0.50,reduction='mean',init=True):
        self.input_dim = input_dim
        self.n_classes = num_targets
        self.s = math.sqrt(2) * math.log(num_targets - 1)
        self.m = m
        self.W = torch.nn.Parameter(torch.FloatTensor(num_targets, input_dim,1))
        
        self.loss_function = torch.nn.CrossEntropyLoss(reduction=reduction)
        if init:
            #  torch.nn.init.xavier_normal_(self.W, gain=1.0)
            torch.nn.init.normal_(self.W, 0., 0.01) # It seems better.
            # torch.nn.init.xavier_uniform_(self.W)

    def forward(self, input, label=None):
        # normalize features
        normalized_x = F.normalize(input.squeeze(dim=2), dim=1)
        # normalize weights
        normalized_weight = F.normalize(self.W.squeeze(dim=2))
        # dot product
        logits = F.linear(normalized_x, normalized_weight)
        self.posterior = (self.s * logits.detach()).unsqueeze(2)
        # if label is None:
        #     return logits
        # feature re-scale
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        with torch.no_grad():
            B_avg = torch.where(one_hot < 1, torch.exp(self.s * logits), torch.zeros_like(logits))
            B_avg = torch.sum(B_avg) / input.size(0)

            theta_med = torch.median(theta[one_hot == 1])
            self.s = torch.log(B_avg) / torch.cos(torch.min(math.pi/4 * torch.ones_like(theta_med), theta_med))

        if not self.training:
            # For valid set.
            outputs = self.s * logits
            return self.loss_function(outputs, label)
            
        
        output = self.s * logits

        return self.loss_function(output,label)

class DenseCrossEntropy(torch.nn.Module):
    def forward(self, x, target):
        x = x.float()
        target = target.float()
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        loss = -logprobs * target
        loss = loss.sum(-1)
        return loss.mean()


class ArcMarginProduct_subcenter(torch.nn.Module):
    def __init__(self, in_features, out_features, k=3):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.FloatTensor(out_features*k, in_features))
        self.reset_parameters()
        self.k = k
        self.out_features = out_features
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        
    def forward(self, features):
        cosine_all = F.linear(F.normalize(features), F.normalize(self.weight))
        cosine_all = cosine_all.view(-1, self.out_features, self.k)
        cosine, _ = torch.max(cosine_all, dim=2)
        return cosine   


class sub_ArcFaceLossAdaptiveMargin(TopVirtualLoss):
    def init(self, in_features,out_features,margins,k=3, s=30.0):

        self.crit = DenseCrossEntropy()
        self.s = s
        self.margins = margins
        self.weight = torch.nn.Parameter(torch.FloatTensor(out_features*k, in_features))
        self.reset_parameters()
        self.k = k
        self.out_features = out_features
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        
            
    def forward(self, features, labels):
        
        ms =[]
        ms = self.margins[labels.cpu().numpy()]

        cos_m = torch.from_numpy(np.cos(ms)).float().cuda()
        sin_m = torch.from_numpy(np.sin(ms)).float().cuda()
        th = torch.from_numpy(np.cos(math.pi - ms)).float().cuda()
        mm = torch.from_numpy(np.sin(math.pi - ms) * ms).float().cuda()
        labels = F.one_hot(labels, self.out_features).float()

        
        logits = F.linear(F.normalize(features.squeeze(dim=2)), F.normalize(self.weight))
        logits = logits.view(-1, self.out_features, self.k)
        logits, _ = torch.max(logits, dim=2)
        self.posterior = (self.s * logits.detach()).unsqueeze(2)

        if not self.training:
            # For valid set.
            output = self.s * logits
            return self.crit(output, labels)

        #################label smoothing###########
        
        labels *= 0.9
        labels += (1 - 0.9) / self.out_features
        ########################################

        logits = logits.float()
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * cos_m.view(-1,1) - sine * sin_m.view(-1,1)
        phi = torch.where(cosine > th.view(-1,1), phi, cosine - mm.view(-1,1))
        output = (labels * phi) + ((1.0 - labels) * cosine)
        output *= self.s
        loss = self.crit(output, labels)
        return loss   



class ArcFaceLossAdaptiveMargin(TopVirtualLoss):
    def init(self, in_features,out_features,margins, s=30.0):

        self.crit = DenseCrossEntropy()
        self.s = s
        self.margins = margins
        self.weight = torch.nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.reset_parameters()
        self.out_features = out_features
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        
            
    def forward(self, features, labels):
        
        # ms =[]
        # ms = self.margins[labels.cpu().numpy()]
        ms = np.array([self.margins])

        cos_m = torch.from_numpy(np.cos(ms)).float().cuda()
        sin_m = torch.from_numpy(np.sin(ms)).float().cuda()
        th = torch.from_numpy(np.cos(math.pi - ms)).float().cuda()
        mm = torch.from_numpy(np.sin(math.pi - ms) * ms).float().cuda()
        labels = F.one_hot(labels, self.out_features).float()

        logits = F.linear(F.normalize(features.squeeze(dim=2)), F.normalize(self.weight))

        self.posterior = (self.s * logits.detach()).unsqueeze(2)

        if not self.training:
            # For valid set.
            output = self.s * logits
            return self.crit(output, labels)

        #################label smoothing###########
        labels *= 0.9
        labels += (1 - 0.9) / self.out_features
        ########################################

        logits = logits.float()
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * cos_m.view(-1,1) - sine * sin_m.view(-1,1)
        phi = torch.where(cosine > th.view(-1,1), phi, cosine - mm.view(-1,1))
        output = (labels * phi) + ((1.0 - labels) * cosine)
        output *= self.s
        loss = self.crit(output, labels)

        return loss   



class MixupTrainLoss(TopVirtualLoss):
    """ An usual log-softmax loss with affine component.
    """
    def init(self, input_dim, num_targets,
             m=0.2, s=30., t=1.,
             feature_normalize=True,
             method="am",
             double=False,
             mhe_loss=False, mhe_w=0.01,
             inter_loss=0.,
             ring_loss=0.,
             curricular=False,
             reduction='mean', eps=1.0e-10, init=True,
             noise=False,total_iter=1000000,double_target=False):

        self.input_dim = input_dim
        self.num_targets = num_targets
        self.weight = torch.nn.Parameter(torch.randn(num_targets, input_dim, 1))
        self.s = s # scale factor with feature normalization
        self.m = m # margin
        self.t = t # temperature
        self.feature_normalize = feature_normalize
        self.method = method # am | aam | sm1 | sm2 | sm3
        self.double = double
        self.feature_normalize = feature_normalize
        self.mhe_loss = mhe_loss
        self.mhe_w = mhe_w
        self.inter_loss = inter_loss
        self.ring_loss = ring_loss
        self.lambda_factor = 0

        self.noise = noise
        self.postion = list(range(0,total_iter))
        self.total_iter=total_iter
        self.double_target=double_target
        # print(self.postion)

        self.curricular = CurricularMarginComponent() if curricular else None

        if self.ring_loss > 0:
            self.r = torch.nn.Parameter(torch.tensor(20.))
            self.feature_normalize = False

        self.eps = eps

        if feature_normalize :
            p_target = [0.9, 0.95, 0.99]
            suggested_s = [ (num_targets-1)/num_targets*np.log((num_targets-1)*x/(1-x)) for x in p_target ]

            if self.s < suggested_s[0]:
                print("Warning : using feature noamlization with small scalar s={s} could result in bad convergence. \
                There are some suggested s : {suggested_s} w.r.t p_target {p_target}.".format(
                s=self.s, suggested_s=suggested_s, p_target=p_target))

        self.loss_function = torch.nn.CrossEntropyLoss(reduction=reduction)

        # Init weight.
        if init:
             # torch.nn.init.xavier_normal_(self.weight, gain=1.0)
            torch.nn.init.normal_(self.weight, 0., 0.01) # It seems better.

    def forward(self, inputs, targets1=None,pre1=None,targets2=None,pre2=None,lam=None,posterior=False):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 3
        assert inputs.shape[2] == 1

        ## Normalize
        normalized_x = F.normalize(inputs.squeeze(dim=2), dim=1)
        normalized_weight = F.normalize(self.weight.squeeze(dim=2), dim=1)
        cosine_theta = F.linear(normalized_x, normalized_weight) # Y = x * W.t [batch * N]

        if not self.feature_normalize :
            self.s = inputs.norm(2, dim=1) # [batch-size, l2-norm]
            # The accuracy must be reported before margin penalty added
            self.posterior = (self.s.detach() * cosine_theta.detach()).unsqueeze(2)

        else:
            self.posterior = (self.s * cosine_theta.detach()).unsqueeze(2)

        if posterior:
            return self.posterior


        if not self.training:
            # For valid set.
            outputs = self.s * cosine_theta
            return self.loss_function(outputs, targets1)

        ## Margin Penalty
        # cosine_theta [batch_size, num_class]
        # targets.unsqueeze(1) [batch_size, 1]
        #################################
        # if self.noise:
        #     current_postion=self.postion.pop(0)
        #     p = (1-current_postion/self.total_iter)  #t 当前迭代   T 总的迭代  ？ 怎么得到
        #     p = np.sqrt(p)
        #     x = np.random.binomial(1,p)
        #     targets1 = targets
        #     if x==1:
        #         pass
        #     else:
        #         targets = torch.argmax(cosine_theta,1)
        #################################

        cosine_theta_target1 = cosine_theta.gather(1, targets1.unsqueeze(1)) #batch*1
        if pre1 is not None:
            cosine_theta_pre1 = cosine_theta.gather(1, pre1.unsqueeze(1))
            cosine_theta_target2 = cosine_theta.gather(1, targets2.unsqueeze(1))
            cosine_theta_pre2 = cosine_theta.gather(1, pre2.unsqueeze(1))

        

        if self.method == "am":
            penalty_cosine_theta_target1 = cosine_theta_target1 - self.m
            if pre1 is not None:
                penalty_cosine_theta_pre1 = cosine_theta_pre1 - self.m
                penalty_cosine_theta_target2 = cosine_theta_target2 - self.m
                penalty_cosine_theta_pre2 = cosine_theta_pre2 - self.m


        penalty_cosine_theta_target1 = 1 / (1 + self.lambda_factor) * penalty_cosine_theta_target1 + \
                               self.lambda_factor / (1 + self.lambda_factor) * cosine_theta_target1
        if pre1 is not None:
            penalty_cosine_theta_pre1 =  1 / (1 + self.lambda_factor) * penalty_cosine_theta_pre1 + \
                                self.lambda_factor / (1 + self.lambda_factor) * cosine_theta_pre1

            penalty_cosine_theta_target2 = 1 / (1 + self.lambda_factor) * penalty_cosine_theta_target2 + \
                                self.lambda_factor / (1 + self.lambda_factor) * cosine_theta_target2
            
            penalty_cosine_theta_pre2 = 1 / (1 + self.lambda_factor) * penalty_cosine_theta_pre2 + \
                                self.lambda_factor / (1 + self.lambda_factor) * cosine_theta_pre2

        outputs = self.s * cosine_theta.scatter(1, targets1.unsqueeze(1), penalty_cosine_theta_target1)
        if pre1 is not None:
            outputs =  outputs.scatter(1, pre1.unsqueeze(1), penalty_cosine_theta_pre1)
            outputs =  outputs.scatter(1, targets2.unsqueeze(1), penalty_cosine_theta_target2)
            outputs =  outputs.scatter(1, pre2.unsqueeze(1), penalty_cosine_theta_pre2)
            # print(lam)
            return lam*(0.2*self.loss_function(outputs/self.t, targets1) + 0.8*self.loss_function(outputs/self.t, pre1)) + \
                    (1-lam)*(0.2*self.loss_function(outputs/self.t, targets2) + 0.8 * self.loss_function(outputs/self.t, pre2))
        else:
            return self.loss_function(outputs/self.t, targets1)
        
    def step(self, lambda_factor):
        self.lambda_factor = lambda_factor

    def extra_repr(self):
        return '(~affine): (input_dim={input_dim}, num_targets={num_targets}, method={method}, double={double}, ' \
               'margin={m}, s={s}, t={t}, feature_normalize={feature_normalize}, mhe_loss={mhe_loss}, mhe_w={mhe_w}, ' \
               'eps={eps})'.format(**self.__dict__)

    def get_posterior_mix(self,inputs):
        return self.affine(inputs)



class HardPredictionLoss(TopVirtualLoss):
    def init(self, input_dim, num_targets,
             m=0.2, s=30., t=1.,
             feature_normalize=True,
             method="am",
             double=False,
             mhe_loss=False, mhe_w=0.01,
             inter_loss=0.,
             ring_loss=0.,
             curricular=False,
             reduction='mean', eps=1.0e-10, init=True,
             noise=False,total_iter=1000000,double_target=False,reg_loss=False,sqrt=False,square=False,
             sub_group=False,only_sub=False,adaptive_alpha=False, reweight=False,margine_reweight=False):

        self.input_dim = input_dim
        self.num_targets = num_targets
        self.weight = torch.nn.Parameter(torch.randn(num_targets, input_dim, 1))
        self.s = s # scale factor with feature normalization
        self.m = m # margin
        self.t = t # temperature
        self.feature_normalize = feature_normalize
        self.method = method # am | aam | sm1 | sm2 | sm3
        self.double = double
        self.feature_normalize = feature_normalize
        self.mhe_loss = mhe_loss
        self.mhe_w = mhe_w
        self.inter_loss = inter_loss
        self.ring_loss = ring_loss
        self.lambda_factor = 0

        self.noise = noise
        self.postion = list(range(0,total_iter))
        self.total_iter=total_iter
        self.double_target=double_target
        self.reg_loss = reg_loss
        self.sqrt = sqrt
        self.square = square
        self.sub_group = sub_group
        self.only_sub=only_sub
        self.adaptive_alpha= adaptive_alpha
        self.reweight=reweight
        self.margine_reweight=margine_reweight

        self.curricular = CurricularMarginComponent() if curricular else None

        if self.ring_loss > 0:
            self.r = torch.nn.Parameter(torch.tensor(20.))
            self.feature_normalize = False

        self.eps = eps

        if feature_normalize :
            p_target = [0.9, 0.95, 0.99]
            suggested_s = [ (num_targets-1)/num_targets*np.log((num_targets-1)*x/(1-x)) for x in p_target ]

            if self.s < suggested_s[0]:
                print("Warning : using feature noamlization with small scalar s={s} could result in bad convergence. \
                There are some suggested s : {suggested_s} w.r.t p_target {p_target}.".format(
                s=self.s, suggested_s=suggested_s, p_target=p_target))

        self.loss_function = torch.nn.CrossEntropyLoss(reduction=reduction)

        if self.sub_group:
            self.K= 3 
            self.weight = torch.nn.Parameter(torch.randn(num_targets*self.K, input_dim, 1))



        # Init weight.
        if init:
             # torch.nn.init.xavier_normal_(self.weight, gain=1.0)
            torch.nn.init.normal_(self.weight, 0., 0.01) # It seems better.

    def forward(self, inputs, targets):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 3
        assert inputs.shape[2] == 1

        ## Normalize
        normalized_x = F.normalize(inputs.squeeze(dim=2), dim=1)
        normalized_weight = F.normalize(self.weight.squeeze(dim=2), dim=1)
        cosine_theta = F.linear(normalized_x, normalized_weight) # Y = x * W.t [batch * N]
        if self.sub_group:
            cosine_theta = cosine_theta.view(-1, self.num_targets, self.K)
            cosine_theta, _ = torch.max(cosine_theta, dim=2)
            # print("11111")
    
        if not self.feature_normalize :
            self.s = inputs.norm(2, dim=1) # [batch-size, l2-norm]
            # print(self.s)
            # The accuracy must be reported before margin penalty added
            self.posterior = (self.s.detach() * cosine_theta.detach()).unsqueeze(2)
        if self.method == "ada":
            self.s = math.sqrt(2) * math.log(self.num_targets - 1)
            self.posterior = (self.s * cosine_theta.detach()).unsqueeze(2) 
        else:
            self.posterior = (self.s * cosine_theta.detach()).unsqueeze(2)

        

        if not self.training:
            # For valid set.
            outputs = self.s * cosine_theta
            return self.loss_function(outputs, targets)

        ## Margin Penalty
        # cosine_theta [batch_size, num_class]
        # targets.unsqueeze(1) [batch_size, 1]
        pre = torch.argmax(cosine_theta,1)
        cosine_theta_target = cosine_theta.gather(1, targets.unsqueeze(1)) #batch*1
        cosine_theta_pre = cosine_theta.gather(1, pre.unsqueeze(1)) #batch*1



        if self.inter_loss > 0:
            inter_cosine_theta = torch.softmax(self.s * cosine_theta, dim=1)
            inter_cosine_theta_target = inter_cosine_theta.gather(1, targets.unsqueeze(1))
            inter_loss = torch.log((inter_cosine_theta.sum(dim=1) - inter_cosine_theta_target)/(self.num_targets - 1) + self.eps).mean()


        if self.method == "am":
            penalty_cosine_theta = cosine_theta_target - self.m
            penalty_cosine_theta_pre = cosine_theta_pre - self.m
            # print(self.reweight)

            if self.reweight:
                same=torch.eq(targets,pre)
                # print(same)
                value=(cosine_theta*same.unsqueeze(1) + 1) *0.1
                # print(value)
                cosine_theta = cosine_theta +value*same.unsqueeze(1)
            if self.margine_reweight:
                same=torch.eq(targets,pre)
                # print(same)
                penalty_cosine_theta=(penalty_cosine_theta*same.unsqueeze(1)) -0.1
                penalty_cosine_theta_pre=(penalty_cosine_theta_pre*same.unsqueeze(1)) -0.1
                # print(value)
                # cosine_theta = cosine_theta +value



            if self.double:
                double_cosine_theta = cosine_theta + self.m


        elif self.method == "aam":
            # Another implementation w.r.t cosine(theta+m) = cosine_theta * cos_m - sin_theta * sin_m
            # penalty_cosine_theta = self.cos_m * cosine_theta_target - self.sin_m * torch.sqrt((1-cosine_theta_target**2).clamp(min=0.))
            penalty_cosine_theta = torch.cos(torch.acos(cosine_theta_target) + self.m)
            if self.double:
                double_cosine_theta = torch.cos(torch.acos(cosine_theta).add(-self.m))
        elif self.method == "sm1":
            # penalty_cosine_theta = cosine_theta_target - (1 - cosine_theta_target) * self.m
            penalty_cosine_theta = (1 + self.m) * cosine_theta_target - self.m
        elif self.method == "sm2":
            penalty_cosine_theta = cosine_theta_target - (1 - cosine_theta_target**2) * self.m
        elif self.method == "sm3":
            penalty_cosine_theta = cosine_theta_target - (1 - cosine_theta_target)**2 * self.m
        else:
            raise ValueError("Do not support this {0} margin w.r.t [ am | aam | sm1 | sm2 | sm3 ]".format(self.method))

        
        penalty_cosine_theta = 1 / (1 + self.lambda_factor) * penalty_cosine_theta + \
                               self.lambda_factor / (1 + self.lambda_factor) * cosine_theta_target

        penalty_cosine_theta_pre = 1 / (1 + self.lambda_factor) * penalty_cosine_theta_pre + \
                               self.lambda_factor / (1 + self.lambda_factor) * cosine_theta_pre

        if self.double:
            cosine_theta = 1/(1+self.lambda_factor) * double_cosine_theta + self.lambda_factor/(1+self.lambda_factor) * cosine_theta

        if self.curricular is not None:
            cosine_theta = self.curricular(cosine_theta, cosine_theta_target, penalty_cosine_theta)

       
        outputs = self.s * cosine_theta.scatter(1, targets.unsqueeze(1), penalty_cosine_theta)
        outputs_pre = self.s * cosine_theta.scatter(1, pre.unsqueeze(1), penalty_cosine_theta_pre)
        ## Other extra loss
        # Final reported loss will be always higher than softmax loss for the absolute margin penalty and 
        # it is a lie about why we can not decrease the loss to a mininum value. We should not report the 
        # loss after margin penalty did but we really report this invalid loss to avoid computing the 
        # training loss twice.
        if self.adaptive_alpha:
            # theta = torch.acos(torch.clamp(cosine_theta, -1.0 + 1e-7, 1.0 - 1e-7))
            # theta_pre_position = torch.argmin(theta,1)
            # theta_pre=theta.gather(1, theta_pre_position.unsqueeze(1))
            # # print(theta-3.14159226/2)
            # alpha = theta_pre.div(math.pi/2).mean()
            # # print(alpha)
            alpha = 0.5
            # alpha=0.1
            # outputs = alpha * outputs
            # outputs_pre = (1-alpha) * outputs_pre





        else:
            if len(self.postion) != 0:
                current_postion=self.postion.pop(0)
                alpha = (1-current_postion/self.total_iter)  #t 当前迭代   T 总的迭代  ？ 怎么得到
                if self.sqrt:
                    alpha = np.sqrt(alpha)
                elif self.square:
                    alpha = np.square(alpha)
            else:
                alpha = 1e-1   #for safe
        # print(alpha)
        # print(alpha)
        if self.reg_loss:
            output_mean = F.softmax(cosine_theta, dim=1)
            tab_mean_class = torch.mean(output_mean,-2)
            # print(tab_mean_class.shape)

            loss_reg = self.reg_loss_class(tab_mean_class, self.num_targets)
            # print(loss_reg)
            return alpha * self.loss_function(outputs/self.t, targets) + (1-alpha) * self.loss_function(outputs_pre/self.t, pre) \
                    + 1.0 * loss_reg

        elif self.only_sub and self.sub_group:
            # print("1111111111111")
            return self.loss_function(outputs/self.t, targets)
            
        else:
            return alpha * self.loss_function(outputs/self.t, targets) + (1-alpha) * self.loss_function(outputs_pre/self.t, pre)
    
    def step(self, lambda_factor):
        self.lambda_factor = lambda_factor

    def extra_repr(self):
        return '(~affine): (input_dim={input_dim}, num_targets={num_targets}, method={method}, double={double}, ' \
               'margin={m}, s={s}, t={t}, feature_normalize={feature_normalize}, mhe_loss={mhe_loss}, mhe_w={mhe_w}, ' \
               'eps={eps})'.format(**self.__dict__)

    def reg_loss_class(self,mean_tab,num_classes=10):
        # loss = 0
        # mean_tab_inv = torch.reciprocal(mean_tab)
        loss = -(1./num_classes) * torch.sum(torch.log(num_classes * mean_tab))
        # print(loss)
        # for items in mean_tab:
        #     loss += (1./num_classes)*torch.log((1./num_classes)/items)
        return loss


class aDCFLoss(TopVirtualLoss):
    def init(self, input_dim, num_targets, init=True):

        self.input_dim = input_dim
        self.num_targets = num_targets
        self.weight = torch.nn.Parameter(torch.randn(num_targets, input_dim, 1))
        self.threshold =  torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.alpha = 30
        self.loss_function = torch.nn.CrossEntropyLoss(reduction='mean')
        self.sigmoid = torch.nn.Sigmoid()


        # Init weight.
        if init:
             # torch.nn.init.xavier_normal_(self.weight, gain=1.0)
            torch.nn.init.normal_(self.weight, 0., 0.01) # It seems better.

    def forward(self, inputs, targets):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 3
        assert inputs.shape[2] == 1
        ## Normalize
        normalized_x = F.normalize(inputs.squeeze(dim=2), dim=1)
        normalized_weight = F.normalize(self.weight.squeeze(dim=2), dim=1)
        cosine_theta = F.linear(normalized_x, normalized_weight) # Y = x * W.t [batch * N]
        self.posterior = cosine_theta.detach()
    
        

        if not self.training:
            # For valid set.
            outputs = cosine_theta
            return self.loss_function(outputs, targets)

        one_hot = torch.zeros(len(targets), self.num_targets).cuda()
        one_hot = one_hot.scatter_(1, targets.unsqueeze(dim=1), 1)
        # print(one_hot)
        loss = 0.5 * ((self.sigmoid(self.alpha * (self.threshold - cosine_theta)) * one_hot).sum()) / (one_hot.sum()) 
        + 0.5* ((self.sigmoid(self.alpha * (cosine_theta - self.threshold )) * (1 - one_hot)).sum()) / ((1 - one_hot).sum())
        

        return loss
    
    def step(self, lambda_factor):
        self.lambda_factor = lambda_factor

    def extra_repr(self):
        return '(~affine): (input_dim={input_dim}, num_targets={num_targets}, method={method}, double={double}, ' \
               'margin={m}, s={s}, t={t}, feature_normalize={feature_normalize}, mhe_loss={mhe_loss}, mhe_w={mhe_w}, ' \
               'eps={eps})'.format(**self.__dict__)
