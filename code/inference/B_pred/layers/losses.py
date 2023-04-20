# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    '''Multi-class Focal loss implementation'''
    def __init__(self, gamma=2, weight=None,ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index=ignore_index

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1-pt)**self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight,ignore_index=self.ignore_index)
        return loss


class LabelSmoothingCrossEntropyWeight(nn.Module):
    def __init__(self, epsilon: float = 0.1, 
                 reduction: str = 'mean',
                 samplesize_per_cls = None,
                 smoothing = False,
                 device = 'cuda'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
        if samplesize_per_cls is not None:
            beta = 0.999
            effective_num = 1.0 - np.power(beta, samplesize_per_cls)
            weights = (1.0 - beta) / np.array(effective_num)
            alpha = weights/np.sum(weights) * len(samplesize_per_cls)
            if smoothing:
                k = 1/len(alpha)*1
                alpha = [(x/sum(alpha) + k)/2 for x in alpha]
        else:
            alpha = None
        if isinstance(alpha,(float,int)): 
            alpha = torch.Tensor([alpha,1-alpha]).to(device)
        if isinstance(alpha, (list, np.ndarray, np.generic)):
            alpha = torch.Tensor(alpha).to(device)
        self.device = device
        self.weights = alpha
        
    def _linear_combination(self, x, y):
        return self.epsilon*x + (1-self.epsilon)*y

    def _reduce_loss(self, loss, reduction='mean'):
        if reduction == 'mean':
            return loss.mean()  
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss

    def forward(self, preds, target):
        
        def inner_work(preds, target):
            num_class = preds.size()[-1]
            log_preds = F.log_softmax(preds, dim=-1)
            if self.weights is not None:
                log_preds = self.weights*log_preds
            loss = self._reduce_loss(-log_preds.sum(dim=-1), self.reduction)
            nll = F.nll_loss(log_preds, target, reduction=self.reduction, weight=self.weights)
            return self._linear_combination(loss/num_class, nll)
        
        if isinstance(preds, list):
            loss = []
            for pred in preds:
                loss.append(inner_work(pred, target))
            return sum(loss)/len(loss)
        else:
            return inner_work(preds, target)
        
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean',ignore_index=-100):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction=='sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction=='mean':
                loss = loss.mean()
        return loss*self.eps/c + (1-self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction,
                                                           ignore_index=self.ignore_index)


class DiceLoss(nn.Module):
    """DiceLoss implemented from 'Dice Loss for Data-imbalanced NLP Tasks'
    Useful in dealing with unbalanced data
    """
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self,input, target):
        '''
        input: [N, C]
        target: [N, ]
        '''
        prob = torch.softmax(input, dim=1)
        prob = torch.gather(prob, dim=1, index=target.unsqueeze(1))
        dsc_i = 1 - ((1 - prob) * prob) / ((1 - prob) * prob + 1)
        dice_loss = dsc_i.mean()
        return dice_loss

class FocalLoss_v2(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss_v2, self).__init__()
        # alpha为一个变量
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        # inputs --> [batch_size, class_num]
        # targets --> [batch_size, ]
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=1)
        
        # class_mask --> [batch_size, class_num]
        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        
        # 根据target,class_mask的第1个维度上除了对应的tgt索引的位置为1，其余为0
        class_mask.scatter_(1, ids.data, 1.)
#         print(class_mask)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
            
        # 这里将根据tgt的值,从alpha中取出相应的值
        alpha = self.alpha[ids.data.view(-1)]
        
        # 根据mask对非对应tgt位置的输出置0
        probs = (P*class_mask).sum(1).view(-1,1)
        
        # log_p --> log(probs)
        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)
        
        # 为cv-4的目标检测的课件160页的focal-loss-y=1
        # -a * (1-probs)^gamma * log(probs)
        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
        #print('-----bacth_loss------')
        #print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

    

def focal_loss_ls(labels, logits, alpha, gamma):
    """Compute the focal loss between `logits` and the ground truth `labels`.
    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
    Args:
      labels: A float tensor of size [batch, num_classes].
      logits: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.
    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """
    BCLoss = F.binary_cross_entropy_with_logits(input = logits, target = labels,reduction = "none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 +
                                                                           torch.exp(-1.0 * logits)))

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss)

    focal_loss /= torch.sum(labels)
    return focal_loss

class LabelSM_Focal(nn.Module):
    def __init__(self, beta=0.9, gamma=2, epsilon=0.1):
        super(LabelSM_Focal, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon
        
    def forward(self,logits, labels,loss_type = 'focal'):
        """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
        Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
        where Loss is one of the standard losses used for Neural Networks.
        Args:
          labels: A int tensor of size [batch].
          logits: A float tensor of size [batch, no_of_classes].
          samples_per_cls: A python list of size [no_of_classes].
          no_of_classes: total number of classes. int
          loss_type: string. One of "sigmoid", "focal", "softmax".
          beta: float. Hyperparameter for Class balanced loss.
          gamma: float. Hyperparameter for Focal loss.
        Returns:
          cb_loss: A float tensor representing class balanced loss
        """
        # self.epsilon = 0.1 #labelsmooth
        beta = self.beta
        gamma = self.gamma

        no_of_classes = logits.shape[1]
        samples_per_cls = torch.Tensor([sum(labels == i) for i in range(logits.shape[1])])
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        samples_per_cls = samples_per_cls.to(device)
            

        effective_num = 1.0 - torch.pow(beta, samples_per_cls)
        weights = (1.0 - beta) / ((effective_num)+1e-8)
        # print(weights)
        weights = weights / torch.sum(weights) * no_of_classes
        labels =labels.reshape(-1,1)

        labels_one_hot  = torch.zeros(len(labels), no_of_classes).to(device).scatter_(1, labels, 1)

        weights = torch.tensor(weights.clone().detach()).float()
        if torch.cuda.is_available():
            weights = weights.cuda()
            labels_one_hot = torch.zeros(len(labels), no_of_classes).cuda().scatter_(1, labels, 1).cuda()

        labels_one_hot = (1 - self.epsilon) * labels_one_hot + self.epsilon / no_of_classes
        weights = weights.unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1,no_of_classes)

        cb_loss = focal_loss_ls(labels_one_hot, logits, weights, gamma)
#         cb_loss = focal_lossls_v2(logits, labels_one_hot, alpha, gamma)
        
        return cb_loss
