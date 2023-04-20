import random
import numpy as np
from sklearn.metrics import f1_score
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.optim import swa_utils
import warnings
import copy
warnings.filterwarnings("ignore")
import torch.nn as nn
import os
from os import path as osp
import torch.nn.functional as F
join = lambda x,y: osp.abspath(osp.join(x, y))

def setup_device(args):
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.n_gpu = torch.cuda.device_count()

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def make_cvids(index, y, cv_dir, fold_num=10, seed=42):
    X = index
    from sklearn.model_selection import StratifiedKFold
    kf = StratifiedKFold(n_splits=fold_num, random_state=seed, shuffle=True)
    for cv, (tr, va) in enumerate(kf.split(X, y)):
        with open(osp.join(cv_dir, f'cv_{cv}.npy'), 'wb') as f:
            np.save(f, X[tr])
            np.save(f, X[va])
            # print(np.sum(y[va]==32))
            
import datetime
def now(onlydate=False):
    n = datetime.datetime.now()
    if not onlydate:
        return '{}{:02d}{:02d}_{}-{}'.format(n.year, n.month, n.day, n.hour, n.minute)
    else:
        return '{}{:02d}{:02d}'.format(n.year, n.month, n.day)            

def bert_base_LLRD(model, no_decay=["bias", "LayerNorm.bias", "LayerNorm.weight"], init_lr=1e-5, 
                   layers=12, weight_decay=0.01, layer_decay_rate=0.8, reinit_n_layers=3):
    opt_parameters = []    # To be passed to the optimizer (only parameters of the layers you want to update).
    # According to AAAMLP book by A. Thakur, we generally do not use any decay 
    # for bias and LayerNorm.weight layers.
    reinited_lr = init_lr*5
    head_lr = init_lr*5
    lr = init_lr
    # # === Pooler ======================================================  
    opt_parameters = [{"params": [p for n,p in model.named_parameters() if ("pooler" in n) 
                        and any(nd in n for nd in no_decay)], "lr": head_lr, "weight_decay": 0.0},                    
                    {"params": [p for n,p in model.named_parameters() if ("pooler" in n)
                    and not any(nd in n for nd in no_decay)], "lr": head_lr, "weight_decay": weight_decay},
    ]
    # === 12 Hidden layers ==========================================================
    counter=0
    for layer in range(layers-1,-1,-1):        # layer 11 -> 0
        if counter < reinit_n_layers:
            params_0 = [p for n,p in model.named_parameters() if f"encoder.layer.{layer}." in n 
                        and any(nd in n for nd in no_decay)]
            params_1 = [p for n,p in model.named_parameters() if f"encoder.layer.{layer}." in n 
                        and not any(nd in n for nd in no_decay)]
            layer_params0 = {"params": params_0, "lr": reinited_lr, "weight_decay": 0.0}
            layer_params1 = {"params": params_1, "lr": reinited_lr, "weight_decay": weight_decay}
            counter+=1
        else:
            params_0 = [p for n,p in model.named_parameters() if f"encoder.layer.{layer}." in n 
                        and any(nd in n for nd in no_decay)]
            params_1 = [p for n,p in model.named_parameters() if f"encoder.layer.{layer}." in n 
                        and not any(nd in n for nd in no_decay)]
            layer_params0 = {"params": params_0, "lr": lr, "weight_decay": 0.0}
            layer_params1 = {"params": params_1, "lr": lr, "weight_decay": weight_decay}
        opt_parameters.append(layer_params0)
        opt_parameters.append(layer_params1)
        
        lr *= layer_decay_rate    
    # === Embeddings layer ==========================================================
    params_0 = [p for n,p in model.named_parameters() if "embeddings" in n 
                and any(nd in n for nd in no_decay)]
    params_1 = [p for n,p in model.named_parameters() if "embeddings" in n
                and not any(nd in n for nd in no_decay)]
    embed_params = {"params": params_0, "lr": lr, "weight_decay": 0.0} 
    opt_parameters.append(embed_params)
    embed_params = {"params": params_1, "lr": lr, "weight_decay": weight_decay} 
    opt_parameters.append(embed_params)        
    
    return opt_parameters

# 设置分层学习率
def build_optimizer(args, model, pretrained_mn='bert'):
    """Prepare optimizer and scheduler (linear warmup and decay)

    Args:
        pretrained_mn (str, optional): this is the layer name for the pretrained part of the model. Defaults to 'bert'.

    Returns:
        optimizer, scheduler
    """
    # 
    no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
    # large_lr = ['']
    optimizer_grouped_parameters = [
        {'params': [j for i, j in model.named_parameters() if (not pretrained_mn in i and not any(nd in i for nd in no_decay))],
         'lr': args.learning_rate, 'weight_decay': args.weight_decay},
        {'params': [j for i, j in model.named_parameters() if (not pretrained_mn in i and any(nd in i for nd in no_decay))],
         'lr': args.learning_rate, 'weight_decay': 0.0},
    #     {'params': [j for i, j in model.named_parameters() if (pretrained_mn in i and not any(nd in i for nd in no_decay))],
    #      'lr': args.bert_learning_rate, 'weight_decay': args.weight_decay},
    #     {'params': [j for i, j in model.named_parameters() if (pretrained_mn in i and any(nd in i for nd in no_decay))],
    #      'lr': args.bert_learning_rate, 'weight_decay': 0.0},
    ]
    
    bert_grouped_params = bert_base_LLRD(model, no_decay, init_lr=args.bert_learning_rate, weight_decay=args.weight_decay, 
                                         layer_decay_rate=args.bert_layer_lr_decay_rate, reinit_n_layers=args.reinit_n_layers)
    optimizer_grouped_parameters.extend(bert_grouped_params)
    # since all model's parameter has been grouped, the default lr in adam will not be assigned to any para
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon, correct_bias=True)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)
    return optimizer, scheduler

def eval_fcn(predictions, labels):
    # prediction and labels are all level class ids
    f1_macro = f1_score(labels, predictions, average='macro')#
    f1_micro = f1_score(labels, predictions, average='micro')#
    f1_weight = f1_score(labels, predictions, average='weighted')#
    eval_results = {'f1_macro':f1_macro,'f1_micro':f1_micro,'f1_weight':f1_weight}
    return eval_results


def consis_loss(p_t, p_s, temp, losstype):
    # the lower the temperature the sharper the distribution
    sharp_p_t = F.softmax(p_t/temp, dim=1)
    p_s = F.softmax(p_s,dim=1)
    if losstype == 'mse':
        return F.mse_loss(sharp_p_t, p_s, reduction='mean')
        # return torch.mean(torch.pow(p_s - sharp_p_t, 2))
    elif losstype == 'kl':
        log_sharp_p_t = torch.log(sharp_p_t+1e-8)
        return F.kl_div(log_sharp_p_t, p_s, reduction = 'mean')
        # return torch.mean(p_s * (torch.log(p_s+1e-8) - log_sharp_p_t))


class LabelSmoothingCrossEntropy(nn.Module):
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
    
from torch.autograd import Variable
class FocalLoss(nn.Module):
    """the modified version of focal loss adopted from: https://arxiv.org/pdf/1901.05555.pdf
    we use samplesize per class to estimate loss weights for focal
    """
    def __init__(self, gamma=0, samplesize_per_cls=None, 
                 size_average=True, device='cuda', smoothing=False):
        super().__init__()
        if samplesize_per_cls is not None:
            beta = 0.999
            effective_num = 1.0 - np.power(beta, samplesize_per_cls)
            weights = (1.0 - beta) / np.array(effective_num)
            alpha = weights/np.sum(weights) * len(samplesize_per_cls)
            if smoothing:
                _factor = 1
                k = 1/len(alpha)*_factor        # uniform smoothing
                alpha = [(x + k)/2 for x in alpha]
                alpha/=np.sum(alpha)
        else:
            alpha = None
        
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): 
            self.alpha = torch.Tensor([alpha,1-alpha]).to(device)
        if isinstance(alpha, (list, np.ndarray, np.generic)):
            self.alpha = torch.Tensor(alpha).to(device)
        self.size_average = size_average

    def forward(self, input, target):
        
        def inner_work(input, target):
            if input.dim()>2:
                input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
                input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
                input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
            target = target.view(-1,1)      # N,1

            logpt = F.log_softmax(input, dim=-1)    # N, C
            logpt = logpt.gather(1,target)          # only gather along C dim
            logpt = logpt.view(-1)
            pt = Variable(logpt.data.exp())

            if self.alpha is not None:
                if self.alpha.type()!=input.data.type():
                    self.alpha = self.alpha.type_as(input.data)
                at = self.alpha.gather(0, target.data.view(-1)) # alpha dim: C, so need to reduce target dim N,1 to N
                # logpt = logpt * Variable(at)
                logpt = logpt * at

            loss = -1 * (1-pt)**self.gamma * logpt
            if self.size_average: 
                return loss.mean()
            else: 
                return loss.sum()
        if isinstance(input, list):           
            loss = [inner_work(i, target) for i in input]
            loss = sum(loss)/len(loss)
        else:
            loss = inner_work(input, target)
        return loss

from torch import Tensor
from typing import Optional
class DiceLoss(nn.Module):
    """
    Dice coefficient for short, is an F1-oriented statistic used to gauge the similarity of two sets.
    Given two sets A and B, the vanilla dice coefficient between them is given as follows:
        Dice(A, B)  = 2 * True_Positive / (2 * True_Positive + False_Positive + False_Negative)
                    = 2 * |A and B| / (|A| + |B|)
    Math Function:
        U-NET: https://arxiv.org/abs/1505.04597.pdf
        dice_loss(p, y) = 1 - numerator / denominator
            numerator = 2 * \sum_{1}^{t} p_i * y_i + smooth
            denominator = \sum_{1}^{t} p_i + \sum_{1} ^{t} y_i + smooth
        if square_denominator is True, the denominator is \sum_{1}^{t} (p_i ** 2) + \sum_{1} ^{t} (y_i ** 2) + smooth
        V-NET: https://arxiv.org/abs/1606.04797.pdf
    Args:
        smooth (float, optional): a manual smooth value for numerator and denominator.
        square_denominator (bool, optional): [True, False], specifies whether to square the denominator in the loss function.
        with_logits (bool, optional): [True, False], specifies whether the input tensor is normalized by Sigmoid/Softmax funcs.
        ohem_ratio: max ratio of positive/negative, defautls to 0.0, which means no ohem.
        alpha: dsc alpha
    Shape:
        - input: (*)
        - target: (*)
        - mask: (*) 0,1 mask for the input sequence.
        - Output: Scalar loss
    Examples:
        >>> loss = DiceLoss(with_logits=True, ohem_ratio=0.1)
        >>> input = torch.FloatTensor([2, 1, 2, 2, 1])
        >>> input.requires_grad=True
        >>> target = torch.LongTensor([0, 1, 0, 0, 0])
        >>> output = loss(input, target)
        >>> output.backward()
    """
    def __init__(self,
                 smooth: Optional[float] = 1e-4,
                 square_denominator: Optional[bool] = False,
                 with_logits: Optional[bool] = True,
                 ohem_ratio: float = 0.0,
                 alpha: float = 0.0,
                 reduction: Optional[str] = "mean",
                 index_label_position=True) -> None:
        super(DiceLoss, self).__init__()

        self.reduction = reduction
        self.with_logits = with_logits
        self.smooth = smooth
        self.square_denominator = square_denominator
        self.ohem_ratio = ohem_ratio
        self.alpha = alpha
        self.index_label_position = index_label_position

    def forward(self, input: Tensor, target: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        logits_size = input.shape[-1]

        if logits_size != 1:
            loss = self._multiple_class(input, target, logits_size, mask=mask)
        else:
            loss = self._binary_class(input, target, mask=mask)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

    def _compute_dice_loss(self, flat_input, flat_target):
        flat_input = ((1 - flat_input) ** self.alpha) * flat_input
        interection = torch.sum(flat_input * flat_target, -1)
        if not self.square_denominator:
            loss = 1 - ((2 * interection + self.smooth) /
                        (flat_input.sum() + flat_target.sum() + self.smooth))
        else:
            loss = 1 - ((2 * interection + self.smooth) /
                        (torch.sum(torch.square(flat_input, ), -1) + torch.sum(torch.square(flat_target), -1) + self.smooth))

        return loss

    def _multiple_class(self, input, target, logits_size, mask=None):
        flat_input = input
        flat_target = F.one_hot(target, num_classes=logits_size).float() if self.index_label_position else target.float()
        flat_input = torch.nn.Softmax(dim=1)(flat_input) if self.with_logits else flat_input

        if mask is not None:
            mask = mask.float()
            flat_input = flat_input * mask
            flat_target = flat_target * mask
        else:
            mask = torch.ones_like(target)

        loss = None
        if self.ohem_ratio > 0 :
            mask_neg = torch.logical_not(mask)
            for label_idx in range(logits_size):
                pos_example = target == label_idx
                neg_example = target != label_idx

                pos_num = pos_example.sum()
                neg_num = mask.sum() - (pos_num - (mask_neg & pos_example).sum())
                keep_num = min(int(pos_num * self.ohem_ratio / logits_size), neg_num)

                if keep_num > 0:
                    neg_scores = torch.masked_select(flat_input, neg_example.view(-1, 1).bool()).view(-1, logits_size)
                    neg_scores_idx = neg_scores[:, label_idx]
                    neg_scores_sort, _ = torch.sort(neg_scores_idx, )
                    threshold = neg_scores_sort[-keep_num + 1]
                    cond = (torch.argmax(flat_input, dim=1) == label_idx & flat_input[:, label_idx] >= threshold) | pos_example.view(-1)
                    ohem_mask_idx = torch.where(cond, 1, 0)

                    flat_input_idx = flat_input[:, label_idx]
                    flat_target_idx = flat_target[:, label_idx]

                    flat_input_idx = flat_input_idx * ohem_mask_idx
                    flat_target_idx = flat_target_idx * ohem_mask_idx
                else:
                    flat_input_idx = flat_input[:, label_idx]
                    flat_target_idx = flat_target[:, label_idx]

                loss_idx = self._compute_dice_loss(flat_input_idx.view(-1, 1), flat_target_idx.view(-1, 1))
                if loss is None:
                    loss = loss_idx
                else:
                    loss += loss_idx
            return loss

        else:
            for label_idx in range(logits_size):
                pos_example = target == label_idx
                flat_input_idx = flat_input[:, label_idx]
                flat_target_idx = flat_target[:, label_idx]

                loss_idx = self._compute_dice_loss(flat_input_idx.view(-1, 1), flat_target_idx.view(-1, 1))
                if loss is None:
                    loss = loss_idx
                else:
                    loss += loss_idx
            return loss

    def _binary_class(self, input, target, mask=None):
        flat_input = input.view(-1)
        flat_target = target.view(-1).float()
        flat_input = torch.sigmoid(flat_input) if self.with_logits else flat_input

        if mask is not None:
            mask = mask.float()
            flat_input = flat_input * mask
            flat_target = flat_target * mask
        else:
            mask = torch.ones_like(target)

        if self.ohem_ratio > 0:
            pos_example = target > 0.5
            neg_example = target <= 0.5
            mask_neg_num = mask <= 0.5

            pos_num = pos_example.sum() - (pos_example & mask_neg_num).sum()
            neg_num = neg_example.sum()
            keep_num = min(int(pos_num * self.ohem_ratio), neg_num)

            neg_scores = torch.masked_select(flat_input, neg_example.bool())
            neg_scores_sort, _ = torch.sort(neg_scores, )
            threshold = neg_scores_sort[-keep_num+1]
            cond = (flat_input > threshold) | pos_example.view(-1)
            ohem_mask = torch.where(cond, 1, 0)
            flat_input = flat_input * ohem_mask
            flat_target = flat_target * ohem_mask

        return self._compute_dice_loss(flat_input, flat_target)

    def __str__(self):
        return f"Dice Loss smooth:{self.smooth}, ohem: {self.ohem_ratio}, alpha: {self.alpha}"

    def __repr__(self):
        return str(self)


class LossFcn:
    def __init__(self, sup_losstype='ce', device='cuda', temp=1, unsup_losstype='mse', 
                 sup_lam=1, unsup_lam=1, pseudo_lam=1, epsilon=0.1,
                 samplesize_per_cls=None, sup2_losstype='focal_weighted'):
        self.device = device
        lts = {'sup_loss_fcn': sup_losstype,
               'sup_loss_fcn2': sup2_losstype,
               }
        for k, v in lts.items():
            if v=='focal':
                lts[k] = FocalLoss(gamma=1, device=device, samplesize_per_cls=None)
            elif v=='focal_weighted':
                lts[k] = FocalLoss(gamma=1, device=device, samplesize_per_cls=samplesize_per_cls)
            elif v=='ce':
                lts[k] = nn.CrossEntropyLoss()
            elif v=='ls_ce_weighted':
                lts[k] = LabelSmoothingCrossEntropy(epsilon=epsilon, device=device, samplesize_per_cls=samplesize_per_cls)
            elif v=='ls_ce':
                lts[k] = LabelSmoothingCrossEntropy(epsilon=epsilon, device=device)
        
        for k,v in lts.items():
            setattr(self, k, v)
        
        self.temp = temp
        self.unsup_losstype = unsup_losstype
        self.sup_lam = sup_lam
        self.unsup_lam = unsup_lam
        self.pseudo_lam = pseudo_lam
        
    def run(self, model, batch, ema, pseudo_b, aug_b, method='also_sup_aug'):
        label = batch['label'].squeeze(dim=1).to(self.device)
        sup_loss, pl_loss, unsup_loss = torch.tensor(0), torch.tensor(0), torch.tensor(0)
        
        if aug_b is not None:
            l = len(batch['text_inputs'])
            batch_aug = {}
            for k in batch.keys():
                batch_aug[k] = torch.concat([batch[k], aug_b[k]], dim=0)
            pred_tot, p_tot, p_tots = model(batch_aug)
            pred = pred_tot[:l]     # recover res from unaugmented data
            p_s = p_tot[:l]
            p_ss = [x[:l] for x in p_tots]
            p_aug = p_tot[l:]
            label_ = batch_aug['label'].squeeze(dim=1).to(self.device)
            sup_loss = self.sup_loss_fcn(p_tots, label_)*self.sup_lam
        else:
            pred, p_s, p_ss = model(batch)
        # if method == 'only_sup_ori' or aug_b is None:
            sup_loss = self.sup_loss_fcn(p_ss, label)*self.sup_lam
        # elif method == 'also_sup_aug' and aug_b is not None:
            # this label is only used for loss calculation
        loss = sup_loss
        # pseudo label loss
        if pseudo_b is not None:
            _, p_s2, p_s2s = model(pseudo_b)
            pl = pseudo_b['label'].squeeze(dim=1).to(self.device)
            pl_loss = self.sup_loss_fcn2(p_s2s, pl)*self.pseudo_lam
            loss += pl_loss
        # regularization loss for time-averaged consistency and data-perturbed consistency
        loss_consis1, loss_consis2, counter = 0, 0, 0
        if aug_b is not None:
            # for consis1: p_s is the anchor dist. and p_aug is the one needs to be pulled
            # for consis2: p_t is the anchor dist. and p_s is the one needs to be pulled
            loss_consis1 = consis_loss(p_s, p_aug, self.temp, self.unsup_losstype)
            counter+=1
        if ema is not None and ema.calc_loss:
            with torch.no_grad():
                _, p_t, _ = ema.teacher_model(batch)
            loss_consis2 = consis_loss(p_t, p_s, self.temp, self.unsup_losstype)
            counter+=1
        unsup_loss = torch.tensor(self.unsup_lam*(loss_consis2+loss_consis1)/max(1,counter))
        loss += unsup_loss
        
        acc = (label == pred).float().sum() / label.shape[0]
        
        self.last_run_losses = {'tr_sup': sup_loss.item(), 'tr_sup2': pl_loss.item(), 'tr_unsup': unsup_loss.item()}
        return loss, acc, pred

class FGM:
    """fgm in NLP - this follows pseudo code in https://zhuanlan.zhihu.com/p/103593948
    对于每个x:
        1.计算x的前向loss、反向传播得到梯度
        2.根据embedding矩阵的梯度计算出r，并加到当前embedding上，相当于x+r
        3.计算x+r的前向loss，反向传播得到对抗的梯度，累加到(1)的梯度上
        4.将embedding恢复为(1)时的值
        5.根据(3)的梯度对参数进行更新
    notice:
        1. Frobenius norm (=L2 when p=2) is used as default in torch.norm
        2. for NLP, attack should be applied to only embeddings (word_embeddings, position_embeddings,
        token_type_embeddings)
    """
    def __init__(self, model: nn.Module, eps=1.):
        self.model = (
            model.module if hasattr(model, "module") else model
        )
        self.eps = eps
        self.backup = {}
        
    # only attack word embedding
    def attack(self, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm and not torch.isnan(norm):
                    r_at = self.eps * param.grad / norm
                    param.data.add_(r_at)
                    
    def restore(self, emb_name='word_embeddings'):
        for name, para in self.model.named_parameters():
            if para.requires_grad and emb_name in name:
                assert name in self.backup
                para.data = self.backup[name]
        self.backup = {}

class PGD:
    """pgd in NLP - this follows pseudo code in https://zhuanlan.zhihu.com/p/103593948
    对于每个x:
        1.计算x的前向loss、反向传播得到梯度并备份embedding & gradient
        对于每步t:
            2.根据embedding矩阵的梯度计算出r，并加到当前embedding上，相当于x+r(超出范围则投影回epsilon内)
            3.t不是最后一步: 将梯度归0，根据1的x+r计算前后向并得到梯度
            4.t是最后一步: 恢复(1)的梯度，计算最后的x+r并将梯度累加到(1)上
        5.将embedding恢复为(1)时的值
        6.根据(4)的梯度对参数进行更新
    notice:
        1. Frobenius norm (=L2 when p=2) is used as default in torch.norm
        2. for NLP, attack should be applied to only embeddings (word_embeddings, position_embeddings,
        token_type_embeddings)
    """

    def __init__(self, model, eps=1., alpha=0.3, steps=3):
        self.model = (
            model.module if hasattr(model, "module") else model
        )
        self.steps = steps
        self.eps = eps
        self.alpha = alpha
        self.emb_backup = {}
        self.grad_backup = {}
        
    def attack(self, emb_name='word_embeddings', is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data)
                    
    def restore(self, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}
        
    def project(self, param_name, param_data):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > self.eps:
            r = self.eps * r / torch.norm(r)
        return self.emb_backup[param_name] + r
    
    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()
                
    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]

class EMA:
    '''
        Current version is a modification on the one described below, it acts as a storage for teacher model's parameters
        Maintain expontential moving average of a model. This should have same function as the `tf.train.ExponentialMovingAverage` of tensorflow.
        usage:
            model = resnet()
            model.train()
            ema = EMA(model, 0.9999)
            ....
            for img, lb in dataloader:
                loss = ...
                loss.backward()
                optim.step()
                ema.update_params() # apply ema
            evaluate(model)  # evaluate with original model as usual
            ema.apply_shadow() # copy ema status to the model
            evaluate(model) # evaluate the model with ema paramters
            ema.restore() # resume the model parameters
        args:
            - model: the model that ema is applied
            - alpha: each parameter p should be computed as p_hat = alpha * p + (1. - alpha) * p_hat
            - buffer_ema: whether the model buffers should be computed with ema method or just get kept
        methods:
            - update(): apply ema to the model, usually call after the optimizer.step() is called
            - apply_shadow(): copy the ema processed parameters to the model
            - restore(): restore the original model parameters, this would cancel the operation of apply_shadow()
    '''
    def __init__(self, model, decay, alpha=10, max_epochs=17, update_every_n_steps=20):
        self.model = model
        self.teacher_model = copy.deepcopy(model)
        for _, param in self.teacher_model.named_parameters():
            param.requires_grad = False
        # next(model.parameters()).is_cuda
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.alpha = alpha
        self.max_epochs = max_epochs
        self.method = 'average'
        def avg_fcn(avg, m, num_averaged):
            return avg + (m - avg) / (num_averaged + 1)
        self.avg_fcn = avg_fcn
        self.n_averaged = 0
        self.calc_loss = False
        self.update_every_n_steps = update_every_n_steps
        # self.__register()

    def update_teacher(self, step, epoch, last_n_epochs=0):
        if self.method == 'exponential_decay':
            # 开始几个step还是以新模型的权重为主, alpha控制权重递减
            decay = min(self.decay, 1 - self.alpha/(step+1))
            # flooding
            if epoch+1 > self.max_epochs-last_n_epochs:
                decay = 0.8     # 0.8 hist 0.2 current
            for mean_param, param in zip(self.teacher_model.parameters(), self.model.parameters()):
                # mean_param.data = (1.0 - decay) * param.data + self.decay * mean_param.data
                mean_param.data.mul_(decay).add_(1-decay, param.data)
        elif self.method == 'average':
            for mean_param, param in zip(self.teacher_model.parameters(), self.model.parameters()):
                mean_param.copy_(self.avg_fcn(mean_param, param.detach(), self.n_averaged))
            self.n_averaged += 1
    
    def __register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update_student(self):
        # to trust new model's parameter more at beginning
        decay = min(self.decay, 1- 1/(self.step + self.alpha))
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - decay) * param.data + decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
                
    def apply_shadow_to_student(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
                
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}