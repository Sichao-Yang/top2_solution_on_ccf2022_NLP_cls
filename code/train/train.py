#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys 
sys.path.append("..") 

import random
import numpy as np
from tqdm import tqdm
import time
from functools import partial
import scipy as sp
import joblib
import gc
from sklearn.model_selection import StratifiedKFold
import re
import math
import json
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from sklearn.metrics import accuracy_score


from layers.losses import LabelSmoothingCrossEntropy, FocalLoss, LabelSM_Focal, DiceLoss, LabelSmoothingCrossEntropyWeight
from torch.nn import CrossEntropyLoss
from layers.adversarial import FGM, PGD, Lookahead

from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from sklearn.metrics import accuracy_score

from NEZHA.modeling_nezha import NeZhaModel
from NEZHA.configuration_nezha import NeZhaConfig
# from transformers import NezhaModel
# from transformers import NezhaConfig

from transformers import *
import logging
import warnings
import transformers
transformers.logging.set_verbosity_error()
warnings.filterwarnings('ignore')


from models import *


# In[2]:


# 写日志
logger = logging.getLogger('mylogger')
logger.setLevel(logging.DEBUG)
timestamp = time.strftime("%Y.%m.%d_%H.%M.%S", time.localtime())
fh = logging.FileHandler('log_model.txt')
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s][%(levelname)s] ## %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)


# In[3]:


# 随机数种子
def seed_all(seed_value):
    random.seed(seed_value) # Python
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False


# In[4]:


class args(object):
    
    def __init__(self):
        
        # 预训练模型路径
        self.model_checkpoint =  '../../user_data/nezha_model'
        self.bert_model_checkpoint = '../../user_data/mac_bert_model'
        # 模型有如下选择，具体详见 models.py
        # BertLastTwoCls BertLastFourCls BertLastFourEmbeddingsPooler BertDynEmbeddings BertRNN
        self.model_type = 'BertLastFourCls'
        self.nezha = True
        self.load_pretrained = True 
        self.loss_type = 'ce'
        self.device = 'cuda'
        self.max_length = 412
        self.bert_dim = 768
        self.test_batch_size = 64
        
        # 对抗训练
        self.fgm = 0
        self.pgd = 0
        self.fp16 = 0
        
        # 伪标签训练设置
        self.pseudo = 0
        self.traindata_epoch = 2
        
        # reinit以及llrd
        self.num_reinit_layers = 0
        self.reinit_pooler = False
        self.layerwise_learning_rate_decay = 1
        
args = args()
tokenizer = BertTokenizer.from_pretrained(args.model_checkpoint)


# In[5]:


# 读取数据
with open('../../raw_data/train.json', 'r') as f:
    train_data = f.readlines()
    train_data = [eval(i.strip())for i in train_data]
    
with open('../../raw_data/train.json', 'r') as f:
    test_a = f.readlines()
    test_a = [eval(i.strip())for i in test_a]
    
with open('../../raw_data/testB.json', 'r') as f:
    test_b = f.readlines()
    test_b = [eval(i.strip())for i in test_b]
    
with open('../../raw_data/final_pseudo_b.json', 'r') as f:
    pseudo_data_b = json.load(f)


# In[6]:


def create_traindata(train_data, tokenizer, max_len=412):
    train_dict ={'input_ids': [], 'token_type_ids': [], 'attention_mask': [], 'input_lengths':[], 'labels': []}

#     max_len = args.max_length - 4
    for train_data_ in tqdm(train_data[: ]):
        title = train_data_['title']
        assignee = train_data_['assignee']
        abstract = train_data_['abstract']
        label = int(train_data_['label_id'])

        title_ids_ori = tokenizer.encode(title, add_special_tokens=False)[:30]
        assignee_ids_ori = tokenizer.encode(assignee, add_special_tokens=False)[-10:]
        abstract_ids_ori = tokenizer.encode(abstract, add_special_tokens=False)

        title_len = len(title_ids_ori)
        assignee_len = len(assignee_ids_ori)
        abstract_len = len(abstract_ids_ori)

        abstract_cutlen = max_len - title_len - assignee_len

        if abstract_cutlen >= abstract_len:
            token_ids = [101] + title_ids_ori + [102] + assignee_ids_ori + [102] + abstract_ids_ori + [102]
        else:
            mid_len = int(abstract_cutlen / 2)
            token_ids = [101] + title_ids_ori + [102] + assignee_ids_ori + [102] + abstract_ids_ori[: mid_len] + abstract_ids_ori[-mid_len: ] + [102]
        token_type_ids = [0] * len(token_ids)
        attention_mask = [1] * len(token_ids)
        input_lengths = len(token_ids)

        train_dict['input_ids'].append(token_ids)
        train_dict['token_type_ids'].append(token_type_ids)
        train_dict['attention_mask'].append(attention_mask)
        train_dict['input_lengths'].append(input_lengths)
        train_dict['labels'].append(label)

    train_dict = {k: np.array(train_dict[k]) for k in train_dict}
    return train_dict


def create_testdata(test_data, tokenizer, max_len=412):
    test_dict ={'input_ids': [], 'token_type_ids': [], 'attention_mask': [], 'input_lengths':[], 'data_idx': []}
    test_data_all = []
    for idx, i in enumerate(test_data):
        i['data_idx']=idx
        test_data_all.append(i)

    for idx, test_data_ in tqdm(enumerate(test_data_all)):
        title = test_data_['title']
        assignee = test_data_['assignee']
        abstract = test_data_['abstract']

        title_ids_ori = tokenizer.encode(title, add_special_tokens=False)[:30]
        assignee_ids_ori = tokenizer.encode(assignee, add_special_tokens=False)[-10:]
        abstract_ids_ori = tokenizer.encode(abstract, add_special_tokens=False)

        title_len = len(title_ids_ori)
        assignee_len = len(assignee_ids_ori)
        abstract_len = len(abstract_ids_ori)

        abstract_cutlen = max_len - title_len - assignee_len

        if abstract_cutlen >= abstract_len:
            token_ids = [101] + title_ids_ori + [102] + assignee_ids_ori + [102] + abstract_ids_ori + [102]
        else:
            mid_len = int(abstract_cutlen / 2)
            token_ids = [101] + title_ids_ori + [102] + assignee_ids_ori + [102] + abstract_ids_ori[: mid_len] + abstract_ids_ori[-mid_len: ] + [102]
        token_type_ids = [0] * len(token_ids)
        attention_mask = [1] * len(token_ids)
        input_lengths = len(token_ids)

        test_dict['input_ids'].append(token_ids)
        test_dict['token_type_ids'].append(token_type_ids)
        test_dict['attention_mask'].append(attention_mask)
        test_dict['input_lengths'].append(input_lengths)
        test_dict['data_idx'].append(idx)


    test_dict = {k: np.array(test_dict[k]) for k in test_dict}
    return test_dict


# In[7]:


class WBDataset(Dataset):

    def __init__(self, data, tokenizer, batch_first=True, test=False):
        self.data = data
        self.tokenizer = tokenizer
        self.pad = tokenizer.pad_token_id
        self.batch_first = batch_first
        self.test = test

    def __len__(self):
        return len(self.data['input_ids'])

    def __getitem__(self, index):
        instance = {}
        instance['input_ids'] = self.data['input_ids'][index]
        instance['token_type_ids'] = self.data['token_type_ids'][index]
        instance['attention_mask'] = self.data['attention_mask'][index]
        instance['input_lengths'] = self.data['input_lengths'][index]
        if not self.test:
            instance['labels'] = self.data['labels'][index]
        else:
            instance['data_idx'] = self.data['data_idx'][index]
        
        return instance

    def collate(self, batch):
        
        input_ids = pad_sequence(
            [torch.tensor(instance["input_ids"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad)
        
        token_type_ids = pad_sequence(
            [torch.tensor(instance["token_type_ids"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad)
        
        attention_mask = pad_sequence(
            [torch.tensor(instance["attention_mask"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad)
        
        input_lengths = torch.tensor([torch.tensor(instance["input_lengths"], dtype=torch.int) for instance in batch])
        
        if not self.test:
            labels = torch.tensor([torch.tensor(instance["labels"], dtype=torch.long) for instance in batch])

            return input_ids, token_type_ids, attention_mask, input_lengths, labels
        else:
            data_idx = torch.tensor([torch.tensor(instance["data_idx"], dtype=torch.long) for instance in batch])
            
            return input_ids, token_type_ids, attention_mask, input_lengths, data_idx


# In[8]:


class OptimizedF1(object):
    def __init__(self):
        self.coef_ = []
    def _kappa_loss(self, coef, X, y):
        """
        y_hat = argmax(coef*X, axis=-1)
        :param coef: (1D array) weights
        :param X: (2D array)logits
        :param y: (1D array) label
        :return: -f1
        """
        X_p = np.copy(X)
        X_p = coef*X_p
#         ll = accuracy_score(y, np.argmax(X_p, axis=-1))
        ll = f1_score(y, np.argmax(X_p, axis=-1), average='macro')
        return -ll
    
    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [1. for _ in range(36)] ###########
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')
        
    def predict(self, X, y):
        X_p = np.copy(X)
        X_p = self.coef_['x'] * X_p
#         return accuracy_score(y, np.argmax(X_p, axis=-1))
        return f1_score(y, np.argmax(X_p, axis=-1), average='macro')
    
    def coefficients(self):
        return self.coef_['x']
    
def metric(y_true, y_pred):
    
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)
    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
        
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average= 'macro')
    return acc, f1


# In[9]:



def train_model(data_loader, model, optimizer, device, scheduler):
    logger.info('training.............')
    model.train()
    losses = []
    train_f1 = []

    if args.fp16==1:
        from torch.cuda import amp
        scaler = amp.GradScaler()
    if args.fgm==1:
        fgm = FGM(model)
    
    tk0 = tqdm(data_loader, total=len(data_loader))
    for step, batch in enumerate(tk0):
        batch = tuple(t.to(device) for t in batch)
        ids, segids, mask, lens, y_truth = batch

        outputs = model(
            input_ids = ids, 
            input_mask = mask, 
            input_segids = segids,
            input_lengths = lens,
            input_labels = y_truth
        ) 
        
        loss, logits = outputs
        loss = loss.mean() / grad_acc
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        
        if args.fp16==1:
            scaler.scale(loss).backward()
        else:
            loss.backward()
            
        if args.fgm==1:
            fgm.attack()
            loss_adv = model(
                input_ids = ids, 
                input_mask = mask, 
                input_segids = segids,
                input_lengths = lens,
                input_labels = y_truth
            )[0]
            loss_adv = loss_adv.mean()
            loss_adv.backward()
            fgm.restore()
            
        if args.pgd==1:
            pgd = PGD(model)
            K = 3
            pgd.backup_grad()
            for t in range(K):
                # 在embedding上添加对抗扰动, first attack时备份param.data
                pgd.attack(is_first_attack=(t == 0))
                if t != K - 1:
                    model.zero_grad()
                else:
                    pgd.restore_grad()
                loss_adv = model(
                    input_ids = ids, 
                    input_mask = mask, 
                    input_segids = segids,
                    input_lengths = lens,
                    input_labels = y_truth
                )[0]
                loss_adv = loss_adv.mean()
                loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            pgd.restore()  # 恢复embedding参数
            
        y_prod_cpu = logits.cpu().detach().numpy()
        y_truth_cpu = y_truth.cpu().detach().numpy()
        
        acc, f1 = metric(y_truth_cpu, y_prod_cpu)
        train_f1.append(acc)
        losses.append(loss.item())  
        if step == len(data_loader)-1 or (step + 1) % grad_acc==0:
            
            if args.fp16==1:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
                
            scheduler.step()
            optimizer.zero_grad()
            model.zero_grad()          
            
        tk0.set_postfix(loss=np.mean(losses), avg_f1=np.mean(train_f1))


# In[10]:


def eval_model(data_loader, model, val_count, device=torch.device("cpu")):
    logger.info('evaling.............')
    model.eval()
    losses = []
    eval_f1 = []
    preds_tags = []
    ture_labels = []
    
    valid_preds_fold = np.zeros((val_count, 36))
    val_true = np.zeros((val_count))
    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader))
        for step, batch in enumerate(tk0):
            batch = tuple(t.to(device) for t in batch)
            ids, segids, mask, lens, y_truth = batch

            outputs = model(
                input_ids = ids, 
                input_mask = mask, 
                input_segids = segids,
                input_lengths = lens,
                input_labels = y_truth
            ) 
            
            loss, logits = outputs
            loss = loss.mean()
            y_prod_cpu = logits.cpu().detach().numpy()
            y_truth_cpu = y_truth.cpu().detach().numpy()
            _, f1 = metric(y_truth_cpu, y_prod_cpu)
            eval_f1.append(f1)        
            losses.append(loss.item())                                 

            val_bidx = step * batch_size
            val_eidx = (step + 1) *batch_size
            if val_eidx > val_count:
                val_eidx = val_count
            valid_preds_fold[val_bidx: val_eidx] = torch.softmax(logits, dim=-1).cpu().numpy()
            val_true[val_bidx: val_eidx] = y_truth_cpu
            tk0.set_postfix(loss=np.mean(losses), avg_f1=np.mean(eval_f1))

        op = OptimizedF1()
        op.fit(valid_preds_fold, val_true)
        class_weights_fold = op.coefficients()
        # print(val_true)
        # print(np.argmax(valid_preds_fold, axis=1))
        befor_f1, befor_f1 = metric(val_true, valid_preds_fold)
        post_f1, post_f1 = metric(val_true, valid_preds_fold * class_weights_fold)
            
        logger.info("***** Report eval result *****")
        logger.info("befor_f1:{:.4f}, post_f1:{:.4f}".format(befor_f1, post_f1))            

        
        return valid_preds_fold, class_weights_fold, befor_f1, post_f1


# In[11]:



def test_model(model, test_loader, fold, test_count):
    
    test_preds_fold = np.zeros((test_count, 36)) ####
    model.load_state_dict(torch.load( '{}_{}.bin'.format(bset_model_path,fold)))
    
    model.eval()
    tk0 = tqdm(test_loader, total=len(test_loader))
    with torch.no_grad():
        for i, batch in enumerate(tk0):
            batch = tuple(t.to(device) for t in batch)
            ids, segids, mask, lens, y_truth = batch

            y_pred = model(
                input_ids = ids, 
                input_mask = mask, 
                input_segids = segids,
                input_lengths = lens,
            )[0] 
            # print(y_pred)
            y_pred = torch.softmax(y_pred, dim=-1).cpu().detach().numpy()
            test_bidx= i * batch_size
            test_eidx= (i + 1) * batch_size
            test_preds_fold[test_bidx:test_eidx] = y_pred
            
    return test_preds_fold


# In[12]:



def run(fold, all_data, train_idx, val_idx, batch_size, learning_rate,other_lr, test_inputs=None):


    train_data = {k: all_data[k][train_idx] for k in train_dict}
    valid_data = {k: all_data[k][val_idx] for k in train_dict}

    train_dataset = WBDataset(train_data, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=train_dataset.collate, shuffle=True, num_workers=4)
    
#     pseudo_data = {k: np.concatenate([pseudo_dict_b[k], all_data[k][train_idx]]) for k in all_data}
    if args.pseudo==1:
        pseudo_data = {k: pseudo_dict[k] for k in pseudo_dict}
        pseudo_dataset = WBDataset(pseudo_data, tokenizer)
        pseudo_loader = DataLoader(pseudo_dataset, batch_size=batch_size, collate_fn=train_dataset.collate, shuffle=True, num_workers=4)
    else:
        val_count = len(val_idx)

    valid_dataset = WBDataset(valid_data, tokenizer)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=train_dataset.collate, shuffle=True, num_workers=4)

    if test_inputs is not None:
        test_dataset =  WBDataset(test_inputs, tokenizer, test=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=test_dataset.collate, shuffle=False, num_workers=4)
    
    
    device = torch.device("cuda")
    # load_pretrained loss_type
    traindata_epoch = args.traindata_epoch
#     BertLastTwoCls BertLastFourCls BertLastFourEmbeddingsPooler BertDynEmbeddings BertRNN
    if args.model_type == 'BertLastTwoCls':
        model = BertLastTwoCls(args, nezha=args.nezha, load_pretrained=args.load_pretrained, n_class=36).to(device)
    elif args.model_type == 'BertLastFourCls':
        model = BertLastFourCls(args, nezha=args.nezha, load_pretrained=args.load_pretrained, n_class=36).to(device)
    elif args.model_type == 'BertDynEmbeddings':
        model = BertDynEmbeddings(args, nezha=args.nezha, load_pretrained=args.load_pretrained, n_class=36).to(device)
    elif args.model_type == 'BertRNN':
        model = BertRNN(args, nezha=args.nezha, load_pretrained=args.load_pretrained, n_class=36).to(device)
    else:
        model = BertLastFourEmbeddingsPooler(args, nezha=args.nezha, load_pretrained=args.load_pretrained, n_class=36).to(device)
            
    if args.pseudo==1:
        if traindata_epoch > 0:
            num_train_steps = math.ceil(len(train_loader)/grad_acc * traindata_epoch + len(pseudo_loader)/grad_acc * (num_epochs-traindata_epoch))
        else:
            num_train_steps = math.ceil((len(pseudo_loader))/grad_acc * num_epochs)
    else:
        num_train_steps = math.ceil(len(train_loader)/grad_acc * num_epochs)
    
    logger.info('num_train_steps:{}'.format(num_train_steps))
    
    weight_decay = 0.0001
    head_decay = 0.
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    if args.layerwise_learning_rate_decay == 1:
        encoder_parameters = [(n, p) for n, p in model.named_parameters() if "pretrained_model"  in n]
        decoder_parameters =  [(n, p) for n, p in model.named_parameters() if "pretrained_model" not in n]
        
        optimizer_parameters = [
            {"params": [p for n, p in encoder_parameters if not any(nd in n for nd in no_decay)], "weight_decay": weight_decay, 'lr': learning_rate},
            {"params": [p for n, p in encoder_parameters if any(nd in n for nd in no_decay)], "weight_decay": 0.0, 'lr': learning_rate},
            
            
            {"params": [p for n, p in decoder_parameters ], "weight_decay": head_decay, 'lr': other_lr},

        ]
    else:
        encoder_parameters = [(n, p) for n, p in model.named_parameters() if "pretrained_model" in n and "pooler" not in n]
        decoder_parameters = [(n, p) for n, p in model.named_parameters() if "pretrained_model" not in n]
        pooler_parameters = [(n, p) for n, p in model.named_parameters() if  "pooler" in n]   + pooler_parameters 
        
        optimizer_parameters = [
            {"params": [p for n, p in decoder_parameters], "weight_decay": head_decay, 'lr': other_lr},
        ]
        
        lr = learning_rate
        layers = [model.pretrained_model.embeddings] + list(model.pretrained_model.encoder.layer)
        layers.reverse()
        for layer in layers:
            lr *= args.layerwise_learning_rate_decay
            optimizer_parameters += [
                {"params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": weight_decay, "lr": lr,},
                {"params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0, "lr": lr,},
            ]

        
    # get_cosine_schedule_with_warmup get_linear_schedule_with_warmup
    optimizer = AdamW(optimizer_parameters)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,  
        num_warmup_steps=int(num_train_steps * 0.0), 
        num_training_steps=num_train_steps
    )

    best_f1 = -99999
    early_stop_count = 0
    for epoch in range(num_epochs):
        
        
        if args.pseudo==1 and epoch >= traindata_epoch:
            train_loader = pseudo_loader
            
        model.train()
        train_model(train_loader, model, optimizer, device, scheduler=scheduler)
        
        if args.pseudo==1:
#             if epoch > num_epochs-2: break
#             if epoch >= 3:
#                 torch.save(model.state_dict(), '{}_{}.bin'.format(bset_model_path,fold))
                
            if epoch >= num_epochs - 2:
                torch.save(model.state_dict(), '{}_{}_epoch{}.bin'.format(bset_model_path, fold, epoch))
                
        else:
            if epoch >= 2:
                valid_preds_fold, class_weights_fold, befor_f1, post_f1 = eval_model(valid_loader, model, val_count, device)

                if befor_f1 > best_f1:
                    early_stop_count = 0
                    best_f1 = befor_f1
                    torch.save(model.state_dict(), '{}_{}.bin'.format(bset_model_path,fold))
                else:
                    early_stop_count += 1
                if early_stop_count > patience:
                    logger.info("Early stopping")
                    torch.cuda.empty_cache()
                    torch.cuda.empty_cache() 
                    break


    if test_inputs is not None:
        test_preds_fold = test_model(model, test_loader, fold, test_count=len(test_dict['input_ids']))

        return test_preds_fold, class_weights_fold, best_f1
    else:
        return class_weights_fold, best_f1


# In[13]:


# 数据集处理
train_dict = create_traindata(train_data, tokenizer)
pseudo_dict = create_traindata(pseudo_data_b, tokenizer)
test_dict = create_testdata(test_b, tokenizer)


# In[ ]:


# 参数设置，多个则是类似网格搜索如下所示
learning_rates = [3e-5]
num_epochses = [5]
batch_sizes = [32]
other_lrs = [1e-4]
patience = 2
grad_acc = 1
n_splits = 5
device = torch.device('cuda')
output_unique = 36
search_model = 'layerwise1-wd0.0001-warmp0.0'
seed_value = 666

# 每一个fold随机数，不同可以使得模型初始化有所差异，提升融合效果
seed_values = list(range(seed_value, seed_value+n_splits))
# seed_values = [412, 927, 1227, 1992, 2020, 2010, 1990, 1966, 2022, 6666]
test_preds_folds = []
thresholds = []
seed_all(seed_value)

all_best_f1 = []
for learning_rate, other_lr, batch_size, num_epochs in zip(learning_rates, other_lrs, batch_sizes, num_epochses):
    
    model_name = 'nezha_base-pseudo_labels_b-{}-{}-seed{}-gkf{}-{}-n_splits{}-grad_acc{}-num_epochs{}'.format(args.model_type, args.loss_type, seed_value, seed_value,
                                                                                                              search_model,n_splits, grad_acc, num_epochs)
    logger.info('==========learning_rate:{},batch_size:{}============='.format(learning_rate, batch_size))
    current_config_f1 = 0
    kf = StratifiedKFold(n_splits=n_splits ,shuffle=True, random_state=seed_value).split(
        X=train_dict['labels'], y=train_dict['labels'])

    bset_model_path = "../../user_data/best_models/{}-{}-{}_fold".format(model_name, learning_rate, batch_size)
    n_count = 0

    class_weights = []
    split_idxes = []
    for fold,(train_index,valid_index) in enumerate(kf):
        logger.info('==========fold:{}============='.format(fold))
        seed_all(seed_values[fold])
        split_idxes.append((train_index,valid_index))
        n = 0
        all_data = train_dict
        test_preds_fold, new_thresholds, best_f1= run(fold, all_data, train_index, valid_index, batch_size, learning_rate, other_lr, test_inputs=test_dict)

        
        all_best_f1.append(best_f1)
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        thresholds.append(new_thresholds)
        test_preds_folds.append(test_preds_fold)
        
    final_results = {'test_preds_folds':test_preds_folds, 'thresholds':thresholds, 'data_idx': test_dict['data_idx'],
                    'all_best_f1': all_best_f1, 'seed_values': seed_values, 'seed_value': seed_value, 'split_idxes': split_idxes}

    with open("../../user_data/results/{}-{}-{}.pk".format(model_name, learning_rate, batch_size),'wb')as f:
        joblib.dump(final_results,f)
    

    logger.info('learning_rate:{},batch_size:{}'.format(learning_rate, batch_size))
    logger.info('=========================================================')


# In[ ]:




