#!/usr/bin/env python
# coding: utf-8

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

import sys
from os import path as osp
sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '../')))

from NEZHA.modeling_nezha import NeZhaModel
from NEZHA.configuration_nezha import NeZhaConfig

from transformers import BertTokenizer
import logging
print(torch.__version__)
import warnings
import transformers
transformers.logging.set_verbosity_error()
warnings.filterwarnings('ignore')

from models import *

root = osp.abspath(osp.join(osp.dirname(__file__), '../../'))

with open(osp.join(root, 'raw_data/train.json'), 'r') as f:
    train_data = f.readlines()
    train_data = [eval(i.strip())for i in train_data]
    
with open(osp.join(root, 'raw_data/testB.json'), 'r') as f:
    test_b = f.readlines()
    test_b = [eval(i.strip())for i in test_b]
    

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


class args(object):
    
    def __init__(self):
        
        # 预训练模型路径
        self.model_checkpoint =  osp.join(root, 'user_data/nezha_model')
        self.bert_model_checkpoint = osp.join(root, 'user_data/mac_bert_model')
        # 模型有如下选择，具体详见 models.py
        # BertLastTwoCls BertLastFourCls BertLastFourEmbeddingsPooler BertDynEmbeddings BertRNN
        self.model_type = 'BertLastFourCls'
#         self.nezha = True
        self.load_pretrained = False 
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

# train_dict = create_traindata(train_data, tokenizer)
# pseudo_dict_b = create_traindata(pseudo_data_b, tokenizer)

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


def test_model(model, test_loader, test_count):
    
    test_preds_fold = np.zeros((test_count, 36)) ####
    
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

            y_pred = torch.softmax(y_pred, dim=-1).cpu().detach().numpy()
            test_bidx= i * test_batch_size
            test_eidx= (i + 1) * test_batch_size
            test_preds_fold[test_bidx:test_eidx] = y_pred
            
    return test_preds_fold


test_dict = create_testdata(test_b, tokenizer)
test_batch_size = args.test_batch_size
test_dataset =  WBDataset(test_dict, tokenizer, test=True)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, collate_fn=test_dataset.collate, shuffle=False, num_workers=4)


import os
file_names = []
file_path = osp.join(root, 'user_data/b_models/')
#提取文件中的所有文件生成一个列表
folders = os.listdir(file_path)
for file in folders:
    if '.bin' in file:
        file_names.append(file_path + file)
        file_names.sort()


device = torch.device('cuda')
nezhaModelBertLastFourEmbeddingsPooler = BertLastFourEmbeddingsPooler(args, nezha=True, load_pretrained=False).to(device)
nezhaModelBertLastTwoCls = BertLastTwoCls(args, nezha=True, load_pretrained=False).to(device)
nezhaModelBertLastFourCls = BertLastFourCls(args, nezha=True, load_pretrained=False).to(device)

bertModelBertLastFourEmbeddingsPooler = BertLastFourEmbeddingsPooler(args, nezha=False, load_pretrained=False).to(device)
bertModelBertLastTwoCls = BertLastTwoCls(args, nezha=False, load_pretrained=False).to(device)
bertModelBertLastFourCls = BertLastFourCls(args, nezha=False, load_pretrained=False).to(device)


b_test_preds = {}
for file_name in file_names[:]:
    if 'LastFourCls' in file_name:
        if 'nezha' in file_name:
            model = nezhaModelBertLastFourCls
        else:
            model = bertModelBertLastFourCls
    elif 'LastFourEmbeddingsPooler' in file_name:
        if 'nezha' in file_name:
            model = nezhaModelBertLastFourEmbeddingsPooler
        else:
            model = bertModelBertLastFourEmbeddingsPooler
    elif 'LastTwoCls' in file_name:
        if 'nezha' in file_name:
            model = nezhaModelBertLastTwoCls
        else:
            model = bertModelBertLastTwoCls
    else:
        print(fine_name)
    
    model.load_state_dict(torch.load(file_name))
    test_pred = test_model(model, test_loader, len(test_b))
    b_test_preds[file_name] = test_pred



df_pred = pd.read_csv(osp.join(root, 'raw_data/submit_example_B.csv'))


test_preds = []
for k, v in b_test_preds.items():
    test_preds.append(v)


final_preds = np.zeros(test_preds[0].shape)
for idx in range(len(test_preds)):
    
    test_preds_fold = test_preds[idx]
    final_preds += test_preds_fold / len(test_preds)
    
final_reses = np.argmax(final_preds, axis=1)

final_sub = []
for i in range(len(df_pred)):
    final_res = final_reses[i]
    final_sub.append(final_res)


df_pred['label'] = final_sub
df_pred.to_csv(osp.join(root, 'prediction_result/submit_finalB.csv'))

