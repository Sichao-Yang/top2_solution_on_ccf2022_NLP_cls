import torch
from torch.utils.data import SequentialSampler, DataLoader
import os
import numpy as np
from config import parse_args
from tqdm import tqdm 
import json
from os import path as osp
import sys
sys.path.append(osp.abspath(osp.dirname(__file__)))
from data_helper import BaseModelDataset

DEBUG = False

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from NEZHA.modeling_nezha import NeZhaModel
from NEZHA.configuration_nezha import NeZhaConfig
from torch.nn import Dropout
import copy
from model import Model_v1 as Model


def inference(args=None, logging=None, recover_from_npy=False):
    if args is None:
        args = parse_args()
    if logging is None:
        from logger import get_logger
        logging = get_logger(filename='tmp.log')
    # print(args.ckpt_file)
    # print(args.test_batch_size)
    anns=list()
    with open(args.test_filepath,'r',encoding='utf8') as f:
        for line in f.readlines():
            ann =json.loads(line)
            anns.append(ann)
    if DEBUG:
        anns = anns[:300]
            
    dataset = BaseModelDataset(args, anns, test_mode=True, method=args.seq_concat_method)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
                            batch_size=args.test_batch_size,
                            sampler=sampler,
                            drop_last=False,
                            pin_memory=True,
                            num_workers=args.num_workers,
                            prefetch_factor=args.prefetch)
    # 2. load model
    model = Model(args, device=args.device, 
                  reinit_n_layers=args.reinit_n_layers, 
                  remove_n_layers=args.remove_n_layers,
                  method=args.cls_method)   

    if torch.cuda.is_available():
        model = model.cuda()
    for cv in range(args.fold_num):
        # if cv!=1:
        #     continue
        # get the filename of the max f1 score from this cv
        thiscv = [x for x in os.listdir(args.savedmodel_filepath) if f'cv{cv}' in x]
        f1s = np.array([eval('0.'+x.split('_')[-1].split('.')[1]) for x in thiscv])
        selected = thiscv[np.where(f1s==max(f1s))[0][0]]
        thiscv.remove(selected)
        # remove all other .bin files
        for x in thiscv:
            os.remove(osp.join(args.savedmodel_filepath, x))
        args.ckpt_file = osp.join(args.savedmodel_filepath, selected)
        
        checkpoint = torch.load(args.ckpt_file)
        model.load_state_dict(checkpoint['model_state_dict'],strict=False)
        
        torch.save({'model_state_dict': model.state_dict()}, args.ckpt_file)


if __name__ == '__main__':
    args = parse_args()
    
    convert_nezha = False
    if convert_nezha:
        paras = torch.load(osp.join(args.bert_dir,'pytorch_model.bin'))
        from collections import OrderedDict
        new_paras = OrderedDict()
        for name, emb in paras.items():
            if 'nezha' in name:
                name = name.replace('nezha.', '')
            new_paras[name] = emb
        torch.save(new_paras, osp.join(args.bert_dir,'pytorch_model.bin'))    
        
        config = NeZhaConfig.from_pretrained(args.bert_dir, output_hidden_states=True)
        model = NeZhaModel.from_pretrained(args.bert_dir, config = config)    
        torch.save(model.state_dict(), osp.join(args.bert_dir,'pytorch_model.bin'))
    
    
    args.output_dir = osp.join(args.output_dir, '20221105_23-22_6448_sub0')
    # m_list = ['', 'ema']
    m_list = ['ema']
    for m in m_list:
        args.savedmodel_filepath = osp.join(args.output_dir, osp.basename(args.savedmodel_filepath), m)
        # args.savedmodel_filepath = osp.join(args.output_dir, osp.basename(args.savedmodel_filepath))
        args.device = 'cuda'
        inference(args)