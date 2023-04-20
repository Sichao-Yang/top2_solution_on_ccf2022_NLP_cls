import json
import random
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import BertTokenizer
import random
from os import path as osp
import numpy as np
import sys
sys.path.append(osp.abspath(osp.dirname(__file__)))
# from edazh.eda import EdaZh

join = lambda x,y: osp.abspath(osp.join(x, y))


def probability_smoothing(samplesize_per_cls, method='prob_inverse', eps=0.00001):
    if method == 'prob_inverse':
        # same as the one used in modified focal loss
        beta = 0.99
        effective_num = 1.0 - np.power(beta, samplesize_per_cls)
        weights = (1.0 - beta) / np.array(effective_num)
        probs = list(weights/np.sum(weights))
    elif method == 'smoothing':
        # add uniform probs to average the original prob
        k = 1/len(samplesize_per_cls)*1
        probs = [(x/sum(samplesize_per_cls) + k)/2 for x in samplesize_per_cls]
    elif method == 'uniform':
        probs = [1/len(samplesize_per_cls)]*len(samplesize_per_cls)
    assert abs(sum(probs) - 1) < eps
    return probs

class BaseModelDataset(Dataset):
    def __init__(self,
                 args,
                 anns,
                 test_mode: bool = False,
                 max_lengths= [30,15,450],
                 method = 'micro_fix'):
        self.test_mode = test_mode
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_dir)
        self.anns = anns
        self.max_lengths = max_lengths
        self.method = method
        
    def __len__(self) -> int:
        return len(self.anns)
    
    def __getitem__(self, idx: int) -> dict:
        # id = self.anns[idx]['id']
        # s1: readout text data
        title = self.anns[idx]['title']
        assignee = self.anns[idx]['assignee']
        abstract = self.anns[idx]['abstract']
        # s2: concat texts & get token_ids
        if self.method == 'macro_fix':
            text = "专利标题："+title+"。"+"申请人："+assignee+"。"+"摘要："+abstract
            text_inputs = self.tokenizer(text, max_length=min(sum(self.max_lengths),sum(self.max_lengths)), padding='max_length', truncation=True)
        elif self.method == 'micro_fix':
            title_inputs = self.tokenizer(title, max_length=self.max_lengths[0], padding='max_length', truncation=True)
            assignee_inputs = self.tokenizer(assignee, max_length=self.max_lengths[1], padding='max_length', truncation=True)
            abstract_inputs = self.tokenizer(abstract, max_length=self.max_lengths[2], padding='max_length', truncation=True)
            # remove the first [cls] token for assginee and abstract parts
            assignee_inputs = {k:v[1:] for k,v in assignee_inputs.items()}
            abstract_inputs = {k:v[1:] for k,v in abstract_inputs.items()}
            # concat 3 parts
            text_inputs = {k: title_inputs[k] + assignee_inputs[k] + abstract_inputs[k] for k in title_inputs}
        # s3: data type formatting
        text_inputs = {k: torch.LongTensor(v) for k,v in text_inputs.items()}
        data = dict(
            text_inputs=text_inputs['input_ids'],
            text_mask=text_inputs['attention_mask'],
            text_type_ids = text_inputs['token_type_ids'],
        )
        # s4: load label if not test mode
        if (not self.test_mode):
            data['label'] = torch.LongTensor([self.anns[idx]['label_id']])
        return data

# class Dataloaders:
#     """it loads ori data, augmented and pseudo labels first and at training time dynamically loads samples of these data
#     """
#     def __init__(self, args, total_epochs, aug_ratios=[0.5, 1.5], pseudo_ratios=[0.5, 3], warmup_ratio=0.2, repreat_num=2):
#         self.args = args
#         self.total_epochs = total_epochs
#         self.aug_ratios = aug_ratios
#         self.pseudo_ratios = pseudo_ratios
#         self.warmup_epochs = max(1,int(warmup_ratio*total_epochs))
#         self.repeat_num = repreat_num
#         anns=list()
#         with open(args.train_filepath,'r',encoding='utf8') as f:
#             for line in f.readlines():
#                 ann =json.loads(line)
#                 anns.append(ann)
#         cv_dir = join(osp.dirname(args.train_filepath), 'cv_ids')
#         with open(join(cv_dir, f'cv_{args.cv_id}.npy'), 'rb') as f:
#             tr_idx = list(np.load(f))
#             va_idx = list(np.load(f))
#         self.train_anns = [anns[i] for i in tr_idx]
#         self.val_anns = [anns[i] for i in va_idx]
        
#         if args.use_aug:
#             val_ids = [x['id'] for x in self.val_anns]
#             path = args.train_filepath.split('.')[0] + '_aug.json'
#             augs = []
#             with open(path,'r',encoding='utf8') as f:
#                 for line in f.readlines():
#                     augs.append(json.loads(line))
#             augs = augs[2*len(anns):]       # remove all true data from aug file
#             augs = [a for a in augs if a['id'] not in val_ids]
#             # train_anns+=train_anns
#             # train_anns+=random.sample(augs, len(train_anns))
#             self.augs = augs
        
#         if args.use_pseudo:
#             pseudo=list()
#             with open(args.pseudo_filepath,'r',encoding='utf8') as f:
#                 for line in f.readlines():
#                     pseudo.append(json.loads(line))        
#             # train_anns+=random.sample(pseudo, len(train_anns)//2)
#             self.pseudo = pseudo
            
#         val_dataset = BaseModelDataset(self.args, self.val_anns)
#         val_sampler = SequentialSampler(val_dataset)
#         self.val_dataloader = DataLoader(val_dataset,
#                                     batch_size=args.val_batch_size,
#                                     sampler=val_sampler,
#                                     drop_last=False,
#                                     pin_memory=True,
#                                     num_workers=args.num_workers,
#                                     prefetch_factor=args.prefetch)
#         self.ori_train_len = len(self.train_anns)
    
#     def get(self, epoch):
#         aug_dataloader, pseudo_dataloader = None, None
#         pseudo_this_epoch, aug_this_epoch = [], []
#         if epoch!=0:
#             f = lambda lo, hi: int((lo+(hi-lo)*min(1,epoch/self.warmup_epochs)) * self.ori_train_len)
#             if self.args.use_pseudo:
#                 num_pseudo = f(self.pseudo_ratios[0], self.pseudo_ratios[1])
#                 pseudo_this_epoch=random.sample(self.pseudo, num_pseudo)
#             if self.args.use_aug:
#                 num_aug = f(self.aug_ratios[0], self.aug_ratios[1])
#                 aug_this_epoch=random.sample(self.augs, num_aug)
            
#         train_this_epoch = self.train_anns*self.repeat_num + pseudo_this_epoch + aug_this_epoch
#         train_dataset = BaseModelDataset(self.args, train_this_epoch)
#         train_sampler = RandomSampler(train_dataset)
#         train_dataloader = DataLoader(train_dataset,
#                                     batch_size=self.args.batch_size,
#                                     sampler=train_sampler,
#                                     drop_last=True,
#                                     pin_memory=True,
#                                     num_workers=self.args.num_workers,
#                                     prefetch_factor=self.args.prefetch)
        
#         return train_dataloader, self.val_dataloader, aug_dataloader, pseudo_dataloader

class Dataloaders_v2:
    """in this version, traindata set is split into three parts to account for different types of data
    """
    def __init__(self, args, total_epochs, aug_ratios=[0.5, 1.5], pseudo_ratios=[0.5, 3], 
                 warmup_ratio=0.2, sample_method='smoothing', resample=False):
        self.args = args
        self.total_epochs = total_epochs
        self.sample_method = sample_method
        self.resmaple = resample
        # self.aug_ratios = aug_ratios
        self.pseudo_ratios = pseudo_ratios
        # self.warmup_epochs = max(1,int(warmup_ratio*total_epochs))
        self.warmup_epochs = 3
        anns=list()
        with open(args.train_filepath,'r',encoding='utf8') as f:
            for line in f.readlines():
                ann =json.loads(line)
                anns.append(ann)
        cv_dir = join(osp.dirname(args.train_filepath), 'cv_ids')
        with open(join(cv_dir, f'cv_{args.cv_id}.npy'), 'rb') as f:
            tr_idx = list(np.load(f))
            va_idx = list(np.load(f))
        self.train_anns = [anns[i] for i in tr_idx]
        self.val_anns = [anns[i] for i in va_idx]
        
        if args.use_aug:
            augs = []
            with open(args.aug_filepath,'r',encoding='utf8') as f:
                for line in f.readlines():
                    augs.append(json.loads(line))
            augs = augs[2*len(anns):]       # remove true data from aug file which is repeated 2*anns
            val_ids = [x['id'] for x in self.val_anns]
            augs = [a for a in augs if a['id'] not in val_ids]      # remove augmented valdata
            # train_anns+=train_anns
            # train_anns+=random.sample(augs, len(train_anns))
            self.augs = augs
            self.aug_idx = np.array([x['id'] for x in self.augs])
        
        if args.use_pseudo:
            pseudo=list()
            with open(args.pseudo_filepath,'r',encoding='utf8') as f:
                for line in f.readlines():
                    pseudo.append(json.loads(line))        
            # train_anns+=random.sample(pseudo, len(train_anns)//2)
            self.pseudo = pseudo
            X = np.arange(len(pseudo))
            fold_num = len(pseudo) // len(self.train_anns)
            y = [x['label_id'] for x in pseudo]
            
            def ret_stratified_split_validx(X,y,fold_num):
                from sklearn.model_selection import StratifiedKFold
                kf = StratifiedKFold(n_splits=fold_num, random_state=args.seed, shuffle=True)
                splits = []
                for tr, va in kf.split(X, y):
                    splits.append(list(va))
                return splits
            
            self.pseudo_split_idx = ret_stratified_split_validx(X,y,fold_num)
            self.pseudo_conter = 0
            self.pseudo_method = 'stratified'
        
        # since valdata is not changing during training, its loader can be made here 
        val_dataset = BaseModelDataset(self.args, self.val_anns, method=self.args.seq_concat_method)
        val_sampler = SequentialSampler(val_dataset)
        self.val_dataloader = DataLoader(val_dataset,
                                    batch_size=args.val_batch_size,
                                    sampler=val_sampler,
                                    drop_last=False,
                                    pin_memory=True,
                                    num_workers=args.num_workers,
                                    prefetch_factor=args.prefetch)
        self.__samplesize_per_cls()

    def __samplesize_per_cls(self):
        anns = self.train_anns + self.val_anns
        labels = [x['label_id'] for x in anns]
        # print(len(labels))
        _, self.samplesize_per_cls = np.unique(labels, return_counts=True)

    def resampler(self, data, samplesize_per_cls, verbose=True):       
        if not self.resmaple:
            return data
        else:
            # there is 3 methods for sampling: smoothing, prob_inverse, uniform
            probs = probability_smoothing(samplesize_per_cls, method=self.sample_method)
            samplesizes = [int(x*sum(samplesize_per_cls)) for x in probs]
            if verbose:
                a = [x['label_id'] for x in data]
                _, counts = np.unique(a, return_counts=True)
                print(f'original counts: {list(counts)}')
                print(f'reference counts: {list(samplesize_per_cls)}')
                print(f'resmapled counts: {samplesizes}')
            
            resampled_data = []
            for i, count in enumerate(samplesizes):
                b = [x for x in data if x['label_id']==i]
                resampled_data.extend(list(np.random.choice(b, size=count, replace=True)))
            return resampled_data

    def get(self, epoch):
        random.shuffle(self.train_anns)
        # resampler for unbalanced class
        train_this_epoch = self.resampler(self.train_anns, self.samplesize_per_cls)
        
        aug_dataloader, pseudo_dataloader = None, None
        train_dataset = BaseModelDataset(self.args, train_this_epoch, method=self.args.seq_concat_method)
        
        if epoch < self.args.warmup_epochs_w_only_oridat:    # for epoch one, 
            train_dataloader = DataLoader(train_dataset,
                                        batch_size=self.args.batch_size,
                                        drop_last=False,
                                        pin_memory=True,
                                        num_workers=self.args.num_workers,
                                        prefetch_factor=self.args.prefetch)
        else:
            pseudo_this_epoch, aug_this_epoch = [], []
            # warmup fcn to control number of samples
            f = lambda lo, hi: int((lo+(hi-lo)*min(1,(epoch-self.args.warmup_epochs_w_only_oridat)
                                                   /self.warmup_epochs)) * len(train_this_epoch))
            if self.args.use_pseudo:
                num_pseudo = f(self.pseudo_ratios[0], self.pseudo_ratios[1])
                if not self.resmaple:
                    if self.pseudo_method=='random':
                        pseudo_this_epoch=random.sample(self.pseudo, num_pseudo)
                    elif self.pseudo_method=='stratified':
                        num_sets = min(epoch+1-self.args.warmup_epochs_w_only_oridat, self.pseudo_ratios[1])
                        idxx = []
                        for _ in range(num_sets):
                            id = self.pseudo_split_idx[self.pseudo_conter]
                            idxx.extend(id)
                            self.pseudo_conter+=1
                            if self.pseudo_conter==len(self.pseudo_split_idx):
                                self.pseudo_conter=0
                                
                        pseudo_this_epoch = [self.pseudo[i] for i in idxx]  
                else:
                    r = num_pseudo/sum(self.samplesize_per_cls)
                    samplesizes = [int(r*x) for x in self.samplesize_per_cls]
                    pseudo_this_epoch = self.resampler(self.pseudo, samplesizes)

            if self.args.use_aug:
                tr_idx = [x['id'] for x in train_this_epoch]
                aug_this_epoch = []
                # make aug dataset in exact the same order as train dataset
                for id in tr_idx:
                    i = np.random.choice(np.where(self.aug_idx==id)[0], size=1)[0]
                    aug_this_epoch.append(self.augs[i])
            
            tot_num = len(train_this_epoch)+len(pseudo_this_epoch)+len(aug_this_epoch)
            # traindata is preshuffled, here 'shuffle' is forced to be false to keep matched order btw train and aug datasets
            train_dataloader = DataLoader(train_dataset,
                                        batch_size=int(self.args.batch_size*len(train_this_epoch)/tot_num),
                                        drop_last=False,
                                        pin_memory=True,
                                        shuffle=False,
                                        num_workers=self.args.num_workers,
                                        prefetch_factor=self.args.prefetch)
            if self.args.use_pseudo:
                pseudo_dataset = BaseModelDataset(self.args, pseudo_this_epoch, method=self.args.seq_concat_method)
                pseudo_dataloader = DataLoader(pseudo_dataset,
                                            batch_size=int(self.args.batch_size*len(pseudo_this_epoch)/tot_num),
                                            drop_last=False,
                                            pin_memory=True,
                                            shuffle=False,
                                            num_workers=self.args.num_workers,
                                            prefetch_factor=self.args.prefetch)
            if self.args.use_aug:
                aug_dataset = BaseModelDataset(self.args, aug_this_epoch, method=self.args.seq_concat_method)
                aug_dataloader = DataLoader(aug_dataset,
                                            batch_size=int(self.args.batch_size*len(aug_this_epoch)/tot_num),
                                            drop_last=False,
                                            pin_memory=True,
                                            shuffle=False,
                                            num_workers=self.args.num_workers,
                                            prefetch_factor=self.args.prefetch)           

        return train_dataloader, self.val_dataloader, aug_dataloader, pseudo_dataloader