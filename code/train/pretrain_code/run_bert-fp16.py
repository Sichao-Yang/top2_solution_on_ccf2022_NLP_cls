#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import absolute_import, division, print_function
import pretraining_args as args
import time
import logging
import os
import random
random.seed(args.seed)
import sys
from glob import glob
import numpy as np
import gc
import jieba
import collections
import math
import torch
from torch import nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import random
from random import randrange, randint, shuffle, choice, sample
from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

# from NEZHA.configuration_nezha import NeZhaConfig
# from NEZHA.modeling_nezha import NeZhaForMaskedLM



from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertConfig
from transformers.optimization import AdamW
import joblib


jieba.enable_parallel(10)
from transformers import get_linear_schedule_with_warmup
logger = logging.getLogger(__name__)
import warnings
warnings.filterwarnings("ignore")


# In[2]:


# seed_value = 666
def seed_all(seed_value):
    random.seed(seed_value) # Python
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False
seed_all(42)


# In[3]:


search_model = 'train_bert'
logger = logging.getLogger('Bert_train')
logger.setLevel(logging.DEBUG)
timestamp = time.strftime("%Y.%m.%d_%H.%M.%S", time.localtime())
fh = logging.FileHandler('log_{}.txt'.format(search_model))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s][%(levelname)s] ## %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)


# In[4]:


def get_word_signs(tokens):
    signs = []
    line = ''
    for token in tokens:
        if len(token) > 1:
            line += ' '
        else:
            line += token
    words = jieba.lcut(line)
    # 带##的会被当做单独词
    sign = 0
    # signs中 0 1连续交替，代表词的区分
    for word in words:
        for i in word:
            signs.append(sign)
        sign = 1 if sign == 0 else 0
    assert len(tokens) == len(signs)
    return signs


# In[5]:


device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

tokenizer = BertTokenizer.from_pretrained(f'{args.pretrained_path}')
train_examples = None
num_train_optimization_steps = None
vocab_words = []
with open(args.vocab_file, 'r') as fr:
     for line in fr:
        vocab_words.append(line.strip("\n"))
print(args.pretrain_train_path)


# In[6]:


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id



def create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, word_ids):
    
    cand_indexes = []
    word_indexes = []
    last_word_sign = 0
    for (i, token) in enumerate(tokens):
        # 特殊符号为 -1
        if token == "[CLS]" or token == "[SEP]" or word_ids[i] == -100:
            continue
        # (len(cand_indexes) >= 1 )
        # 之前的处理中，会把##的去掉
        # word_ids中连续0或者1为同一个词
        if word_ids[i] == last_word_sign:
            word_indexes.append(i)
        elif token.startswith("##"):
            word_indexes.append(i)
            last_word_sign = word_ids[i]
        else:
            # token不带## 且对应wordid不等则为另外一个词
            cand_indexes.append(word_indexes)
            word_indexes = []
            word_indexes.append(i)
            last_word_sign = word_ids[i]

    random.shuffle(cand_indexes)
    output_tokens = list(tokens)
    masked_lm = collections.namedtuple("masked_lm", ["index", "label"])  
    num_to_predict = min(max_predictions_per_seq,
                       max(1, int(round(len(tokens) * masked_lm_prob))))
    
    masked_lms = []
    covered_indexes = set()
    for word_indexes in cand_indexes:
        if str(word_indexes) in covered_indexes:
              continue
        covered_indexes.add(str(word_indexes))

        random1 = random.random()
        random2 = random.random()
        for index in word_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            masked_token = None
            # 80% of the time, replace with [MASK]
            if random1 < 0.8:
                 masked_token = "[MASK]"
            else:
        # 10% of the time, keep original
               if random2 < 0.5:
                   masked_token = tokens[index]
        # 10% of the time, replace with random word
               else:
                  masked_token = vocab_words[random.randint(0, len(vocab_words) - 1)]

            output_tokens[index] = masked_token
            masked_lms.append(masked_lm(index=index, label=tokens[index]))
            
    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)
    
    
    return output_tokens, masked_lm_positions, masked_lm_labels


def create_examples(data_path, tokenizer, max_seq_length, masked_lm_prob, max_predictions_per_seq, vocab_words):
    """Creates examples for the training and dev sets."""
#     vocab_check = {w:0 for w in vocab_words}
    examples = []
    max_num_tokens = max_seq_length - 2
    fr = open(data_path, "r")
    for (i, line) in tqdm(enumerate(fr), desc="Creating Example"):

        words_ids = []
        line = line.strip().split('---')
        title = line[0]
        assignee = line[1]
        abstract = line[2]
        label = line[3]

        words_ids = []
        tokens_title = tokenizer.tokenize(title)
        tokens_assignee= tokenizer.tokenize(assignee)
        tokens_abstract = tokenizer.tokenize(abstract)

        tokens = ["[CLS]"] + tokens_title + ["[SEP]"] + tokens_assignee + ["[SEP]"] + tokens_abstract + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        
        signs_title = get_word_signs(tokens_title)
        signs_assignee = get_word_signs(tokens_assignee)
        signs_abstract = get_word_signs(tokens_abstract)
        
        words_ids.append(-100)
        for sign in signs_title:
            words_ids.append(sign)
        words_ids.append(-100)

        for sign in signs_assignee:
            words_ids.append(sign)
        words_ids.append(-100)

        for sign in signs_abstract:
            words_ids.append(sign)
        words_ids.append(-100)

        tokens, masked_lm_positions, masked_lm_labels=create_masked_lm_predictions(
            tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, words_ids)  
        
        example = {
            "tokens": tokens,
            "segment_ids": segment_ids,
            "masked_lm_positions": masked_lm_positions,
            "masked_lm_labels": masked_lm_labels}
        examples.append(example)
    fr.close()
    return examples

def convert_examples_to_features(examples, max_seq_length, tokenizer):
    features = []
    for i, example in enumerate(examples):
        if i % 300000 == 0:
            print(f'{i} have finished!')
        tokens = example["tokens"]
        segment_ids = example["segment_ids"]
        masked_lm_positions = example["masked_lm_positions"]
        masked_lm_labels = example["masked_lm_labels"]
#         print(len(tokens), len(segment_ids), max_seq_length)
        assert len(tokens) == len(segment_ids) <= max_seq_length  # The preprocessed data should be already truncated

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        masked_label_ids = tokenizer.convert_tokens_to_ids(masked_lm_labels)

        input_array = np.zeros(max_seq_length, dtype=np.int)
        input_array[:len(input_ids)] = input_ids

        mask_array = np.zeros(max_seq_length, dtype=np.bool)
        mask_array[:len(input_ids)] = 1

        segment_array = np.zeros(max_seq_length, dtype=np.bool)
        segment_array[:len(segment_ids)] = segment_ids

        lm_label_array = np.full(max_seq_length, dtype=np.int, fill_value=-100)
        lm_label_array[masked_lm_positions] = masked_label_ids
        
        feature = InputFeatures(input_ids=input_array,
                         input_mask=mask_array,segment_ids=segment_array, label_id=lm_label_array)
        features.append(feature)
        # if i < 10:
        #     logger.info("input_ids: %s\ninput_mask:%s\nsegment_ids:%s\nlabel_id:%s" %(input_array, mask_array, segment_array, lm_label_array))
    return features
        
# train_features = convert_examples_to_features(train_examples, args.max_seq_length, tokenizer)


# In[7]:


if args.do_train:
    train_examples = create_examples(data_path=args.pretrain_train_path,
                                     tokenizer=tokenizer,
                                     max_seq_length=args.max_seq_length,
                                     masked_lm_prob=args.masked_lm_prob,
                                     max_predictions_per_seq=args.max_predictions_per_seq,
                                     vocab_words=vocab_words)
#     with open('./train_examples.pk', 'rb')as f:
#         train_examples = joblib.load(f)


    num_train_optimization_steps = int(
        math.ceil(len(train_examples) / args.train_batch_size) / args.gradient_accumulation_steps) * args.num_train_epochs
    num_train_optimization_steps = num_train_optimization_steps


# In[8]:


# import joblib
# with open('./train_examples.pk', 'wb')as f:
#     joblib.dump(train_examples, f)

# with open('./train_examples.pk', 'rb')as f:
#     train_examples = joblib.load(f)


# In[9]:


pre_trained_dict = torch.load(f'{args.pretrained_path}pytorch_model.bin')
model = BertForMaskedLM(config=BertConfig.from_json_file(args.bert_config_json))
# BertForMaskedLM
# model = NeZhaForMaskedLM(config=NeZhaConfig.from_json_file(args.bert_config_json))


# In[10]:


pre_trained_dict = collections.OrderedDict(pre_trained_dict)
# pre_trained_dict['cls.seq_relationship.weight']
pre_trained_dict.pop('cls.seq_relationship.weight')
pre_trained_dict.pop('cls.seq_relationship.bias')
n=0
model_init = dict(model.state_dict())
for k in pre_trained_dict:
    if k in model_init:
        n+=1
        model_init[k] = pre_trained_dict[k]
model.load_state_dict(model_init)


# In[11]:


model = model.cuda()
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
warmup_step = int(args.warmup_proportion * num_train_optimization_steps)
scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=warmup_step, 
    num_training_steps=int(num_train_optimization_steps)
)


# In[12]:

from torch.cuda import amp
scaler = amp.GradScaler()


# In[ ]:


# init epoch1:
train_features = convert_examples_to_features(train_examples, args.max_seq_length, tokenizer)
all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

global_step = 0
best_loss = 100000
tr_loss = 0
if args.do_train:
    
    logger.info('training.........................')
    model.train()
    #int(args.num_train_epochs)
    for epoch in range(int(args.num_train_epochs)):
        if epoch % 5 == 0 and epoch != 0:
            model_weights = dict(model.state_dict())
            model_weights = collections.OrderedDict(model_weights)
            torch.save(model_weights, f'./outputs/pytorch_model_{epoch}.bin')            
            
            
        if epoch % 20 == 0 and epoch != 0:
            train_examples = create_examples(data_path=args.pretrain_train_path,
                                 tokenizer=tokenizer,
                                 max_seq_length=args.max_seq_length,
                                 masked_lm_prob=args.masked_lm_prob,
                                 max_predictions_per_seq=args.max_predictions_per_seq,
                                 vocab_words=vocab_words)
            
        
            train_features = convert_examples_to_features(train_examples, args.max_seq_length, tokenizer)
            all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
            all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
#         train_sampler = RandomSampler(train_data)
#         train_dataloader = DataLoader(train_data, sampler=train_sampler,num_workers=4, batch_size=args.train_batch_size)
#         train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, num_replicas=2)
        print('train_data len:', len(train_data))
        train_dataloader = torch.utils.data.DataLoader(train_data,
                                         num_workers=4,
                                         batch_size=args.train_batch_size)
        print('one epoch steps: ', len(train_dataloader))
        
        train_loss = []
        nb_tr_examples, nb_tr_steps = 0, 0
        tk0 = tqdm(train_dataloader, total=len(train_dataloader))
        for step, batch in enumerate(tk0):
#             if nb_tr_steps > 0 and nb_tr_steps % 1000 == 0:
#                 logger.info("=====-epoch %d -train_step %d -train_loss %.4f\n" % (epoch,
#                                                                                   nb_tr_steps,
#                                                                                   np.mean(train_loss)))
                
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            # masked_lm_loss
            loss = model(input_ids=input_ids, 
                         token_type_ids=segment_ids, 
                         attention_mask=input_mask, 
                         labels=label_ids)[0]
#             loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            scaler.scale(loss).backward()
#             loss.backward()

            train_loss.append(loss.item()*args.gradient_accumulation_steps)
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % args.gradient_accumulation_steps==0 or step==len(train_dataloader) - 1:
#                 optimizer.step()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                
            tk0.set_postfix(epoch_loss=np.mean(train_loss))
    model_weights = dict(model.state_dict())
#         model_weights.pop('cls.seq_relationship.weight')
#         model_weights.pop('cls.seq_relationship.bias')
    model_weights = collections.OrderedDict(model_weights)
    #collections.OrderedDict
    torch.save(model_weights, f'./outputs/pytorch_model_{epoch}.bin')
#         gc.collect()


# In[ ]:


# class InputFeatures(object):
#     """A single set of features of data."""
#     def __init__(self, input_ids, input_mask, segment_ids, label_id):
#         self.input_ids = input_ids
#         self.input_mask = input_mask
#         self.segment_ids = segment_ids
#         self.label_id = label_id
        
# InputFeatures(input_ids=1,input_mask=2,segment_ids=3, label_id=4)


# In[ ]:




