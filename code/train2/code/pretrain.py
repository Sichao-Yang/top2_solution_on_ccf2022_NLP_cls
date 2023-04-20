# -*- coding: utf-8 -*-
import json
from torch.utils.data import DataLoader, Dataset
import torch
from transformers import (AutoTokenizer, DataCollatorForLanguageModeling, Trainer, 
    AutoModelForMaskedLM, TrainingArguments, PrinterCallback)
import pandas as pd
# import evaluate
import numpy as np
from os import path as osp
import sys
sys.path.append(osp.abspath(osp.dirname(__file__)))
from config import parse_args
from logger import get_logger
from plots import lineplot
from utils import seed_everything

args = parse_args()
output_dir = args.output_dir
model_name = ['chinese-roberta-wwm-ext','macbert_base','Nezha_cn_base', 'ernie3_base']
model_name = model_name[3]
model_dir = f'data/model/pytorch/{model_name}'

DEBUG = False


def read_jsonfile(file_name):
    data = []
    with open(file_name, encoding='utf-8') as f:
        data = json.loads(f.read(), strict=False)
    return data

class PDataset(Dataset):
    def __init__(self, df, tokenizer, data_collator, max_length=125):
        super().__init__()
        self.df = df.reset_index(drop=True)
        # maxlen allowed by model config
        self.tokenizer = tokenizer
        self.data_collator = data_collator
        self.max_length = max_length

    def __getitem__(self, index):
        row = self.df.iloc[index]
        doc = row.source
        inputs = {}
        try:
          doc_id = tokenizer(doc, truncation=True, max_length=self.max_length)
          doc_id = self.data_collator([doc_id])
          inputs['input_ids'] = doc_id['input_ids'][0].tolist()
          inputs['labels'] = doc_id['labels'][0].tolist()       
          if 'token_type_ids' in inputs:
            inputs['token_type_ids'] = [0] * len(inputs['input_ids'])
        except:
          print('*'*20)
          print(doc)
          print('*'*20)
          
        return inputs

    def __len__(self):
        return self.df.shape[0]


class CustomCallback(PrinterCallback):
    fp = osp.join(output_dir, f'pretrain_{model_name}.log')
    def __init__(self):
        self.logging = get_logger(filename=self.fp)
        super().__init__()

    def on_log(self, args, state, control, logs=None, **kwargs):
        self.logging.info(state.log_history[-1])
        
class CustomDataCollator:
    def __init__(self, mask_id):
        self.mask_id = mask_id
        
    def __call__(self, batch):
        return data_collator_p(batch, self.mask_id)

def data_collator_p(batch, mask_id):
    # padding & convert to tensor datatype
    max_length = max([len(i['input_ids']) for i in batch])
    input_id, token_type, labels = [], [], []
    for i in batch:
        input_id.append(i['input_ids'] + [mask_id]*(max_length-len(i['input_ids'])))
        #token_type.append(i['token_type_ids'] + [1] * (max_length - len(i['token_type_ids'])))
        labels.append(i['labels'] + [-100] * (max_length - len(i['labels'])))
    output={}
    output['input_ids'] = torch.as_tensor(input_id, dtype=torch.long)
    #output['token_type_ids'] = torch.as_tensor(token_type, dtype=torch.long)
    output['labels'] = torch.as_tensor(labels, dtype=torch.long)
    return output


if __name__=='__main__':
    
    dat = pd.DataFrame(read_jsonfile("data/invent_patent.json"))
    dat = dat[['pat_name', 'pat_applicant', 'pat_summary']]
    dat = dat.dropna()
    dat['source'] = dat['pat_name'] +'。'+ dat['pat_applicant'] +'。'+ dat['pat_summary']
    dat = dat[['source']]
    per_device_train_batch_size = 160
    num_train_epochs = 20
    logging_steps = 50
    learning_rate = 5e-5
    split_ratio = 0.9
    save_total_limit = num_train_epochs
    gradient_accumulation_steps = 8
    if DEBUG:
        dat = dat[:300]
        per_device_train_batch_size = 10
        num_train_epochs = 3
        logging_steps = 2

    seed_everything(42)
    dat = dat.sample(frac=1, random_state=42).reset_index(drop=True)
    idx = int(len(dat)*split_ratio)
    train_dat, eval_dat = dat[:idx], dat[idx:]
    if 'Nezha' in model_name:
        from transformers import BertTokenizer, NezhaModel, NezhaForPreTraining
        tokenizer = BertTokenizer.from_pretrained(model_dir)
        # model = NezhaModel.from_pretrained(model_dir)
        model = NezhaForPreTraining.from_pretrained(model_dir)
        per_device_train_batch_size-=40
    elif 'ernie' in model_name:
        from transformers import BertTokenizer, ErnieForMaskedLM
        tokenizer = BertTokenizer.from_pretrained(model_dir)
        model = ErnieForMaskedLM.from_pretrained(model_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForMaskedLM.from_pretrained(model_dir)
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    out_dir = 'data/model/pretrained'
    training_args = TrainingArguments(
        output_dir=out_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        save_total_limit=save_total_limit,
        save_strategy='epoch',
        learning_rate=learning_rate,
        # fp16=True,
        warmup_ratio=0.05,
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_steps=logging_steps,
        evaluation_strategy="steps",
        label_names=['labels'],
    )
    
    dataset = PDataset(train_dat, tokenizer, data_collator, max_length=150)
    test_dataset = PDataset(eval_dat, tokenizer, data_collator, max_length=150)
    data_collator_custom = CustomDataCollator(tokenizer.mask_token_id)
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator_custom,
        train_dataset=dataset,
        eval_dataset=test_dataset,
        callbacks=[CustomCallback],
    )
    trainer.train()
    eval_loss = [x['eval_loss'] for x in trainer.state.log_history if 'eval_loss' in x.keys()]
    train_loss = [x['loss'] for x in trainer.state.log_history if 'loss' in x.keys()]
    
    
    xs = [[*range(len(train_loss))],[*range(len(train_loss))]]
    lineplot(xs=xs, 
             ys=[train_loss, eval_loss], 
             title='loss plot',
             xlabel=f'per {logging_steps} steps', 
             ylabel='loss',
             legend=['train', 'eval'],
             save_path=osp.join(output_dir, f'pretrain_loss_plot_{model_name}.png'))
    
    trainer.save_model(out_dir)
