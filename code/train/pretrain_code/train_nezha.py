# coding:utf-8
import numpy as np
import random
import os
random.seed(0)
np.random.seed(0)#seed应该在main里尽早设置，以防万一
os.environ['PYTHONHASHSEED'] =str(0)#消除hash算法的随机性
import transformers as _
from transformers1 import Trainer, TrainingArguments,BertTokenizer
from NLP_Utils import MLM_Data,train_data,blockShuffleDataLoader

from NEZHA.configuration_nezha import NeZhaConfig
from NEZHA.modeling_nezha import NeZhaForMaskedLM

maxlen=510
batch_size=16
file_dir = '/root/autodl-tmp/CCF-小样本/Nezha_pytorch/nezha_model/'
tokenizer = BertTokenizer.from_pretrained(file_dir)
config = NeZhaConfig.from_pretrained(file_dir)
# config = NeZhaConfig(
#     vocab_size=len(tokenizer),
#     hidden_size=768,
#     num_hidden_layers=12,
#     num_attention_heads=12,
#     max_position_embeddings=512,
# )

# nezha-cn-base初始权重需要从Github下载https://github.com/lonePatient/NeZha_Chinese_PyTorch
model = NeZhaForMaskedLM.from_pretrained(file_dir)

# model.resize_token_embeddings(len(tokenizer))
# # 只训练word_embedding。能缩短两倍的训练时间
# for name, p in model.named_parameters():
#     if name != 'bert.embeddings.word_embeddings.weight':
#         p.requires_grad = False
# print(model)
# print('train_data:', train_data)

train_MLM_data=MLM_Data(train_data, maxlen, tokenizer)

#自己定义dataloader，不要用huggingface的
dl=blockShuffleDataLoader(train_MLM_data,None,key=lambda x:len(x[0])+1,shuffle=False
                          ,batch_size=batch_size,collate_fn=train_MLM_data.collate)

training_args = TrainingArguments(
    output_dir='/root/autodl-tmp/CCF-小样本/Nezha_pytorch/nezha_model/outputs',# 此处必须是绝对路径
    overwrite_output_dir=True,
    num_train_epochs=40,
    per_device_train_batch_size=batch_size,
    save_steps=len(dl)*5,#每5个epoch save一次
    save_total_limit=3,
    logging_steps=len(dl),#每个epoch log一次
    seed=2021,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=int(len(dl)*num_train_epochs*0.03),
    gradient_accumulation_steps = 4
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataLoader=dl,
    prediction_loss_only=True,
)

if __name__ == '__main__':
    trainer.train()
    trainer.save_model('./nezha_model')
