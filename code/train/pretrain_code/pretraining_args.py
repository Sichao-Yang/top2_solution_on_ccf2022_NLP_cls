# -----------ARGS---------------------
pretrain_train_path = "./abstracts_new.txt"
# pretrain_train_path = "./abstracts_new_test.txt"
# pretrain_dev_path = "data/pretrain_dev.txt"

max_seq_length = 512
do_train = True
do_lower_case = True
# 在训练过程中将模拟该BS进行梯度累计，train的过程bs=这里的bs/grad_step
train_batch_size = 256
eval_batch_size = 32
learning_rate = 3e-5
num_train_epochs = 5
warmup_proportion = 0.1
no_cuda = False
local_rank = -1
seed = 412
gradient_accumulation_steps = 16
fp16 = False
loss_scale = 0.
# bert_config_json = "/root/autodl-tmp/retrained_models/NEZHA-base-wwm/config.json"
# vocab_file = "/root/autodl-tmp/retrained_models/NEZHA-base-wwm/vocab.txt"

# pretrained_path = "/root/autodl-tmp/pretrained_models/chinese_roberta_wwm_ext/"
pretrained_path = '/root/autodl-tmp/CCF-小样本/Nezha_pytorch/nezha_model/'

# pretrained_path = "./ori_model/"


bert_config_json = f"{pretrained_path}config.json"
vocab_file = f"{pretrained_path}vocab.txt"

output_dir = "./outputs"
masked_lm_prob = 0.15
max_predictions_per_seq = 60