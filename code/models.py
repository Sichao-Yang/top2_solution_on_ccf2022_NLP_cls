import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.losses import LabelSmoothingCrossEntropy, FocalLoss, LabelSM_Focal, DiceLoss, LabelSmoothingCrossEntropyWeight
from torch.nn import CrossEntropyLoss
from layers.adversarial import FGM, PGD, Lookahead

from NEZHA.modeling_nezha import NeZhaModel
from NEZHA.configuration_nezha import NeZhaConfig
# from transformers import NezhaModel
# from transformers import NezhaConfig

from transformers import BertConfig, BertModel
import logging
import warnings
import transformers
transformers.logging.set_verbosity_error()
warnings.filterwarnings('ignore')


class MaskedGlobalMaxPool1D(nn.Module):
    
    def __init__(self, **kwargs):
        super(MaskedGlobalMaxPool1D, self).__init__(**kwargs)
        self.supports_masking = True       
        
    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return input_shape[:-2] + (input_shape[-1],)
    
    def forward(self,inputs,mask = None):
        if mask is not None:
            mask = mask.float()
            inputs = inputs - torch.unsqueeze((1.0-mask)*1e6,dim = -1)
        return torch.max(inputs,dim = -2).values
    

class MaskedGlobalAveragePooling1D(nn.Module):

    def __init__(self, **kwargs):
        super(MaskedGlobalAveragePooling1D, self).__init__(**kwargs)
        self.supports_masking = True
        
    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return input_shape[:-2] + (input_shape[-1],)
    
    def forward(self,inputs, mask=None):
        if mask is not None:
            mask = mask.float()
            mask = torch.unsqueeze(mask,dim = -1)
            inputs = inputs*mask
            return torch.sum(inputs,dim = 1)/torch.sum(mask,dim = 1)
        else:
            return torch.mean(inputs,dim = 1)


def reinit_layers(encoder, config, args):
    num_hidden_layers = config.num_hidden_layers
    num_reinit_layers = args.num_reinit_layers
    
    if args.reinit_pooler:
        for module in encoder.pooler.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
                module.bias.data.zero_()
        for p in encoder.pooler.parameters():
            p.requires_grad = True
    
    for layer in encoder.encoder.layer[-num_reinit_layers :]:
        for module in layer.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()  
    return encoder


class BertLastTwoCls(nn.Module):
    
    def __init__(self, args, nezha, load_pretrained, n_class=36):  ######
        super(BertLastTwoCls,self).__init__()
          
        if nezha:
            self.config = NeZhaConfig.from_pretrained(args.model_checkpoint, output_hidden_states=True)
            if load_pretrained:
                self.pretrained_model = NeZhaModel.from_pretrained(args.model_checkpoint, config = self.config)
            else:
                self.pretrained_model = NeZhaModel(config = self.config)
        else:
            self.config = BertConfig.from_pretrained(args.bert_model_checkpoint, output_hidden_states=True)
            if load_pretrained:
                self.pretrained_model = BertModel.from_pretrained(args.bert_model_checkpoint, config = self.config)
            else:
                self.pretrained_model = BertModel(config = self.config)
        self.loss_type = args.loss_type
        
        if args.num_reinit_layers > 0:
            self.pretrained_model = reinit_layers(self.pretrained_model, self.config, args)
        
        if args.num_reinit_layers > 0:
            self.pretrained_model = reinit_layers(self.pretrained_model, self.config, args)
        
        self.avg_pooling = MaskedGlobalAveragePooling1D()
        self.max_pooling = MaskedGlobalMaxPool1D()
        
        concat_dim = args.bert_dim * 3
        self.output = nn.Linear(concat_dim, n_class)
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])
        
        
    def forward(self, input_ids, input_segids, input_mask, input_lengths, input_labels=None):
                
        output = self.pretrained_model(
            input_ids, attention_mask=input_mask, token_type_ids=input_segids)
        

        sequence_output = output[0]
        pooler_output = output[1]
        hidden_states = output[2]
        
        concat_result = torch.cat(
            (hidden_states[-1][:, 0], hidden_states[-2][:, 0], pooler_output), dim=1)

                
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                h = self.output(dropout(concat_result))
            else:
                h += self.output(dropout(concat_result))
        h = h / len(self.dropouts)
        
        logits = h
        outputs = (h, )
        
        if input_labels is not None:
            assert self.loss_type in ['ls', 'focal', 'ce']
            if self.loss_type =='ls':
                loss_fct = LabelSmoothingCrossEntropy()
            elif self.loss_type == 'focal':
                loss_fct = FocalLoss()
            else:
                loss_fct = CrossEntropyLoss()
            
            loss = loss_fct(logits, input_labels)
        
            
            outputs = (loss,) + outputs
            
        return outputs
    

class BertLastFourCls(nn.Module):
    
    def __init__(self, args, nezha, load_pretrained, n_class=36):  ######
        super(BertLastFourCls,self).__init__()
        
        if nezha:
            self.config = NeZhaConfig.from_pretrained(args.model_checkpoint, output_hidden_states=True)
            if load_pretrained:
                self.pretrained_model = NeZhaModel.from_pretrained(args.model_checkpoint, config = self.config)
            else:
                self.pretrained_model = NeZhaModel(config = self.config)
        else:
            self.config = BertConfig.from_pretrained(args.bert_model_checkpoint, output_hidden_states=True)
            if load_pretrained:
                self.pretrained_model = BertModel.from_pretrained(args.bert_model_checkpoint, config = self.config)
            else:
                self.pretrained_model = BertModel(config = self.config)
        self.loss_type = args.loss_type
        
        if args.num_reinit_layers > 0:
            self.pretrained_model = reinit_layers(self.pretrained_model, self.config, args)
            
        
        self.avg_pooling = MaskedGlobalAveragePooling1D()
        self.max_pooling = MaskedGlobalMaxPool1D()
        
        concat_dim = args.bert_dim * 4
        self.output = nn.Linear(concat_dim, n_class)

        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])
        
        
    def forward(self, input_ids, input_segids, input_mask, input_lengths, input_labels=None):
                
        output = self.pretrained_model(
            input_ids, attention_mask=input_mask, token_type_ids=input_segids)

        sequence_output = output[0]
        pooler_output = output[1]
        hidden_states = output[2]
        
        concat_result = torch.cat(
            (hidden_states[-1][:, 0], hidden_states[-2][:, 0], hidden_states[-3][:, 0], hidden_states[-4][:, 0]), dim=1)

        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                h = self.output(dropout(concat_result))
            else:
                h += self.output(dropout(concat_result))
        h = h / len(self.dropouts)
        
        logits = h
        outputs = (h, )
        
        if input_labels is not None:
            assert self.loss_type in ['ls', 'focal', 'ce']
            if self.loss_type =='ls':
                loss_fct = LabelSmoothingCrossEntropy()
            elif self.loss_type == 'focal':
                loss_fct = FocalLoss()
            else:
                loss_fct = CrossEntropyLoss()
            
            loss = loss_fct(logits, input_labels)
        
            
            outputs = (loss,) + outputs
            
        return outputs
    
    
class BertLastFourEmbeddingsPooler(nn.Module):
    
    def __init__(self, args, nezha=True, load_pretrained=False, n_class=36):
        super().__init__()
        
        if nezha:
            self.config = NeZhaConfig.from_pretrained(args.model_checkpoint, output_hidden_states=True)
            if load_pretrained:
                self.pretrained_model = NeZhaModel.from_pretrained(args.model_checkpoint, config = self.config)
            else:
                self.pretrained_model = NeZhaModel(config = self.config)
        else:
            self.config = BertConfig.from_pretrained(args.bert_model_checkpoint, output_hidden_states=True)
            if load_pretrained:
                self.pretrained_model = BertModel.from_pretrained(args.bert_model_checkpoint, config = self.config)
            else:
                self.pretrained_model = BertModel(config = self.config)
        self.loss_type = args.loss_type
        
        if args.num_reinit_layers > 0:
            self.pretrained_model = reinit_layers(self.pretrained_model, self.config, args)
        
        self.avg_pooling = MaskedGlobalAveragePooling1D()
        self.max_pooling = MaskedGlobalMaxPool1D()
        
        concat_dim = args.bert_dim * 4
        self.output = nn.Linear(concat_dim, n_class)
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])
        
    def forward(self, input_ids, input_segids, input_mask, input_lengths, input_labels=None):
                
        outputs = self.pretrained_model(
            input_ids, attention_mask=input_mask, token_type_ids=input_segids)
        

        sequence_output = outputs[0]
        pooler_output = outputs[1]
        hidden_states = outputs[2]
        
        hidden_states1 = self.avg_pooling(hidden_states[-1], input_mask)
        hidden_states2 = self.avg_pooling(hidden_states[-2], input_mask)
        hidden_states3 = self.avg_pooling(hidden_states[-3], input_mask)
        hidden_states4 = self.avg_pooling(hidden_states[-4], input_mask)       
        
        concat_result = torch.cat((hidden_states1, hidden_states2, hidden_states3, hidden_states4), dim=1)
                
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                h = self.output(dropout(concat_result))
            else:
                h += self.output(dropout(concat_result))
        h = h / len(self.dropouts)
        
        logits = h
        outputs = (h, )
        
        if input_labels is not None:
            assert self.loss_type in ['ls', 'focal', 'ce', 'lsw']
            if self.loss_type =='ls':
                loss_fct = LabelSmoothingCrossEntropy()
            elif self.loss_type == 'focal':
                loss_fct = FocalLoss()
            elif self.loss_type == 'lsw':
                loss_fct = LabelSmoothingCrossEntropyWeight()
            else:
                loss_fct = CrossEntropyLoss()
            
            loss = loss_fct(logits, input_labels)
        
            
            outputs = (loss,) + outputs
            
        return outputs
    
    
class BertDynEmbeddings(nn.Module):
    
    def __init__(self, args, nezha, load_pretrained, n_class=36):  ######
        super(BertDynEmbeddings,self).__init__()
        
        if nezha:
            self.config = NeZhaConfig.from_pretrained(args.model_checkpoint, output_hidden_states=True)
            if load_pretrained:
                self.pretrained_model = NeZhaModel.from_pretrained(args.model_checkpoint, config = self.config)
            else:
                self.pretrained_model = NeZhaModel(config = self.config)
        else:
            self.config = BertConfig.from_pretrained(args.bert_model_checkpoint, output_hidden_states=True)
            if load_pretrained:
                self.pretrained_model = BertModel.from_pretrained(args.bert_model_checkpoint, config = self.config)
            else:
                self.pretrained_model = BertModel(config = self.config)
        self.loss_type = args.loss_type
                
        if args.num_reinit_layers > 0:
            self.pretrained_model = reinit_layers(self.pretrained_model, self.config, args)
        self.loss_type = loss_type
        self.avg_pooling = MaskedGlobalAveragePooling1D()
        self.max_pooling = MaskedGlobalMaxPool1D()
        
        
        self.dynWeight = nn.Linear(args.bert_dim, 1)
        self.dence = nn.Linear(args.bert_dim, 512)
        self.hidden_size = args.bert_dim
        
        concat_dim = args.bert_dim * 1
        self.output = nn.Linear(concat_dim, n_class)
        self.dropout = nn.Dropout(0.2)
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])
        
    def forward(self, input_ids, input_segids, input_mask, input_lengths, input_labels=None):

        output = self.pretrained_model(
            input_ids, attention_mask=input_mask, token_type_ids=input_segids)
        

        sequence_output = output[0]
        pooler_output = output[1]
        hidden_states = output[2]
        batch_size = pooler_output.shape[0]

        hid_avg_list = None
        weight_list = None
        for i, hidden in enumerate(hidden_states[-6: ]):

            hid_avg = self.avg_pooling(hidden_states[-(i + 1)], input_mask)
            weight = self.dynWeight(hid_avg).repeat(
                1, self.hidden_size)
            if hid_avg_list is None:
                hid_avg_list = hid_avg
            else:
                hid_avg_list = torch.cat((hid_avg_list, hid_avg), dim=1)

            if weight_list is None:
                weight_list = hid_avg
            else:
                weight_list = torch.cat((weight_list, weight), dim=1)

        concat_out = weight_list.mul_(hid_avg_list)
        concat_out = concat_out.reshape(
            batch_size, -1, self.hidden_size)
        concat_result = torch.sum(concat_out, dim=1)
                
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                h = self.output(dropout(concat_result))
            else:
                h += self.output(dropout(concat_result))
        h = h / len(self.dropouts)
        
        logits = h
        outputs = (h, )
        
        if input_labels is not None:
            assert self.loss_type in ['ls', 'focal', 'ce']
            if self.loss_type =='ls':
                loss_fct = LabelSmoothingCrossEntropy()
            elif self.loss_type == 'focal':
                loss_fct = FocalLoss()
            else:
                loss_fct = CrossEntropyLoss()
            
            loss = loss_fct(logits, input_labels)
        
            
            outputs = (loss,) + outputs
            
        return outputs
    
    
class BertRNN(nn.Module):
    
    def __init__(self, args, nezha, load_pretrained, n_class=36):  ######
        super(BertRNN,self).__init__()   

        if nezha:
            self.config = NeZhaConfig.from_pretrained(args.model_checkpoint, output_hidden_states=True)
            if load_pretrained:
                self.pretrained_model = NeZhaModel.from_pretrained(args.model_checkpoint, config = self.config)
            else:
                self.pretrained_model = NeZhaModel(config = self.config)
        else:
            self.config = BertConfig.from_pretrained(args.bert_model_checkpoint, output_hidden_states=True)
            if load_pretrained:
                self.pretrained_model = BertModel.from_pretrained(args.bert_model_checkpoint, config = self.config)
            else:
                self.pretrained_model = BertModel(config = self.config)
        self.loss_type = args.loss_type
        
        if args.num_reinit_layers > 0:
            self.pretrained_model = reinit_layers(self.pretrained_model, self.config, args)
            
        self.loss_type = loss_type
        self.avg_pooling = MaskedGlobalAveragePooling1D()
        self.max_pooling = MaskedGlobalMaxPool1D()
        
        
        self.rnn = nn.LSTM(args.bert_dim,
                           hidden_size=args.bert_dim,
                           num_layers=2,
                           bidirectional=True,
                           batch_first=True,
                           dropout=0.1)
        
        concat_dim = args.bert_dim * 4
        self.output = nn.Linear(concat_dim, n_class)
        
        self.dropout = nn.Dropout(0.2)
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])
        
    def forward(self, input_ids, input_segids, input_mask, input_lengths, input_labels=None):

        output = self.pretrained_model(
            input_ids, attention_mask=input_mask, token_type_ids=input_segids)
        

        sequence_output = output[0]
        pooler_output = output[1]
        hidden_states = output[2]
        batch_size = pooler_output.shape[0]

        sequence_output = (hidden_states[-1] + hidden_states[-2]) / 2
        output, hidden = self.rnn(sequence_output)
        rnn_avg = self.avg_pooling(output, input_mask)
        rnn_max = self.max_pooling(output, input_mask)
        concat_result = torch.cat((rnn_avg, rnn_max), dim=1)
                
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                h = self.output(dropout(concat_result))
            else:
                h += self.output(dropout(concat_result))
        h = h / len(self.dropouts)
        
        logits = h
        outputs = (h, )
        
        if input_labels is not None:
            assert self.loss_type in ['ls', 'focal', 'ce']
            if self.loss_type =='ls':
                loss_fct = LabelSmoothingCrossEntropy()
            elif self.loss_type == 'focal':
                loss_fct = FocalLoss()
            else:
                loss_fct = CrossEntropyLoss()
            
            loss = loss_fct(logits, input_labels)
        
            
            outputs = (loss,) + outputs
            
        return outputs
    