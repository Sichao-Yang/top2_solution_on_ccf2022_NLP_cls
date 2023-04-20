import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from NEZHA.modeling_nezha import NeZhaModel
from NEZHA.configuration_nezha import NeZhaConfig
from torch.nn import Dropout
import copy

class Model_v1(nn.Module):
    def __init__(self, args, last_n_layers=4, device='cuda', method='sep_merge', 
                 hidden_state=768, reinit_n_layers=3, remove_n_layers=4, 
                 freeze_n_layers=0, verbose=True):
        super().__init__()
        if "Nezha" in args.bert_dir:
            config = NeZhaConfig.from_pretrained(args.bert_dir, output_hidden_states=True)
            self.bert = NeZhaModel.from_pretrained(args.bert_dir, config = config)
            self.nezha = True
            # torch.save(self.bert.state_dict(), 'model.bin')
        else:
            self.bert = BertModel.from_pretrained(args.bert_dir, output_hidden_states=True)
        # config = BertConfig(output_hidden_states=True)
        # self.bert = BertModel(config=config)
        self.last_n_layers = last_n_layers
        self.classifier = nn.Linear(hidden_state*self.last_n_layers, args.class_num)
        # self.softmax = torch.nn.Softmax(dim=1)      # out dim: N, C, apply softmax on C dim
        if verbose:
            print("Init paras for cls:")
            for name, param in self.classifier.named_parameters():
                if param.requires_grad:
                    print(name, param.data)
        self.dropouts = [Dropout(args.do-0.1*i) for i in range(args.multi_do)]
        if method == 'sep_merge':
            self.weight_vector = torch.nn.Parameter(torch.FloatTensor(3,1))
            # Variable(torch.FloatTensor(1, 4), requires_grad=True)
        
        self.text_embedding = self.bert.embeddings
        self.device = device
        self.method = method
        if reinit_n_layers > 0:
            self._do_reinit(reinit_n_layers)
            if verbose: print(f'last {reinit_n_layers} layers of bert is reinitialized')       
        if remove_n_layers > 0:
            self._deleteEncodingLayers(remove_n_layers)
            if verbose: print(f'last {reinit_n_layers} layers of bert is removed')
        if freeze_n_layers > 0:
            self._freeze_n_layers(freeze_n_layers)
            if verbose: print(f'first {freeze_n_layers} layers of bert is freezed')
    
    def _freeze_n_layers(self, freeze_n_layers):
        # freeze bottom n layers with embeddings
        unfreeze_layers = [f'layer.{i}' for i in range(len(self.bert.encoder.layer)) if i >= freeze_n_layers]
        unfreeze_layers.extend(['pooler'])
        for name, param in self.bert.named_parameters():
            param.requires_grad = False
            for ele in unfreeze_layers:
                if ele in name:
                    param.requires_grad = True
                    break
    
    def _deleteEncodingLayers(self, remove_n_layers):  # must pass in the full bert model
        oldModuleList = self.bert.encoder.layer
        newModuleList = nn.ModuleList()
        total_layers = len(oldModuleList)
        num_layers_to_keep = total_layers - remove_n_layers
        # Now iterate over all layers, only keepign only the relevant bottom layers.
        for i in range(num_layers_to_keep):
            newModuleList.append(oldModuleList[i])
        self.bert.encoder.layer = newModuleList
       
    def _do_reinit(self, reinit_n_layers):
        # Re-init pooler.
        self.bert.pooler.dense.weight.data.normal_(mean=0.0, std=self.bert.config.initializer_range)
        self.bert.pooler.dense.bias.data.zero_()
        for param in self.bert.pooler.parameters():
            param.requires_grad = True
        # Re-init last n layers.
        for n in range(reinit_n_layers):
            self.bert.encoder.layer[-(n+1)].apply(self._init_weight_and_bias)
            
    def _init_weight_and_bias(self, module):                        
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.bert.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)   

    def build_pre_input(self, data):
        text_inputs = data['text_inputs'].to(self.device)
        text_mask = data['text_mask'].to(self.device)
        text_type_ids = data['text_type_ids'].to(self.device)
        textembedding = self.text_embedding(text_inputs, text_type_ids)
        return textembedding, text_mask
    
    def forward(self, data):
        inputs_embeds, mask = self.build_pre_input(data)
        emb = self.bert(attention_mask=mask, inputs_embeds=inputs_embeds)
        if self.nezha:
            from easydict import EasyDict as edict
            embeddings = edict(dict())
            embeddings.sequence_output = emb[0]
            embeddings.pooler_output = emb[1]
            embeddings.hidden_states = emb[2]
        else:
            embeddings = emb
        pooled_layers = [*range(-4,0)]      # by default last 4 layers are pooled
        if self.method == 'avg_seq_concat':
            hidden_stats = embeddings.hidden_states[pooled_layers[0]:]
            # expand mask [b,s] to [batch,seq,feat]
            mask_ = mask.unsqueeze(dim=-1)
            hidden_stats = [(mask_*hs).sum(dim=1) for hs in hidden_stats]
            # hidden_stats [layer][b,s,f] -> [layer][b,f]
            # normalize to mean value b [batch, seq] -> [b] -> [b,f]
            b = torch.sum(mask, dim=1).unsqueeze(dim=1)
            hidden_stats = [hs/b for hs in hidden_stats]
            pooled_output = torch.cat(hidden_stats,dim=1)
        # hidden_states dim: [num_layers][batch, seq, embedding_dim]
        elif self.method == 'first_concat':
            # here only the first token of seq [cls] is concatenated
            pooled_output = torch.cat(tuple([embeddings.hidden_states[i][:,0,:] for i in pooled_layers]), dim=-1)
        elif self.method == 'sep_merge':
            max_lengths= [30, 30+15-1, 30+15+450-2]
            tmp = list()
            def _avg_pool(tensor, weight, mask):
                return torch.sum(tensor*weight*mask, dim=1)/torch.sum(mask,dim=1)
            for i in pooled_layers:
                tmp.append(_avg_pool(embeddings.hidden_states[i][:,:max_lengths[0],:], self.weight_vector[0], mask[:, :max_lengths[0]].unsqueeze(-1)) \
                    + _avg_pool(embeddings.hidden_states[i][:,max_lengths[0]:max_lengths[1],:], self.weight_vector[1], mask[:, max_lengths[0]:max_lengths[1]].unsqueeze(-1)) \
                    + _avg_pool(embeddings.hidden_states[i][:,max_lengths[1]:max_lengths[2],:], self.weight_vector[2], mask[:, max_lengths[1]:max_lengths[2]].unsqueeze(-1))
                )
            pooled_output = torch.cat(tmp, dim=-1)
        pooled_outputs = [do(pooled_output) for do in self.dropouts]
        outs = [self.classifier(po) for po in pooled_outputs]
        # for name, param in self.named_parameters():
        #     if param.requires_grad == True:
        #         print(name)
        out = torch.mean(torch.stack(outs,dim=0), dim=0)
        return torch.argmax(out, dim=1), out, outs