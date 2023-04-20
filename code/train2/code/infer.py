import torch
from torch.utils.data import SequentialSampler, DataLoader
import os
import numpy as np
from config import parse_args
from model import Model_v1 as Model
from tqdm import tqdm 
import json
from os import path as osp
import sys
sys.path.append(osp.abspath(osp.dirname(__file__)))
from data_helper import BaseModelDataset

DEBUG = False

def get_cvf1(path, fold_num, logging=None):
    assert logging is not None, 'please pass logger in'
    best_scores = []
    for cv in range(fold_num):
        thiscv = [x for x in os.listdir(path) if f'cv{cv}' in x]
        f1s = np.array([eval('0.'+x.split('_')[-1].split('.')[1]) for x in thiscv])
        logging.info(f'cv{cv},{max(f1s)}')
        best_scores.append(max(f1s))
    return best_scores


def voting(logging, args, cv_list):
    res = {}   
    for i, cv in enumerate(cv_list):
        with open(osp.join(args.output_dir, f'submit_cv{cv}.csv'), 'r') as f:
            line = f.readline()
            # print(line)
            while True:
                line = f.readline()
                if len(line)==0:
                    break
                id,label = line.split(',')
                if i==0:
                    res[id] = [eval(label.strip())]
                else:
                    res[id].append(eval(label.strip()))
        
    with open(osp.join(args.output_dir, f'submit_ensemble.csv'), 'w') as f:
        f.write(f'id,label\n')
        for id, labels in res.items():
            f.write(f'{id},{max(labels, key=labels.count)}\n')    
    
    if args.use_pseudo:
        # make pseudo labels
        pseudo = {}
        counter = 0
        for id, labels in res.items():
            label = max(labels, key=labels.count)
            if sum([1 for x in labels if x==label]) >= int(args.pseudo_confidence_threshold*len(cv_list)):
                pseudo[id] = label
                counter+=1
        logging.info(f"selected pseudo labels: {counter}/{len(res)}")
        ids = pseudo.keys()
        
        anns=list()
        with open(args.test_filepath,'r',encoding='utf8') as f:
            for line in f.readlines():
                ann =json.loads(line)
                if ann['id'] in ids:
                    ann['label_id'] = pseudo[ann['id']]
                    anns.append(ann)
        # print(len(anns))
        with open(osp.join(args.output_dir, f'pseudo_labels.json'), 'w', encoding='utf8') as f:
            for n in anns:
                f.writelines(json.dumps(n, ensure_ascii=False)+'\n')


def prob_merging(logging, args, cv_list, cv_scores, temperature=1):
    def softmax(arr, temp=1):
        sum_z = np.sum(np.exp(arr/temp), axis=1).reshape(-1,1)
        return np.exp(arr/temp)/sum_z
        
    selected_cv_scores = np.array([cv_scores[cv] for cv in cv_list]).reshape(1,-1)
    weights = softmax(selected_cv_scores, temperature).squeeze().tolist()
    for i, cv in enumerate(cv_list):
        with open(osp.join(args.output_dir, f'prob_cv{cv}.npy'), 'rb') as f:
            if i==0:
                prob = softmax(np.load(f))*weights[i]
            else:
                prob += softmax(np.load(f))*weights[i]
    
    labels = np.argmax(prob, axis=1)
    label_confience = np.max(prob, axis=1)
    
    anns=list()
    with open(args.test_filepath,'r',encoding='utf8') as f:
        for line in f.readlines():
            ann =json.loads(line)
            anns.append(ann)
    if DEBUG:
        anns = anns[:300]
    
    pseudo = {}
    counter = 0
    with open(osp.join(args.output_dir, f'submit_ensemble.csv'), 'w') as f:
        f.write(f'id,label\n')
        for label, ann, lc in zip(labels, anns, label_confience):
            id = ann['id']
            if args.use_pseudo and lc > args.pseudo_confidence_threshold:
                pseudo[id] = int(label)
                counter+=1
            f.write(f'{id},{label}\n')
    logging.info("Finished ensembling!")

    if args.use_pseudo:                
        logging.info(f"selected pseudo labels: {counter}/{len(labels)}")
        ids = pseudo.keys()
        anns=list()
        with open(args.test_filepath,'r',encoding='utf8') as f:
            for line in f.readlines():
                ann =json.loads(line)
                if ann['id'] in ids:
                    ann['label_id'] = pseudo[ann['id']]
                    anns.append(ann)
        # print(len(anns))
        with open(osp.join(args.output_dir, f'pseudo_labels.json'), 'w') as f:
            for n in anns:
                f.writelines(json.dumps(n, ensure_ascii=False)+'\n')


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
        
    # for name, param in model.named_parameters():
    #     if 'embeddings.weight' in name:
    #         print(name)
    #         print(param)
        
    #     if 'classifier.weight' in name:
    #         print(name)
    #         print(param)

    if torch.cuda.is_available():
        model = model.cuda()
    
    model.dropouts = [torch.nn.Dropout(0.0) for _ in range(args.multi_do)]
    
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
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        # 3. inference
        model.eval()
        predictions = []
        probs = np.zeros((len(anns), args.class_num))
        start_idx = 0
        if not recover_from_npy:
            with torch.no_grad():
                for batch in tqdm(dataloader):
                    batch = {k:v.to('cuda') for k,v in batch.items()}
                    pred_label_id, out, _ = model(data=batch)
                    predictions.extend(pred_label_id)
                    probs[start_idx:start_idx+len(out)] = out.cpu().numpy()
                    start_idx+=len(out)
        else:
            with open(osp.join(args.output_dir, f'prob_cv{cv}.npy'), 'rb') as f:
                prob = np.load(f)
                predictions = np.argmax(prob, axis=1)
            
        # if args.ensemble_method == 'voting':
        # 4. dump results
        with open(osp.join(args.output_dir, f'submit_cv{cv}.csv'), 'w') as f:
            f.write(f'id,label\n')
            for pred_label_id, ann in zip(predictions, dataset.anns):
                id = ann['id']
                f.write(f'{id},{pred_label_id}\n')
        with open(osp.join(args.output_dir, f'prob_cv{cv}.npy'), 'wb') as f:
            np.save(f, probs)
        logging.info(f'Finished inference for cv {cv}')


def reduce_cv(cv_scores, method='median', ratio=0.5):
    if method=='median':
        return [i for i,x in enumerate(cv_scores) if x>np.median(cv_scores)]
    elif method=='knn':
        from sklearn.cluster import KMeans
        X = np.array(cv_scores).reshape(-1,1)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
        selected_cluster = np.argmax(kmeans.cluster_centers_)
        y = kmeans.predict(X)
        return np.where(y==selected_cluster)[0]
    elif method=='ratio':
        remove_len = int(len(cv_scores)*ratio)
        import copy
        rank = copy.copy(cv_scores)
        rank.sort()
        rank = rank[remove_len:]
        return [i for i, x in enumerate(cv_scores) if x in rank]

def remove_probs(args):
    for cv in range(args.fold_num):
        os.remove(osp.join(args.output_dir, f'prob_cv{cv}.npy'))

def remove_labels(args):
    for cv in range(args.fold_num):
        os.remove(osp.join(args.output_dir, f'submit_cv{cv}.csv'))
        
def ensemble(args=None, logging=None, cv_confidence_masking=True):
    if args is None:
        args = parse_args()
    if logging is None:
        from logger import get_logger
        logging = get_logger(filename='tmp.log')
    cv_scores = get_cvf1(path=args.savedmodel_filepath, fold_num=args.fold_num, logging=logging)
    if cv_confidence_masking:
        # result ensemble: remove lower half of the cvs and majority vote for final results
        cv_list = reduce_cv(cv_scores, method=args.cv_masking_method)
    else:
        cv_list=[*range(args.fold_num)]
    logging.info(f'cv_confidence_maskining is {cv_confidence_masking}, the reduced cv_list is {cv_list}')
    logging.info(f'ensemble method is {args.ensemble_method}')
    if args.ensemble_method == 'voting':
        voting(logging, args, cv_list)
        # remove_labels(args)
    elif args.ensemble_method == 'prob_merging':
        prob_merging(logging, args, cv_list, cv_scores)
        # remove_probs(args)
 
if __name__ == '__main__':
    args = parse_args()
    args.output_dir = osp.join(args.output_dir, '20221108_22-47_final')
    
    m_list = ['', 'ema']
    out_dir = ['max_f1', 'ema']
    # m_list = ['ema']
    # out_dir = ['ema']
    out_root = args.output_dir
    for m, od in zip(m_list, out_dir):
        if m == '':
            continue
        
        args.savedmodel_filepath = osp.join(out_root, osp.basename(args.savedmodel_filepath), m)
        args.output_dir = osp.join(out_root, od)
        os.makedirs(args.output_dir, exist_ok=True)
        args.device = 'cuda'
        if DEBUG:
            args.test_filepath = args.test_filepath.replace('testA.json', 'testA_debug.json')
            args.fold_num=2
        
        inference(args)
        # inference(args, recover_from_npy=True)
        ensemble(args, cv_confidence_masking=args.cv_confidence_masking)