import shutil
import numpy as np
import os
import torch
import json
from torch.cuda.amp import autocast as ac
from os import path as osp
from tqdm import tqdm
import sys
sys.path.append(osp.abspath(osp.dirname(__file__)))
from config import parse_args
from model import Model_v1 as Model
# os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'
from utils import (seed_everything, setup_device, build_optimizer, eval_fcn,
                   EMA, PGD, FGM, make_cvids, now, LossFcn)
from logger import get_logger
from infer import inference, ensemble
# from data_helper import Dataloaders
from data_helper import Dataloaders_v2 as Dataloaders
from plots import Expplots, ExpplotsLosses
from swa_utils import SWA

DEBUG = False

@torch.no_grad()
def validate(model, val_dataloader, ema, loss_fcn, epoch, global_step, logging):
    model.eval()
    predictions = []
    labels = []
    losses = []
    pbar = tqdm(total=int(len(val_dataloader.dataset)), ascii=True)
    pbar.set_description(f'Evaluation:')
    pseudo_b, aug_b = None, None 
    for batch in val_dataloader:
        bs = batch['text_inputs'].shape[0]
        label = batch['label'].squeeze(dim=1)
        pred_label_id, _, _ = model(batch)
        loss, _, _ = loss_fcn.run(model, batch, ema, pseudo_b, aug_b)
        loss = loss.mean()
        predictions.extend(pred_label_id.cpu().numpy())
        labels.extend(label)
        losses.append(loss.cpu().numpy())
        pbar.set_postfix(loss=f'{loss.item():.4f}')
        pbar.update(bs)
    pbar.close()
    results = eval_fcn(predictions, labels)
    results = {k: round(v, 4) for k, v in results.items()}
    logging.info(f"Epoch {epoch} step {global_step}: Eval loss {np.average(losses):.4f}, {results}")    
    return results['f1_macro'], np.average(losses)

def train(train_dataloader, aug_dataloader, pseudo_dataloader, model, optimizer, scheduler, 
          args, scaler, fgm, pgd, ema, epoch, logging, global_step, loss_fcn, swa, 
          expP2=None, grad_accum_steps=4):

    if grad_accum_steps!=0 and args.use_fp16:
        raise NotImplementedError('Gradient accumulation is not implemented with fp16!')

    def inner_work(batch, pseudo_b, aug_b, pbar, losses, accs, global_step, bs, epoch):
        if args.use_fp16:
            with ac():
                loss, acc, pred = loss_fcn.run(model, batch, ema, pseudo_b, aug_b)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
        else:         
            loss, acc, pred = loss_fcn.run(model, batch, ema, pseudo_b, aug_b)
            if grad_accum_steps!=0:
                loss = loss/(grad_accum_steps+1)
            loss.backward()
        if fgm is not None:
            fgm.attack()
            if args.use_fp16:
                with ac():
                    loss_adv, _, _ = loss_fcn.run(model, batch, ema, pseudo_b, aug_b)
            else:
                loss_adv, _, _ = loss_fcn.run(model, batch, ema, pseudo_b, aug_b)
            loss_adv = loss_adv.mean()
            if args.use_fp16:
                scaler.scale(loss_adv).backward()
            else:
                if grad_accum_steps!=0:
                    loss_adv = loss_adv/(grad_accum_steps+1)
                loss_adv.backward()
            fgm.restore()
        elif pgd is not None:
            pgd.backup_grad()
            pgd_k = pgd.steps
            for _t in range(pgd_k):
                pgd.attack(is_first_attack=(_t == 0))
                if _t != pgd_k - 1:
                    model.zero_grad()
                else:
                    pgd.restore_grad()
                if args.use_fp16:
                    with ac():
                        loss_adv, _, _ = loss_fcn.run(model, batch, ema, pseudo_b, aug_b)
                else:
                    loss_adv, _, _ = loss_fcn.run(model, batch, ema, pseudo_b, aug_b)
                loss_adv = loss_adv.mean()
                if args.use_fp16:
                    scaler.scale(loss_adv).backward()
                else:
                    if grad_accum_steps!=0:
                        loss_adv = loss_adv/(grad_accum_steps+1)
                    loss_adv.backward()
            pgd.restore()
        
        if args.use_fp16:
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
        else:
            if grad_accum_steps!=0:
                if ((global_step+1)%(grad_accum_steps+1)==0) or (batch_idx+1==data_len):
                    optimizer.step()
                    model.zero_grad()
            else:
                optimizer.step()
        model.zero_grad()
        if (args.use_ema and epoch+1>args.ema_start_epoch 
            and global_step%ema.update_every_n_steps==0):
            ema.update_teacher(global_step, epoch)
        # if args.use_swa and global_step+1>args.swa_start_step:
        if (args.use_swa and epoch+1>args.swa_start_epoch 
            and global_step%swa.update_every_n_steps==0):
            swa.update_parameters(model)
            swa.scheduler_step()
        else:
            scheduler.step()
        global_step += 1
       
        losses.append(loss.item())
        accs.append(100.*acc.item())
        if expP2 is not None:
            loss_dict = loss_fcn.last_run_losses
            expP2.update(loss_dict, global_step, tag=args.cv_id)
        
        pbar.set_postfix(loss=f'{loss.item():.4f}', acc=f'{100.*acc.item():.2f}%')
        pbar.update(bs)
        return losses, accs, pbar, losses, accs, global_step, pred
    
    model.train()
    losses, accs, preds, labels = [], [], [], []
    pseudo_b, aug_b = None, None
    if pseudo_dataloader is None and aug_dataloader is None:
        pbar = tqdm(total=int(len(train_dataloader.dataset)), ascii=True)
        pbar.set_description(f'Training Epoch {epoch}:')
        data_len = len(train_dataloader)
        for batch_idx, batch in enumerate(train_dataloader):
            bs = batch['text_inputs'].shape[0]
            labels.extend(list(batch['label'].squeeze(-1).numpy()))
            losses, accs, pbar, losses, accs, global_step, pred = inner_work(batch, pseudo_b, aug_b, pbar, 
                                                                             losses, accs, global_step, bs, epoch)
            preds.extend(list(pred.cpu().numpy()))
    elif pseudo_dataloader is not None and aug_dataloader is not None:
        pbar = tqdm(total=int(len(train_dataloader.dataset)+len(pseudo_dataloader.dataset)+len(aug_dataloader.dataset)), ascii=True)
        pbar.set_description(f'Training Epoch {epoch}:')
        data_len = len(train_dataloader)+len(pseudo_dataloader)+len(aug_dataloader)
        for batch_idx, (batch, pseudo_b, aug_b) in enumerate(zip(train_dataloader, pseudo_dataloader, aug_dataloader)):
            bs = batch['text_inputs'].shape[0] + pseudo_b['text_inputs'].shape[0] + aug_b['text_inputs'].shape[0]
            labels.extend(list(batch['label'].squeeze(-1).numpy()))
            losses, accs, pbar, losses, accs, global_step, pred = inner_work(batch, pseudo_b, aug_b, pbar, 
                                                                             losses, accs, global_step, bs, epoch)
            preds.extend(list(pred.cpu().numpy()))
    elif pseudo_dataloader is not None:
        pbar = tqdm(total=int(len(train_dataloader.dataset)+len(pseudo_dataloader.dataset)), ascii=True)
        pbar.set_description(f'Training Epoch {epoch}:')
        data_len = len(train_dataloader)+len(pseudo_dataloader)
        for batch_idx, (batch, pseudo_b) in enumerate(zip(train_dataloader, pseudo_dataloader)):
            bs = batch['text_inputs'].shape[0] + pseudo_b['text_inputs'].shape[0]
            labels.extend(list(batch['label'].squeeze(-1).numpy()))
            losses, accs, pbar, losses, accs, global_step, pred = inner_work(batch, pseudo_b, aug_b, pbar, 
                                                                             losses, accs, global_step, bs, epoch)
            preds.extend(list(pred.cpu().numpy()))
    elif aug_dataloader is not None:
        raise RuntimeError('aug data w/o pseudo is not implemented')
    pbar.close()
    logging.info(f"Training Epoch {epoch} Average train loss: {np.average(losses):.4f}, acc.: {np.average(accs):.2f}%")
    results = eval_fcn(preds, labels)
    results = {k: round(v, 4) for k, v in results.items()}
    logging.info(f"Training f1 scores: {results}")
    return global_step, results['f1_macro'], np.average(losses)
        

def train_loop(args, logging):
    logging.info(f'Training on CV No.{args.cv_id}:')
    # load data
    dataloaders = Dataloaders(args,total_epochs=args.max_epochs, resample=args.resample)
    # build model, optimizers, scheduler, lossFcn and expplots
    # avg_seq_concat, first_concat
    model = Model(args, device=args.device, 
                  reinit_n_layers=args.reinit_n_layers, 
                  remove_n_layers=args.remove_n_layers,
                  method=args.cls_method,
                  freeze_n_layers=args.freeze_n_layers)
    model = model.to(args.device)
    
    expP = Expplots(dir=args.output_dir)
    expP2 = ExpplotsLosses(dir=args.output_dir)
    optimizer, scheduler = build_optimizer(args, model)
    loss_fcn = LossFcn(sup_losstype=args.sup_losstype, 
                       device=args.device, 
                       temp=args.temp, 
                       unsup_losstype=args.unsup_losstype,
                       sup_lam=args.sup_lam,
                       unsup_lam=args.unsup_lam,
                       pseudo_lam=args.pseudo_lam,
                       sup2_losstype=args.sup2_losstype,
                       samplesize_per_cls=dataloaders.samplesize_per_cls)
    # setup training wrappers
    ema, fgm, pgd, scaler, swa = None, None, None, None, None
    if args.use_ema:
        ema = EMA(model, decay=args.ema_decay, max_epochs=args.max_epochs, 
                  update_every_n_steps=args.ema_update_every_n_steps)
    if args.use_swa:
        swa = SWA(model, optimizer, anneal_steps=args.swa_anneal_steps,
                  update_every_n_steps=args.swa_update_every_n_steps)
    if args.use_attack:
        if args.attack_method == 'fgm':
            fgm = FGM(model=model)
        elif args.attack_method == 'pgd':
            pgd = PGD(model=model)
        logging.info(f'Adversarial attack used, method: {args.attack_method}')
    if args.use_fp16:
        scaler = torch.cuda.amp.GradScaler()
    
    # training
    global_step, early_stop_counter = 0, 0
    best_score_tr, best_score_va = 0., 0.
    for epoch in range(args.max_epochs):
        # dynamically update datasets
        train_dataloader, val_dataloader, aug_dataloader, pseudo_dataloader = dataloaders.get(epoch)
        
        global_step, train_f1_macro, train_loss = train(train_dataloader, aug_dataloader, pseudo_dataloader, 
                                                    model, optimizer, scheduler, args, scaler, fgm, pgd, 
                                                    ema, epoch, logging, global_step, loss_fcn, swa, expP2, 
                                                    grad_accum_steps=args.grad_accum_steps)
        if train_f1_macro > best_score_tr:
            best_score_tr = train_f1_macro
        # validation
        eval_f1_macro, eval_loss = validate(model, val_dataloader, ema, loss_fcn, epoch, global_step, logging)
        # save best model
        if eval_f1_macro > best_score_va:
            best_score_va = eval_f1_macro
            early_stop_counter = 0
            if best_score_va > args.save_score_threshold:
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'f1_macro': eval_f1_macro},
                        f'{args.savedmodel_filepath}/best_model_cv{args.cv_id}_e{epoch}_f1{eval_f1_macro}.bin')
                logging.info(f'Epoch {epoch} - Saved Best Model (f1_macro: {best_score_va:.4f})')
        else:
            if best_score_va > args.save_score_threshold:
                early_stop_counter+=1
        
        if early_stop_counter > args.early_stop:
            logging.info(f'Early stop at epoch {epoch} due to training overfitting! '\
                    f'f1 score: tr-{best_score_tr:.4f}, val-{best_score_va:.4f}')
            break
        losses = {'train': train_loss, 'valid': eval_loss}
        aps = {'train': train_f1_macro, 'valid': eval_f1_macro}
        lr = optimizer.param_groups[0]['lr']
        expP.new_epoch_draw(aps, losses, lr, epoch, tag=str(args.cv_id))
    if ema is not None:
        torch.save({'epoch': epoch, 'model_state_dict': ema.teacher_model.state_dict()},
                f'{args.savedema_filepath}/ema_cv{args.cv_id}_e{epoch}_0.99.bin')
    
    if args.use_swa:
        swa.calc_bn(train_dataloader)
        torch.save({'epoch': epoch, 'model_state_dict': swa.model.state_dict()},
            f'{args.savedswa_filepath}/swa_cv{args.cv_id}_e{epoch}_0.99.bin')
    
    del model, train_dataloader, val_dataloader
    torch.cuda.empty_cache()


def setup_everything():
    start_time = now()
    args = parse_args()
    
    if DEBUG:
        args.fold_num=2
        args.max_epochs=3
        args.batch_size=4
        args.num_workers=2
        args.prefetch=1
        args.test_batch_size=128
        args.warmup_epochs_w_only_oridat=0
        args.update_ema_every_n_steps=1
        args.ema_start_epoch=0
        args.swa_start_epoch=0
        # args.train_filepath = args.train_filepath.replace('train.json', 'train_debug.json')
        # args.test_filepath = args.test_filepath.replace('testA.json', 'testA_debug.json')
        args.swa_start_epoch=2
   
    
    args.output_dir = osp.join(args.output_dir, start_time)
    os.makedirs(args.output_dir, exist_ok=True)
    args.savedmodel_filepath = osp.join(args.output_dir, osp.basename(args.savedmodel_filepath))
    os.makedirs(args.savedmodel_filepath, exist_ok=True)
    if args.use_ema:
        args.savedema_filepath = osp.join(args.output_dir, osp.basename(args.savedmodel_filepath), 'ema')
        os.makedirs(args.savedema_filepath, exist_ok=True)
    if args.use_swa:
        args.savedswa_filepath = osp.join(args.output_dir, osp.basename(args.savedmodel_filepath), 'swa')
        os.makedirs(args.savedswa_filepath, exist_ok=True)
    
    logging = get_logger(filename=osp.join(args.output_dir, 'log.log'))
    
    setup_device(args)
    seed_everything(args.seed)
    
    logging.info("Training/evaluation arguments: %s", args)
    shutil.make_archive(osp.join(args.output_dir, 'code'), 'zip', osp.dirname(__file__))
    shutil.copyfile(osp.join(osp.dirname(__file__), 'cfg.toml'), osp.join(args.output_dir, 'cfg.toml'))
    
    if args.make_cvids:
        cv_dir = osp.join(osp.dirname(args.train_filepath), 'cv_ids')
        os.makedirs(cv_dir, exist_ok=True)
        anns = []
        with open(args.train_filepath,'r',encoding='utf8') as f:
            for line in f.readlines():
                ann =json.loads(line)
                anns.append(ann)
        y = np.array([x['label_id'] for x in anns])
        index = np.arange(len(anns))
        make_cvids(index, y, cv_dir=cv_dir, fold_num=args.fold_num, seed=args.seed)

    return args, logging


def main():
    args, logging = setup_everything()
    # train
    for cv in range(args.fold_num):
        args.seed+=args.seed_add
        seed_everything(args.seed)
        args.cv_id = cv
        train_loop(args, logging)
    inference(args, logging)
    ensemble(args, logging, cv_confidence_masking=args.cv_confidence_masking)

if __name__ == '__main__':
    main()