# -*- coding: UTF-8 -*-

""" run
ex. python main_base.py --config config/sign_cvpr_A_hands/INCLUDE/joint_include.yaml
"""

from turtle import pendown

import sys
# sys.path.append('/raid/zhengjian/HA-SLR-GCN/')
# sys.path.append('/openbayes/home/HA-SLR-GCN-master/')
# from Code.Network.SL_GCN.parser import get_parser
from parser import get_parser

#!/usr/bin/env python
import argparse
import os
import time
import numpy as np
import yaml
import pickle
import glob
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
import shutil
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
import random
import inspect
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import pdb
from pytorchtools import EarlyStopping

try:
    import wandb  # type: ignore[reportMissingImports]
except Exception:
    wandb = None


# class LabelSmoothingCrossEntropy(nn.Module):
#     def __init__(self):
#         super(LabelSmoothingCrossEntropy, self).__init__()
#     def forward(self, x, target, smoothing=0.1):
#         confidence = 1. - smoothing
#         logprobs = F.log_softmax(x, dim=-1)
#         nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
#         nll_loss = nll_loss.squeeze(1)
#         smooth_loss = -logprobs.mean(dim=-1)
#         loss = confidence * nll_loss + smoothing * smooth_loss
#         return loss.mean()

def init_seed(_):
    torch.cuda.manual_seed_all(1)
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collate_batch_with_index(batch):
    """
    Custom collate function to handle data, label, and index.
    Converts data and label to tensors, but keeps index as a list.
    Handles edge cases like empty string labels.
    """
    data_list = []
    label_list = []
    index_list = []
    
    for item in batch:
        data, label, index = item
        data_list.append(torch.from_numpy(data))
        
        # Convert label to int, handling edge cases
        try:
            if isinstance(label, str) and label.strip() == '':
                # Skip empty string labels or assign default (0)
                label_int = 0
            else:
                label_int = int(label)
        except (ValueError, TypeError) as e:
            print(f"Warning: Invalid label '{label}' at index {index}, assigning 0")
            label_int = 0
            
        label_list.append(label_int)
        index_list.append(index)
    
    # Stack data and label into tensors
    batch_data = torch.stack(data_list, dim=0)
    batch_label = torch.tensor(label_list, dtype=torch.long)
    
    # Return index as a list (not converted to tensor)
    return batch_data, batch_label, index_list


class Processor():
    """ 
        Processor for Skeleton-based Action Recgnition
    """

    def __init__(self, args):

        from datetime import datetime
        resume_ckpt = getattr(args, 'resume_checkpoint', '')
        if args.phase.lower() == "train" and (not resume_ckpt) and getattr(args, 'auto_resume', False):
            # Auto-discover the newest *_latest.pt from prior runs of the same experiment.
            auto_pattern = os.path.join(
                "./work_dir",
                args.Experiment_name,
                "**",
                "checkpoints",
                "{}_latest.pt".format(os.path.basename(args.Experiment_name)))
            auto_candidates = glob.glob(auto_pattern, recursive=True)
            if auto_candidates:
                resume_ckpt = max(auto_candidates, key=os.path.getmtime)
                args.resume_checkpoint = resume_ckpt

        if args.phase.lower() == "train" and resume_ckpt:
            # Resume mode: reuse the existing run directory from checkpoint path.
            args.work_dir = os.path.dirname(os.path.dirname(resume_ckpt))
        else:
            args.work_dir = os.path.join("./work_dir", "{}/bs{}_f{}_lr{}{}{}{}{}/{:%Y-%m-%d_%H-%M-%S}".format(
                args.Experiment_name,
                args.batch_size, args.train_feeder_args['window_size'], 
                str(args.train_base_lr) if args.phase.lower() == "train" else str(args.test_base_lr), 
                "_trainlr{}".format(args.train_base_lr) if args.phase.lower() == "test" else "", 
                "_warmup{}".format(args.warm_up_epoch), 
                "_EarlyStop{}".format(args.es_patience) if args.es else "",
                "_test" if args.phase.lower() == "test" else "", 
                datetime.now()))

        # tensorboard
        self.model_saved_dir = os.path.join(args.work_dir, "checkpoints")
        self.sum_dir  = os.path.join(args.work_dir, "runs")
        self.score_dir = os.path.join(args.work_dir, "scores")

        args.model_saved_name = os.path.join(self.model_saved_dir, os.path.basename(args.Experiment_name))

        args.base_lr = args.train_base_lr if args.phase.lower() == "train" else args.test_base_lr
        self.args = args
        self.global_step = 0
        self.lr = self.args.base_lr
        self.best_acc = 0
        self.best_acc_epoch = 0
        self.best_loss = 1e6
        self.best_loss_epoch = 0
        self.latest_ckpt_path = ''
        self.wandb_run = None
        self._wandb_enabled = False
        
    def mk_dir(self):
        
        for mk_dir in [self.model_saved_dir, self.sum_dir, self.score_dir]:
            if not os.path.exists(mk_dir):
                os.makedirs(mk_dir)
        
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(self.sum_dir)

    def setup_wandb(self):
        if not getattr(self.args, 'use_wandb', False):
            return

        if self.wandb_run is not None:
            return

        if wandb is None:
            self.print_log('WandB is enabled but the package is not installed. Skipping WandB init.')
            return

        api_key = os.environ.get('WANDB_API_KEY', '').strip()
        if api_key:
            try:
                wandb.login(key=api_key, relogin=True)
            except Exception as exc:
                self.print_log('WandB login failed: {}'.format(exc))
                return

        wandb_kwargs = {
            'project': self.args.wandb_project or self.args.Experiment_name.split('/')[0],
            'name': self.args.wandb_run_name or os.path.basename(self.args.work_dir),
            'group': self.args.wandb_group or self.args.Experiment_name,
            'tags': self.args.wandb_tags or None,
            'mode': self.args.wandb_mode,
            'resume': self.args.wandb_resume,
        }
        if self.args.wandb_entity:
            wandb_kwargs['entity'] = self.args.wandb_entity
        if self.args.wandb_id:
            wandb_kwargs['id'] = self.args.wandb_id

        self.wandb_run = wandb.init(**wandb_kwargs)
        self._wandb_enabled = True
        wandb.config.update(vars(self.args), allow_val_change=True)
        self.print_log('WandB initialized: {}'.format(self.wandb_run.name))

    def log_wandb(self, data, step=None):
        if self._wandb_enabled and self.wandb_run is not None:
            wandb.log(data, step=step)

    def log_wandb_artifact(self, file_path, artifact_name, artifact_type='model'):
        if not (self._wandb_enabled and self.wandb_run is not None):
            return
        if not file_path or not os.path.exists(file_path):
            return

        artifact = wandb.Artifact(artifact_name, type=artifact_type)
        artifact.add_file(file_path)
        self.wandb_run.log_artifact(artifact)

    def finish_wandb(self):
        if self._wandb_enabled and self.wandb_run is not None:
            wandb.finish()
            self.wandb_run = None
            self._wandb_enabled = False

    def load_data(self):
        Feeder = import_class(self.args.feeder)
        self.data_loader = dict()
        requested_workers = int(self.args.num_worker)
        if os.name == 'nt' or not torch.cuda.is_available():
            requested_workers = 0
        self.args.num_worker = max(0, requested_workers)
        # if self.args.phase.lower() == 'train':
        self.data_loader['train'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.args.train_feeder_args),
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_worker,
            drop_last=True,
            collate_fn=collate_batch_with_index,
            worker_init_fn=init_seed)
        self.data_loader['val'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.args.val_feeder_args),
            batch_size=self.args.test_batch_size,
            shuffle=False,
            num_workers=self.args.num_worker,
            drop_last=False,
            collate_fn=collate_batch_with_index,
            worker_init_fn=init_seed)
        # else:
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.args.test_feeder_args),
            batch_size=self.args.test_batch_size,
            shuffle=False,
            num_workers=self.args.num_worker,
            drop_last=False,
            collate_fn=collate_batch_with_index,
            worker_init_fn=init_seed)
    def load_model(self):
        # 1. Xác định thiết bị (CPU hoặc GPU)
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            # Lấy device index từ args, mặc định là 0 nếu có lỗi
            output_device = self.args.device[0] if isinstance(self.args.device, list) else self.args.device
            self.output_device = output_device
            self.device = torch.device('cuda:{}'.format(output_device))
        else:
            output_device = -1  # Gán giá trị mặc định để tránh lỗi UnboundLocalError
            self.output_device = None
            self.device = torch.device('cpu')

        # 2. Khởi tạo Model
        Model = import_class(self.args.model)
        shutil.copy2(inspect.getfile(Model), self.args.work_dir)  # Lưu lại file kiến trúc mạng
        self.model = Model(**self.args.model_args)

        # 3. Đưa model lên GPU nếu khả dụng
        if output_device != -1:
            self.model = self.model.to(self.device)

        # 4. Thiết lập hàm Loss
        self.loss = nn.CrossEntropyLoss().to(self.device)

        # 5. Load trọng số (Weights)
        if self.args.weights:
            self.print_log('Load weights from {}.'.format(self.args.weights))
            if '.pkl' in self.args.weights:
                with open(self.args.weights, 'rb') as f: # Mở file pkl nên dùng chế độ 'rb'
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.args.weights, map_location=self.device, weights_only=False)

            # Support both plain state_dict files and full training checkpoints.
            if isinstance(weights, dict) and 'model_state_dict' in weights:
                weights = weights['model_state_dict']

            if not isinstance(weights, (dict, OrderedDict)):
                raise ValueError('Unsupported weight format: {}'.format(type(weights)))

            # Xử lý OrderedDict và đưa lên device
            cleaned_weights = OrderedDict()
            for k, v in weights.items():
                if torch.is_tensor(v):
                    cleaned_weights[k.split('module.')[-1]] = v.to(self.device)
            weights = cleaned_weights

            if len(weights) == 0:
                raise ValueError('No tensor weights found in checkpoint: {}'.format(self.args.weights))

            # Loại bỏ các trọng số không cần thiết (ignore_weights)
            for w in self.args.ignore_weights:
                if weights.pop(w, None) is not None:
                    self.print_log('Successfully Remove Weights: {}.'.format(w))
                else:
                    self.print_log('Can Not Remove Weights: {}.'.format(w))

            # Nạp trọng số vào model
            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights in your checkpoint:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)

        # 6. Hỗ trợ chạy đa GPU (DataParallel)
        if use_cuda and isinstance(self.args.device, list):
            if len(self.args.device) > 1:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.args.device,
                    output_device=output_device)
                
        
    def load_optimizer(self):
        """
        设置优化器optimizer 和 学习率调整策略
        """

        if self.args.optimizer == 'SGD':

            params_dict = dict(self.model.named_parameters())
            params = []

            for key, value in params_dict.items():
                decay_mult = 0.0 if 'bias' in key else 1.0

                lr_mult = 1.0
                weight_decay = 1e-4

                params += [{'params': value, 'lr': float(self.args.base_lr), 'lr_mult': lr_mult,
                            'decay_mult': decay_mult, 'weight_decay': weight_decay}]

            self.optimizer = optim.SGD(
                params,
                momentum=0.9,
                nesterov=self.args.nesterov)

        elif self.args.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=float(self.args.base_lr),
                weight_decay=float(self.args.weight_decay))
        else:
            raise ValueError()

        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1,
                                              patience=10,
                                              threshold=1e-4, threshold_mode='rel',
                                              cooldown=0)

    def _model_for_io(self):
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    def maybe_resume_training(self):
        if self.args.phase.lower() != 'train':
            return

        ckpt_path = self.args.resume_checkpoint
        if (not ckpt_path) and self.args.auto_resume:
            auto_ckpt = os.path.join(
                self.model_saved_dir,
                "{}_latest.pt".format(os.path.basename(self.args.Experiment_name)))
            if os.path.exists(auto_ckpt):
                ckpt_path = auto_ckpt

        if not ckpt_path:
            return

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError('Resume checkpoint not found: {}'.format(ckpt_path))

        self.print_log('Resume training from checkpoint: {}'.format(ckpt_path))
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        if not isinstance(ckpt, dict) or 'model_state_dict' not in ckpt:
            raise ValueError('Invalid resume checkpoint format: {}'.format(ckpt_path))

        self._model_for_io().load_state_dict(ckpt['model_state_dict'])

        if 'optimizer_state_dict' in ckpt and ckpt['optimizer_state_dict'] is not None:
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        resumed_epoch = int(ckpt.get('epoch', -1)) + 1
        self.args.start_epoch = max(int(self.args.start_epoch), resumed_epoch)
        self.best_acc = float(ckpt.get('best_acc', self.best_acc))
        self.best_acc_epoch = int(ckpt.get('best_acc_epoch', self.best_acc_epoch))
        self.best_loss = float(ckpt.get('best_loss', self.best_loss))
        self.best_loss_epoch = int(ckpt.get('best_loss_epoch', self.best_loss_epoch))

        self.print_log('Continue from epoch {}.'.format(self.args.start_epoch + 1))

    def save_latest_checkpoint(self, epoch, val_loss, accuracy):
        ckpt = {
            'epoch': int(epoch),
            'model_state_dict': self._model_for_io().state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_acc': float(self.best_acc),
            'best_acc_epoch': int(self.best_acc_epoch),
            'best_loss': float(self.best_loss),
            'best_loss_epoch': int(self.best_loss_epoch),
            'val_loss': float(val_loss),
            'val_acc': float(accuracy),
        }
        torch.save(ckpt, self.latest_ckpt_path)

    def _select_test_checkpoints(self, checkpoint_paths):
        if not checkpoint_paths:
            return []

        # Use checkpoints from the most recently updated run directory.
        run_dirs = list(set(os.path.dirname(os.path.dirname(p)) for p in checkpoint_paths))
        latest_run_dir = max(run_dirs, key=os.path.getmtime)
        run_checkpoints = [p for p in checkpoint_paths if os.path.dirname(os.path.dirname(p)) == latest_run_dir]

        selected = []
        for suffix in ['_latest.pt', '_best_acc', '_best_loss']:
            matches = [p for p in run_checkpoints if suffix in os.path.basename(p)]
            if matches:
                selected.append(max(matches, key=os.path.getmtime))

        if not selected:
            selected.append(max(run_checkpoints, key=os.path.getmtime))

        return selected

    def adjust_learning_rate(self, epoch):
        """
        GCN的默认策略: warm_up  
        """
        if self.args.optimizer == 'SGD' or self.args.optimizer == 'Adam':
            if epoch < self.args.warm_up_epoch:
                lr = float(self.args.base_lr) * (epoch + 1) / self.args.warm_up_epoch     # ex.  0.1/20  0.1/20*2 0.1/20*3 ...
            else:
                lr = float(self.args.base_lr) * (
                    0.1 ** np.sum(epoch >= np.array(self.args.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr   # 发挥作用 work!
            return lr
        else:
            raise ValueError()

    def train(self, epoch, save_model=False):

        if epoch >= self.args.only_train_epoch:
            print('only train part, require grad')
            for key, value in self.model.named_parameters():
                if 'DecoupleA' in key:
                    value.requires_grad = True
                    print(key + '-require grad')
        else:
            print('only train part, do not require grad')
            for key, value in self.model.named_parameters():
                if 'DecoupleA' in key:
                    value.requires_grad = False
                    print(key + '-not require grad')  # ...

        self.model.train()
        self.print_log('Training epoch: {}'.format(epoch + 1))
        loader = self.data_loader['train']
        self.adjust_learning_rate(epoch)  # 执行warmup
        all_loss = []
        all_acc = []

        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)

        for batch_idx, (data, label, index) in enumerate(tqdm(loader)):
            # print("train ", data.size())
            self.global_step += 1
            # get data
            data = Variable(data.float().to(self.device), requires_grad=False)
            label = Variable(label.long().to(self.device), requires_grad=False)
            timer['dataloader'] += self.split_time()

            # forward
            # if epoch < 100:
            #     keep_prob = -(1 - self.args.keep_rate) / 100 * epoch + 1.0
            # else:
            #     keep_prob = self.args.keep_rate
            # output = self.model(data, keep_prob)
            output = self.model(data)   # N, C, T, V, M  (64, 3, 100, 27, 1)   # 100帧 <- feeder.window_size

            if isinstance(output, tuple):
                output, l1 = output
                l1 = l1.mean()
            else:
                l1 = 0

            loss = self.loss(output, label) + l1

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            timer['model'] += self.split_time()

            value, predict_label = torch.max(output.data, 1)
            acc = torch.mean((predict_label == label.data).float())
            all_acc.append(acc.data.cpu().numpy())
            all_loss.append(loss.data.cpu().numpy())

            self.lr = self.optimizer.param_groups[0]['lr']
            if self.global_step % self.args.log_interval == 0:
                self.print_log(
                    '\tBatch({}/{}) done. Loss: {:.4f}  lr:{:.12f}'.format(
                        batch_idx, len(loader), loss.data, self.lr))


        training_acc = np.mean(all_acc)
        training_loss = np.mean(all_loss)

        state_dict = self.model.state_dict()
        weights = OrderedDict([[k.split('module.')[-1],
                                v.cpu()] for k, v in state_dict.items()])
        
        # Log
        self.writer.add_scalars('Loss', {'train': training_loss}, epoch+1)
        self.writer.add_scalars('Accuracy', {'train': training_acc}, epoch+1)
        self.log_wandb({
            'train/loss': float(training_loss),
            'train/accuracy': float(training_acc),
            'train/lr': float(self.lr),
        }, step=epoch + 1)
        
        return weights

    def eval(self, epoch, loader_name=['test'], wrong_file=None, result_file=None):
        
        if wrong_file is not None:
            if not os.path.exists(os.path.dirname(wrong_file)):
                os.makedirs(os.path.dirname(wrong_file))
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            if not os.path.exists(os.path.dirname(result_file)):
                os.makedirs(os.path.dirname(result_file))
            f_r = open(result_file, 'w')
        self.model.eval()

        pred_list, true_list, wrong_csv = [], [], []

        with torch.no_grad():

            if not self.args.phase.lower() == "test":
                self.print_log('Eval epoch: {}'.format(epoch + 1))
            
            self.print_log('lr: {}'.format(self.lr))
                
            for ln in loader_name:
                all_loss = []
                all_acc = []
                step = 0

                for batch_idx, (data, label, index) in enumerate(tqdm(self.data_loader[ln])):
                    data = Variable(data.float().to(self.device), requires_grad=False) # torch.Size([64, 3, 150, 27, 1])
                    # print("data.size() ", data.size())
                    # pdb.set_trace()
                    label = Variable(label.long().to(self.device), requires_grad=False)

                    with torch.no_grad():
                        output = self.model(data)

                    if isinstance(output, tuple):
                        output, l1 = output
                        l1 = l1.mean()
                    else:
                        l1 = 0
                    loss = self.loss(output, label)
                    all_acc.append(output.data.cpu().numpy())
                    all_loss.append(loss.data.cpu().numpy())

                    _, predict_label = torch.max(output.data, 1)
                    step += 1

                    if wrong_file is not None or result_file is not None:
                        predict = list(predict_label.cpu().numpy())
                        true = list(label.data.cpu().numpy())
                        for i, _ in enumerate(predict):
                            pred_list.append(predict[i])
                            true_list.append(true[i])
                            if result_file is not None:
                                f_r.write(str(predict[i]) + ',' + str(true[i]) + '\n')
                            if predict[i] != true[i] and wrong_file is not None:
                                
                                assert(true[i] == self.data_loader[ln].dataset.label[int(index[i])])
                                try:
                                    if (self.data_loader[ln].dataset.label==predict[i]).all() == False:
                                        wrong_csv.append([index[i], predict[i], true[i], os.path.dirname(self.data_loader['train'].dataset.sample_name[int(np.where(self.data_loader['train'].dataset.label==predict[i])[0][0])]), self.data_loader[ln].dataset.sample_name[int(index[i])]])
                                    else:
                                        wrong_csv.append([index[i], predict[i], true[i], os.path.dirname(self.data_loader[ln].dataset.sample_name[int(np.where(self.data_loader[ln].dataset.label==predict[i])[0][0])]), self.data_loader[ln].dataset.sample_name[int(index[i])]])
                                    f_w.write(str(index[i]) + ',' + str(predict[i]) + ',' + str(true[i]) + '\n')
                                except Exception as e:
                                    print(e)
                                    print(index[i])
                                    print(predict[i])
                                    print(true[i])
                                    # print(os.path.dirname(self.data_loader[ln].dataset.sample_name[int(np.where(self.data_loader[ln].dataset.label==predict[i])[0][0])]))
                                    pdb.set_trace()
                
                score = np.concatenate(all_acc)

                # 获取混淆矩阵   
                # %matplotlib inline
                # from sklearn.metrics import confusion_matrix
                # cm = confusion_matrix(true_list, pred_list)
                # from Code.utils import plot_confusion_matrix
                # plot_confusion_matrix(cm, './work_dir/' + self.args.Experiment_name + f"/eval_results/{self.args.eval_pkl}" + '_confusion_matrix.png', title='confusion matrix')

                # wrong csv
                if wrong_file is not None:
                    import pandas as pd
                    name=['index','pred_label','true_label','pred_sample','true_sample']
                    csv_df=pd.DataFrame(columns=name,data=wrong_csv) 
                    csv_path = wrong_file[:-4] + '.csv'
                    csv_df.to_csv(csv_path,encoding='gbk', index=None)

                accuracy = self.data_loader[ln].dataset.top_k(score, 1)
                score_dict = dict(zip(self.data_loader[ln].dataset.sample_name, score))   # [(sample_name(str), (226, )(ndarray))] ...

                if accuracy > self.best_acc:
                    self.best_acc = accuracy
                    self.best_acc_epoch = epoch

                    with open(self.eval_pkl, 'wb') as f:
                        pickle.dump(score_dict, f)
                
                if self.args.phase.lower() == 'test':
                    with open(self.eval_pkl, 'wb') as f:
                        pickle.dump(score_dict, f)

                
                self.print_log('\tMean {} loss of {} batches: {}.'.format(
                    ln, len(self.data_loader[ln]), np.mean(all_loss)))
                
                
                for k in self.args.show_topk:
                    self.print_log('\tTop{}: {:.2f}%'.format(
                        k, 100 * self.data_loader[ln].dataset.top_k(score, k)))

                 # 记录每个epoch的output
                # score_dict = dict(zip(self.data_loader[ln].dataset.sample_name, score))  
                # with open('./work_dir/' + args.Experiment_name + '/eval_results/epoch_' + str(epoch) + '_' + str(accuracy) + '.pkl'.format(
                #         epoch, accuracy), 'wb') as f:
                #     pickle.dump(score_dict, f)

                print('Eval Accuracy: ', accuracy, '\n model: ', self.args.work_dir)

        # Log
        validation_loss = np.mean(all_loss)
        self.writer.add_scalars('Loss', {'validation': validation_loss}, epoch+1)
        self.writer.add_scalars('Accuracy', {'validation': accuracy}, epoch+1)
        self.log_wandb({
            'validation/loss': float(validation_loss),
            'validation/accuracy': float(accuracy),
        }, step=epoch + 1)

        return validation_loss, accuracy

    def start(self):
        if self.args.phase.lower() == 'train':

            self.mk_dir()   # 放在save_arg\load_model 前面
            self.setup_wandb()

            self.load_model()   # train_finetune时 会加载权重...
            self.load_optimizer()
            self.maybe_resume_training()
            self.load_data()
            
            self.save_arg()  # 不放在init中， 避免 test phase 测试时 产生无效文件

            self.eval_pkl = os.path.join(self.score_dir, "{}_best_acc_score.pkl".format(os.path.basename(self.args.Experiment_name)))

            best_acc_ckpt = '{}_best_acc.pt'.format(self.args.model_saved_name) 
            best_loss_ckpt = '{}_best_loss.pt'.format(self.args.model_saved_name) 
            self.latest_ckpt_path = '{}_latest.pt'.format(self.args.model_saved_name)

            # self.print_log('Parameters:\n{}\n'.format(str(vars(self.args))))  # see work_dir/xxx/config.yaml
            self.global_step = self.args.start_epoch * \
                len(self.data_loader['train']) / self.args.batch_size
            
            # initialize the early_stopping object
            if self.args.es:
                early_stopping = EarlyStopping(patience=self.args.es_patience)

            for epoch in range(self.args.start_epoch, self.args.num_epoch):
                save_model = ((epoch + 1) % self.args.save_interval == 0) or (
                    epoch + 1 == self.args.num_epoch)

                weights = self.train(epoch, save_model=save_model)

                val_loss, accuracy = self.eval(epoch, loader_name=['val']) 

                if accuracy == self.best_acc:
                    torch.save(weights, best_acc_ckpt)

                if self.best_loss > val_loss:
                    self.best_loss  = val_loss
                    self.best_loss_epoch = epoch

                    # torch.save(weights, self.args.model_saved_name + '-' + str(epoch) + '-' + str(val_loss)  + '.pt')
                    torch.save(weights, best_loss_ckpt)
                    print(val_loss)

                self.save_latest_checkpoint(epoch, val_loss, accuracy)


                # self.lr_scheduler.step(val_loss)

                # early_stopping needs the validation loss to check if it has decresed, 
                # and if it has, it will make a checkpoint of the current model

                if self.args.es:
                    early_stopping(val_loss)
                
                    if early_stopping.early_stop:
                        self.print_log("Early stopping")
                        break

            best_acc_ckpt_final = "{}_best_acc_{}_{}.pt".format(self.args.model_saved_name, str(self.best_acc_epoch), str(self.best_acc)[2:6])
            best_loss_ckpt_final = "{}_best_loss_{}_{}.pt".format(self.args.model_saved_name, str(self.best_loss_epoch), str(self.best_loss)[2:6])
            if os.path.exists(best_acc_ckpt):
                os.rename(src=best_acc_ckpt, dst=best_acc_ckpt_final)
            if os.path.exists(best_loss_ckpt):
                os.rename(src=best_loss_ckpt, dst=best_loss_ckpt_final)

            self.log_wandb_artifact(self.latest_ckpt_path, '{}_latest'.format(os.path.basename(self.args.Experiment_name)), artifact_type='checkpoint')
            self.log_wandb_artifact(best_acc_ckpt_final, '{}_best_acc'.format(os.path.basename(self.args.Experiment_name)), artifact_type='checkpoint')
            self.log_wandb_artifact(best_loss_ckpt_final, '{}_best_loss'.format(os.path.basename(self.args.Experiment_name)), artifact_type='checkpoint')

            self.print_log('\n')
            self.print_log('model_name: {}'.format(self.args.Experiment_name))
            self.print_log('best accuracy: {}'.format(self.best_acc))
            self.print_log('best accuracy_epoch: {}'.format(best_acc_ckpt_final))
            self.print_log('best loss: {}'.format(self.best_loss))
            self.print_log('best loss_epoch: {}'.format(best_loss_ckpt_final))
            self.log_wandb({
                'best/accuracy': float(self.best_acc),
                'best/accuracy_epoch': int(self.best_acc_epoch),
                'best/loss': float(self.best_loss),
                'best/loss_epoch': int(self.best_loss_epoch),
            })
            self.finish_wandb()

        elif self.args.phase.lower() == 'test':

            if not os.path.exists(self.args.weights):
                
                from datetime import datetime
                self.args.pretrain_work_dir = os.path.join("./work_dir", self.args.Experiment_name+"/bs{}_f{}_lr{}_warmup*{}/{}".format(
                    self.args.batch_size, self.args.train_feeder_args['window_size'], str(self.args.train_base_lr), 
                    "_EarlyStop{}".format(self.args.es_patience) if self.args.es else "",
                    self.args.pretrain_cpt_time_tag))
                self.args.pt_dir = os.path.join(self.args.pretrain_work_dir,  "checkpoints/*.pt")
                print("pt_dir ", self.args.pt_dir)

                import glob

                self.args.check_dir = glob.glob(self.args.pt_dir)
                if self.args.check_dir == []:
                    raise FileNotFoundError("No checkpoint found. Please check Experiment_name or set weights explicitly.")

                self.args.check_dir = self._select_test_checkpoints(self.args.check_dir)

                print("check_dir ", self.args.check_dir, len(self.args.check_dir))
        
                self.mk_dir()
                self.save_arg()
                
                for fp in sorted(self.args.check_dir):

                    for i in range(len(self.args.splits)):
                        split = self.args.splits[i]

                        self.args.test_feeder_args['label_path'] = "./data/{}/{}/{}_label.pkl".format(self.args.joint_sign, self.args.joint_type, split)
                        print(self.args.test_feeder_args['label_path'])

                        self.args.test_feeder_args['data_path'] = "./data/{}/{}/{}_data_{}.npy".format(self.args.joint_sign, self.args.joint_type, split, self.args.modality)
                    
                        self.args.weights = fp

                        self.test_phase(split)
            else:
                
                self.mk_dir()
                self.setup_wandb()
                self.save_arg()
                
                for i in range(len(self.args.splits)):
                    
                        split = self.args.splits[i]

                        self.args.test_feeder_args['data_path'] = "./data/{}/{}/{}_data_{}.npy".format(self.args.joint_sign, self.args.joint_type, split, self.args.modality)

                        self.args.test_feeder_args['label_path'] = "./data/{}/{}/{}_label.pkl".format(self.args.joint_sign, self.args.joint_type, split)
                        print(self.args.test_feeder_args['label_path'])
                    
                        self.test_phase(split)
                    

    def test_phase(self, split):

        self.args.eval_pkl = os.path.basename(self.args.weights)[:-3] + "_{}".format(split)

        self.eval_pkl = os.path.join(self.score_dir, "{}_score.pkl".format(self.args.eval_pkl))

        if not self.args.test_feeder_args['debug']:
            wf = os.path.join(self.args.work_dir, "sample_pred_true/{}_wrong.txt".format(self.args.eval_pkl))
            rf = os.path.join(self.args.work_dir, "sample_pred_true/{}_right.txt".format(self.args.eval_pkl))
        else:
            wf = rf = None
        
        # test 阶段， 自动匹配 相应 数据和模型，不可在__init__中率先执行
        self.setup_wandb()
        self.load_data()
        self.load_model()
        self.load_optimizer()

        self.args.print_log = True   # or False
        self.print_log('eval_pkl: ', self.eval_pkl)  
        self.print_log('feeder: {}.'.format(self.args.feeder))
        self.print_log('data_path: {}.'.format(self.args.test_feeder_args['data_path']))
        self.print_log('label_path: {}.'.format(self.args.test_feeder_args['label_path']))
        self.print_log('Model: {}.'.format(self.args.model))
        self.eval(epoch=self.args.start_epoch, loader_name=['test'], wrong_file=wf, result_file=rf)
        self.log_wandb({'test/split_done': split, 'test/weight_name': os.path.basename(self.args.weights)})
        self.print_log('Done.\n')


    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.args.print_log:
            with open('{}/log.txt'.format(self.args.work_dir), 'a', encoding='utf-8') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def save_arg(self):
        # save arg
        arg_dict = vars(self.args)
        with open('{}/config.yaml'.format(self.args.work_dir), 'w', encoding='utf-8') as f:
            yaml.dump(arg_dict, f)

        # 保存当前文件
        shutil.copy(os.path.abspath(__file__), self.args.work_dir)

def import_class(name):
    components = name.split('.')
    
    mod = __import__(components[0])  # import return model
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

if __name__ == '__main__':

    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r', encoding='utf-8') as f:
            default_arg = yaml.safe_load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                # assert (k in key)
        parser.set_defaults(**default_arg)
        
    arg = parser.parse_args()
    init_seed(0)
    processor = Processor(arg)
    
    processor.start()
