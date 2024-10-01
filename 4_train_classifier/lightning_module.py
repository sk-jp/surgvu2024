#from collections import OrderedDict
#import numpy as np
import lightning.pytorch as pl
import torch
#import torch.nn.functional as F
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MultilabelAccuracy,
    MultilabelF1Score,
    MultilabelPrecision, 
    MultilabelRecall
)

from fix_model_state_dict import fix_model_state_dict
from get_optimizer import get_optimizer
from loss import Loss


class LightningModule(pl.LightningModule):
    def __init__(self, cfg):
        super(LightningModule, self).__init__()

        self.cfg = cfg
        self.lossfun = Loss(cfg.Loss)
        self.lossfun_valid = Loss(cfg.Loss)
        self.txt_logger = cfg.txt_logger

        self.training_step_outputs = []
        self.validation_step_outputs = []

        # model
        if cfg.Model.arch == 'timm':
            from timm_model import TimmModel
            self.model = TimmModel(**cfg.Model.params)
        else:
            raise ValueError(f'{cfg.Model.arch} is not supported.')

        if cfg.Model.pretrained is not None:
            # Load pretrained model weights
            print(f'Loading: {cfg.Model.pretrained}')
            checkpoint = torch.load(cfg.Model.pretrained, map_location='cpu')
            state_dict = checkpoint['state_dict']
            state_dict = fix_model_state_dict(state_dict)
            self.model.load_state_dict(state_dict)

        # metrics
        metrics_fun = MetricCollection([
            MultilabelAccuracy(num_labels=self.cfg.Data.dataset.num_labels,
                               average='none',
                               multidim_average='global'),
            MultilabelF1Score(num_labels=self.cfg.Data.dataset.num_labels,
                              average='none',
                              multidim_average='global'),
            MultilabelPrecision(num_labels=self.cfg.Data.dataset.num_labels,
                                average='none',
                                multidim_average='global'),
            MultilabelRecall(num_labels=self.cfg.Data.dataset.num_labels,
                             average='none',
                             multidim_average='global')
        ])
        self.metrics_list = ['MultilabelAccuracy', 'MultilabelF1Score', 
                             'MultilabelPrecision', 'MultilabelRecall']

        self.train_metrics_fun = metrics_fun.clone(prefix='train_')
        self.valid_metrics_fun = metrics_fun.clone(prefix='valid_')

        # flag to check the validation is performed or not at the end of epoch
        self.did_validation = False

    def forward(self, x):
        y = self.model(x)
        return y

    def on_train_epoch_start(self):
        # clear up GPU memory
        torch.cuda.empty_cache()
        pl.utilities.memory.garbage_collection_cuda()
        self.did_validation = False
        
    def training_step(self, batch, batch_idx):
        image = batch['image']
        label = batch['label']

        pred = self.model(image)

        loss, _ = self.lossfun(pred, label)
        
        # Metrics
        for metric in self.metrics_list:
            self.train_metrics_fun[metric].update(torch.sigmoid(pred).detach(), label)

        output = {"loss": loss}
        self.training_step_outputs.append(output)

        return loss

    def on_train_epoch_end(self):
        # print the results
        outputs_gather = self.all_gather(self.training_step_outputs)

        # metrics (each metric is synced and reduced across each process)
        train_metrics = self.train_metrics_fun.compute()
        self.train_metrics_fun.reset()

        if self.trainer.is_global_zero:
            epoch = int(self.current_epoch)
            self.txt_logger.info(f'Epoch: {epoch}')
            
            train_loss = torch.stack([o['loss']
                                      for o in outputs_gather]).mean().detach()

            d = dict()
            d['epoch'] = epoch
            d['train_loss'] = train_loss
#            d.update(train_metrics)
            for metric in self.metrics_list:
                d[f'train_{metric}'] = train_metrics[f'train_{metric}'].mean()

            print('\n Mean:')
            s = f'  Train:\n'
            s += f'    loss: {train_loss.item():.3f}'
            for metric in self.metrics_list:
                s += f', {metric.replace("Multilabel", "")}: {train_metrics[f"train_{metric}"].mean().cpu().numpy():.3f}'
            print(s)
            self.txt_logger.info(s)
            if self.did_validation:
                s = '  Valid:\n'
                s += f'    loss: {self.valid_loss:.3f}'
                s += '\n'
                s += '  '
                for metric in self.metrics_list:
                    s += f'  {metric.replace("Multilabel", "")}: {self.valid_metrics[f"valid_{metric}"].mean().cpu().numpy():.3f}'
                print(s)
                self.txt_logger.info(s)
                
            # log
            self.log_dict(d, prog_bar=False, rank_zero_only=True)

            # free up the memory
            self.training_step_outputs.clear()

    def on_validation_start(self):
        # clear GPU cache before the validation
        torch.cuda.empty_cache()

    def validation_step(self, batch, batch_idx):
        image = batch['image']
        label = batch['label']
        
        # prediction
        pred = self.model(image)
        
        # loss
        loss, _ = self.lossfun_valid(pred, label)
        
        # metrics
        for metric in self.metrics_list:
            self.valid_metrics_fun[metric].update(torch.sigmoid(pred), label)
        
        output = {'loss': loss.detach()}
        self.validation_step_outputs.append(output)
        
        return

    def on_validation_epoch_end(self):
        epoch = int(self.current_epoch)
        valid_loss = torch.stack([o['loss']
                                  for o in self.validation_step_outputs]).mean().item()
        self.valid_loss = valid_loss

        # metrics (in torchmetrics, gathered automatically)
        self.valid_metrics = self.valid_metrics_fun.compute()
        self.valid_metrics_fun.reset()  

        # log
        d = dict()
        d['epoch'] = epoch
        d['valid_loss'] = valid_loss
#        d.update(self.valid_metrics)
        for metric in self.metrics_list:
            d[f'valid_{metric}'] = self.valid_metrics[f'valid_{metric}'].mean()
        self.log_dict(d, prog_bar=False, rank_zero_only=True)

        # free up the memory
        self.validation_step_outputs.clear()
        
        # setup flag
        self.did_validation = True

    def configure_optimizers(self):
        conf_optim = self.cfg.Optimizer

        if hasattr(conf_optim.optimizer, 'params'):
            optimizer_cls, scheduler_cls = get_optimizer(conf_optim)
            optimizer = optimizer_cls(
                self.parameters(),
                **conf_optim.optimizer.params)
        else:
            optimizer_cls, scheduler_cls = get_optimizer(conf_optim)

        if scheduler_cls is None:
            return [optimizer]
        else:
            optimizer = torch.optim.Adam(
                self.parameters(), **conf_optim.optimizer.params)
            scheduler = scheduler_cls(
                optimizer, **conf_optim.lr_scheduler.params)
            return [optimizer], [scheduler]

    def get_progress_bar_dict(self):
        items = dict()

        return items
