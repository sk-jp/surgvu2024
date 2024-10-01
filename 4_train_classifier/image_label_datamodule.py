import csv
import lightning.pytorch as pl
import numpy as np
import torch
from torch.utils.data import DataLoader

from monai.data import (
    Dataset,
    PersistentDataset,
)

from get_transform import get_transform


# load csv file
def load_csv(csv_file, topdir, num_labels):    
    rets = []
    with open(csv_file, 'rt') as f:
        f.readline()    # skip the header
        rows = csv.reader(f)
        for row in rows:
            case_id = row[0]
            video_id = row[1]
            frame_no = int(row[2])
            image_path = f'{topdir}/{case_id}/{video_id}/{frame_no:08d}.jpg'
            label = [int(r) for r in row[3:3+num_labels]]

            # return values
            ret = {}
            ret['image'] = image_path
            ret['label'] = np.array(label, dtype=np.float32)
            ret['filepath'] = image_path
            rets.append(ret)
            
    return rets

class ImageLabelDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super(ImageLabelDataModule, self).__init__()

        # configs
        self.cfg = cfg

    # called once from main process
    # called only within a single process
    def prepare_data(self):
        # prepare data
        pass
    
    # perform on every GPU
    def setup(self, stage):
        self.dataset = {}
        train_datalist = load_csv(self.cfg.Data.dataset.train_datalist,
                                  self.cfg.Data.dataset.top_dir,
                                  self.cfg.Data.dataset.num_labels)
        valid_datalist = load_csv(self.cfg.Data.dataset.valid_datalist,
                                  self.cfg.Data.dataset.top_dir,
                                  self.cfg.Data.dataset.num_labels)

        train_transforms = get_transform(self.cfg.Transform.train)
        valid_transforms = get_transform(self.cfg.Transform.valid)

        if self.cfg.General.mode == "predict":
            predict_datalist = load_csv(self.cfg.Data.dataset.predict_datalist,
                                        self.cfg.Data.dataset.top_dir)
            predict_transforms = get_transform(self.cfg.Transform.predict)
            
#        self.dataset['train'] = PersistentDataset(
        self.dataset['train'] = Dataset(
           data=train_datalist,
            transform=train_transforms,
#            cache_dir=self.cfg.Data.dataset.cache_dir
        )
#        self.dataset['valid'] = PersistentDataset(
        self.dataset['valid'] = Dataset(
            data=valid_datalist,
            transform=valid_transforms,
#            cache_dir=self.cfg.Data.dataset.cache_dir
        )            

        if self.cfg.General.mode == "predict":
            self.dataset['predict'] = Dataset(
                data=predict_datalist,
                transform=predict_transforms,
            )

    # call in Trainer.fit()
    def train_dataloader(self):
        train_loader = DataLoader(
            self.dataset['train'],
            batch_size=self.cfg.Data.dataloader.batch_size,
            shuffle=self.cfg.Data.dataloader.train.shuffle,
            num_workers=self.cfg.Data.dataloader.num_workers,
            pin_memory=self.cfg.Data.dataloader.pin_memory,
            persistent_workers=self.cfg.Data.dataloader.persistent_workers,
            drop_last=True
        )
        return train_loader

    # call in Trainer.fit() and Trainer.validate()
    def val_dataloader(self):
        val_loader = DataLoader(
            self.dataset['valid'],
            batch_size=self.cfg.Data.dataloader.batch_size,
            shuffle=self.cfg.Data.dataloader.valid.shuffle,
            num_workers=self.cfg.Data.dataloader.num_workers,
            pin_memory=self.cfg.Data.dataloader.pin_memory,
            persistent_workers=self.cfg.Data.dataloader.persistent_workers
        )
        return val_loader

    # call in Trainer.predict()
    def predict_dataloader(self):
        if self.cfg.General.mode == "predict":
            predict_loader = DataLoader(
                self.dataset['predict'],
                batch_size=self.cfg.Data.dataloader.batch_size,
                shuffle=self.cfg.Data.dataloader.predict.shuffle,
                num_workers=self.cfg.Data.dataloader.num_workers,
                pin_memory=False
            )
        else:
            predict_loader = None
        return predict_loader
