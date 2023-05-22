import torch
import numpy as np
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
from sudoku.loader.dataloader import DataLoader
from sudoku.loader.dataloader_df import SudokuData
from sudoku.trainer.model.backbone.resnet18 import ResNet_18
from sudoku.trainer.model.loss.focal_loss import FocalLoss
from torchmetrics.classification import MulticlassF1Score, MulticlassRecall, MulticlassPrecision, MulticlassAveragePrecision
from sudoku.trainer.model.backbone.Lenet import CNN
from sklearn.model_selection import KFold
from sudoku import load_config
from torchvision.transforms import Resize, Compose, ColorJitter, RandomRotation, ToTensor, GaussianBlur
import pathlib
import pandas as pd
import glob
import os

class Train:
    
    def __init__(self, config) -> None:
        self.kfold = KFold()
        self.lr = float(config['Optimizer']['learning_rate'])
        self.epochs = int(config['Global']['epoch_num'])
        self.num_classes = config['Architecture']['Backbone']['num_classes']
        self.device = 'cuda:0' if config['Global']['use_gpu'] else 'cpu'
        self.data_dir = config['Train']['data_dir']
        self.batch_size = config['Dataloader']['batch_size']
        optim = {
            'Adam': torch.optim.Adam
        }
        
        loss_func = {
            'CE': nn.CrossEntropyLoss(),
            'Focal': FocalLoss(gamma=2.0)
        }
        
        self.model = ResNet_18(3, self.num_classes).to(self.device)
        self.optim = optim[config['Optimizer']['name']](self.model.parameters())
        self.criterion = loss_func[config['Loss']['name']]
        self.scheduler = OneCycleLR(optimizer=self.optim,
                                    max_lr=self.lr,
                                    total_steps=10000,
                                    epochs=self.epochs)
    
    def train_step(self, batch):
        inputs, targets = batch
        preds = self.model(inputs)
        loss = self.criterion(preds, targets)
        loss.backward()
        self.optim.step()
        self.scheduler.step()
        
    def load_data_kfold(self):
        folders = glob.glob(self.data_dir + "/*/")

        data = {
            'file': [],
            'label': []
        }
        for folder in folders:
            files = [os.path.basename(file) for file in glob.glob(folder + '/*')]
            for file in files:
                data['file'].append(file)
                data['label'].append(pathlib.PurePath(folder).name)

        return data
    
    def kfold_split(self):
        data = self.load_data_kfold()
        df = pd.DataFrame(data)
        df_sample = df.sample(frac=1).reset_index(drop=True)
        return df_sample
    
    def metrics(self, preds, targets):
        metric_f1_scores = MulticlassF1Score(
            num_classes=self.num_classes).to(self.device)
        metric_recall = MulticlassRecall(
            num_classes=self.num_classes).to(self.device)
        metric_precision = MulticlassPrecision(
            num_classes=self.num_classes).to(self.device)
        metric_ap = MulticlassAveragePrecision(
            num_classes=self.num_classes).to(self.device)
        
        
        f1_scores = metric_f1_scores(preds, targets)
        recall = metric_recall(preds, targets)
        precision = metric_precision(preds, targets)
        ap = metric_ap(preds, targets)
        return f1_scores, recall, precision, ap
    
    def eval(self):
        total = {
            'loss': 0,
            'f1_scores': 0,
            'recall': 0,
            'precision': 0,
            'AP': 0
        }

        with torch.no_grad():
            for vdata in tqdm(self.val_loader, ncols=100):
                loss, f1_scores, recall, precision, ap = self.valid_step(vdata)
                total['loss'] += loss
                total['f1_scores'] += f1_scores.cpu().numpy()
                total['recall'] += recall.cpu().numpy()
                total['precision'] += precision.cpu().numpy()
                total['AP'] += ap.cpu().numpy()
                
        avg = {metric: total[metric] / len(self.val_loader) for metric in total}
        return avg
    
    def valid_step(self, batch):
        self.model.eval()
        inputs, targets = batch
        preds = self.model(inputs)
        loss = self.criterion(preds, targets)
        f1_scores, recall, precision, ap = self.metrics(preds, targets)
        return loss.item(), f1_scores, recall, precision, ap

    def train(self):
        EPOCHS = self.epochs

        for epoch in range(1, EPOCHS+1):
            print("epoch {}/{}".format(epoch, EPOCHS))

            self.model.train()
            for batch in tqdm(self.train_loader):
                self.train_step(batch)
            
            if epoch % 5 == 0:
                metrics = self.eval()
                print(metrics)
                
    def main(self):
        df = self.kfold_split()

        transform = Compose([Resize(32),
                             RandomRotation(20),
                             GaussianBlur(3),
                             ColorJitter(hue=.05, saturation=.05),
                             ToTensor()])

        for i, (train_indices, valid_indices) in enumerate(self.kfold.split(df)):
            print('Fold:', i+1)
            df_train = df.loc[train_indices]
            df_valid = df.loc[valid_indices]
            train_data = SudokuData(self.data_dir, df_train, self.device, transform)
            valid_data = SudokuData(self.data_dir, df_valid, self.device, transform)
            self.train_loader = torch.utils.data.DataLoader(train_data,
                                                       batch_size=self.batch_size)
            self.val_loader = torch.utils.data.DataLoader(valid_data,
                                                       batch_size=self.batch_size)
            self.train()
        