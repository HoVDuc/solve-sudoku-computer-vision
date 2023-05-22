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

    def __init__(self):
        self.write = SummaryWriter('runs/sudoku_experiment')

    def check_gpu(self, use_gpu):
        if use_gpu and not torch.cuda.is_available():
            print("Device not supported gpu")
            use_gpu = False
        use_cuda = "cuda" if use_gpu else "cpu"
        return torch.device(use_cuda)

    def metrics(self, preds, targets):
        metric_f1_scores = MulticlassF1Score(
            num_classes=self.num_classes).to(self.cuda)
        metric_recall = MulticlassRecall(
            num_classes=self.num_classes).to(self.cuda)
        metric_precision = MulticlassPrecision(
            num_classes=self.num_classes).to(self.cuda)
        metric_ap = MulticlassAveragePrecision(
            num_classes=self.num_classes).to(self.cuda)

        f1_scores = metric_f1_scores(preds, targets)
        recall = metric_recall(preds, targets)
        precision = metric_precision(preds, targets)
        ap = metric_ap(preds, targets)
        return f1_scores, recall, precision, ap

    def load_checkpoint(self):
        self.model.load_state_dict(torch.load(self.path_checkpoint))

    def train_one_epoch(self):
        total_loss = 0
        for data in tqdm(self.train_loader, ncols=100):
            x, y = data
            x, y = x.to(device=self.cuda), y.to(device=self.cuda)

            y_hat = self.model(x)
            loss = self.criterion(y_hat, y)
            loss.backward()
            self.optim.step()
            self.scheduler.step()

            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def eval(self):
        self.model.eval()

        total_loss = []
        total_acc = []
        total_f1_scores = []
        total_recall = []
        total_precision = []
        total_ap = []

        with torch.no_grad():
            for vdata in tqdm(self.val_loader, ncols=100):
                x, y = vdata
                x, y = x.to(device=self.cuda), y.to(device=self.cuda)
                y_hat = self.model(x)
                loss = self.criterion(y_hat, y)
                y_hat = torch.softmax(y_hat, dim=1)
                f1_scores, recall, precision, ap = self.metrics(y_hat, y)
                acc = sum(torch.argmax(y_hat, dim=1) == y) / len(y_hat)
                total_acc.append(acc)
                total_loss.append(loss.item())
                total_f1_scores.append(f1_scores)
                total_recall.append(recall)
                total_precision.append(precision)
                total_ap.append(ap)

                del y_hat
                del loss

        total_acc = sum(total_acc) / len(self.val_loader)
        total_loss = sum(total_loss) / len(total_loss)
        total_f1_scores = sum(total_f1_scores) / len(self.val_loader)
        total_recall = sum(total_recall) / len(self.val_loader)
        total_precision = sum(total_precision) / len(self.val_loader)
        total_ap = sum(total_ap) / len(self.val_loader)

        self.model.train()
        return total_loss, total_acc, total_f1_scores, total_recall, total_precision, total_ap

    def train(self):
        EPOCHS = self.epoch_num

        for epoch in range(1, EPOCHS+1):
            print("epoch {}/{}".format(epoch, EPOCHS))

            self.model.train()
            loss = self.train_one_epoch()
            if epoch % 5 == 0:
                vloss, vacc, total_f1_scores, total_recall, total_precision, total_ap = self.eval()
                print("loss: {}, val_loss: {}, accuracy: {}, f1_scores: {}, recall: {}, precision: {}, AP: {}".format(
                    loss, vloss, vacc, total_f1_scores, total_recall, total_precision, total_ap))

    def load_data(self, data_dir):
        loader = DataLoader(data_dir)
        dataset = loader.load_data()
        train_set, val_set = loader.split_data(dataset, self.test_size)
        train_loader = loader.dataloader(
            train_set, self.batch_size, self.shuffle)
        val_loader = loader.dataloader(val_set, len(val_set), shuffle=False)
        return train_loader, val_loader

    def run(self, config):
        use_kfold = True
        use_gpu = config["Global"]["use_gpu"]
        save_model_dir = config["Global"]["save_model_dir"]
        self.data_dir = config["Train"]["data_dir"]
        name_backbone = config["Architecture"]["Backbone"]["name"]
        self.epoch_num = config["Global"]["epoch_num"]
        self.num_classes = config["Architecture"]["Backbone"]["num_classes"]
        self.path_checkpoint = config["Global"]["checkpoint_dir"]
        self.use_checkpoints = config["Global"]["use_checkpoints"]

        self.learning_rate = float(config["Optimizer"]["learning_rate"])

        self.imgs_size = config["Dataloader"]["imgs_size"]
        self.test_size = config["Dataloader"]["test_size"]
        self.batch_size = config["Dataloader"]["batch_size"]
        self.shuffle = config["Dataloader"]["shuffle"]

        self.cuda = self.check_gpu(use_gpu)

        backbone = {
            "ResNet18": ResNet_18,
            "LeNet5": CNN
        }
        assert name_backbone in [
            'ResNet18', 'LeNet5'], "program not support model {}".format(name_backbone)
        self.model = backbone[name_backbone](
            3, self.num_classes).to(device=self.cuda)

        self.criterion = nn.CrossEntropyLoss()
        self.optim = torch.optim.AdamW(params=self.model.parameters())

        if self.use_checkpoints:
            self.load_checkpoint()

        if use_kfold:
            self.train_kfold()
        else:
            self.train_loader, self.val_loader = self.load_data(self.data_dir)
            self.scheduler = OneCycleLR(
                self.optim, max_lr=self.learning_rate, total_steps=self.epoch_num*len(self.train_loader))
            images, labels = next(iter(self.train_loader))
            img_grid = torchvision.utils.make_grid(images)
            self.write.add_image('number_images', img_grid)
            self.train()

        accept_save = input("Save model: [yes/no]:")

        if accept_save == "yes":
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            torch.save(self.model.state_dict(),
                       '{}/pytorch_{}.pth'.format(save_model_dir, timestamp))
            print("Saved!")
