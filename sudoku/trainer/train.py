import torch
import torch.nn as nn
from datetime import datetime
from tqdm import tqdm
from sudoku.trainer.loader.dataloader import DataLoader
from sudoku.trainer.model.backbone.resnet18 import ResNet_18
from sudoku.trainer.model.backbone.Lenet import CNN
from torchsummary import summary
from sudoku import load_config


class Train:

    def __ini__(self):
        pass

    def check_gpu(self, use_gpu):
        if use_gpu and not torch.cuda.is_available():
            print("Device not supported gpu")
            use_gpu = False
        use_cuda = "cuda" if use_gpu else "cpu"
        return torch.device(use_cuda)

    def train_one_epoch(self, optim, loss_fn):
        avg_loss = 0
        losses = []
        accuracy = []
        for i, data in tqdm(enumerate(self.train_loader)):
            x, y = data
            y_onehot = nn.functional.one_hot(
                y, num_classes=self.num_classes).type(torch.float32)
            x, y = x.to(device=self.cuda), y.to(device=self.cuda)
            y_onehot = y_onehot.to(device=self.cuda)

            y_hat = self.model(x)
            loss = loss_fn(y_hat, y_onehot)
            loss.backward()
            optim.step()
            acc = sum(torch.argmax(y_hat, 1) == y) / len(y)
            losses.append(loss.item())
            accuracy.append(acc)

        avg_loss = sum(losses) / len(losses)
        avg_acc = sum(accuracy) / len(accuracy)
        return avg_loss, avg_acc

    def train(self):

        # writer = SummaryWriter('runs/numper_trainer_{}'.format(timestamp))
        loss_fn = nn.CrossEntropyLoss()
        optim = torch.optim.Adam(params=self.model.parameters(), lr=0.00001)

        self.history = {
            "loss": [],
            "val_loss": [],
            "acc": [],
            "val_acc": []
        }

        epochs_number = 0
        EPOCHS = self.epoch_num

        for epoch in range(EPOCHS):
            print("epoch {}/{}".format(epoch, EPOCHS))

            self.model.train()
            avg_loss, avg_acc = self.train_one_epoch(optim, loss_fn)
            self.history["loss"].append(avg_loss)
            self.history["acc"].append(avg_acc.to(device="cpu").tolist())
            self.model.train(False)

            val_accs = []
            val_losses = []
            for vdata in self.val_loader:
                vinputs, vlabels = vdata
                vinputs = vinputs.to(device=self.cuda)
                vlabels = vlabels.to(device=self.cuda)
                voutputs = self.model(vinputs)
                val_acc = sum(torch.argmax(voutputs, 1)
                              == vlabels) / len(vlabels)
                val_accs.append(val_acc)
                vloss = loss_fn(voutputs, vlabels)
                val_losses.append(vloss.item())

            avg_vloss = sum(val_losses)/len(val_losses)
            avg_vacc = sum(val_accs)/len(val_accs)
            self.history["val_loss"].append(avg_vloss)
            self.history["val_acc"].append(avg_vacc.to(device="cpu").tolist())
            print('LOSS train {} valid {} - Accuracy train {} valid {}'.format(avg_loss,
                                                                               avg_vloss, avg_acc, avg_vacc))

            epochs_number += 1

    def load_data(self, data_dir):
        loader = DataLoader(data_dir)
        dataset = loader.load_data()
        train_set, val_set = loader.split_data(dataset, 0.2)
        train_loader = loader.dataloader(train_set)
        val_loader = loader.dataloader(val_set)
        return train_loader, val_loader

    def run(self, config):
        use_gpu = config["Global"]["use_gpu"]
        save_model_dir = config["Global"]["save_model_dir"]
        data_dir = config["Train"]["data_dir"]
        name_backbone = config["Architecture"]["Backbone"]["name"]
        self.epoch_num = config["Global"]["epoch_num"]
        self.num_classes = config["Global"]["num_classes"]

        self.cuda = self.check_gpu(use_gpu)

        backbone = {
            "ResNet18": ResNet_18,
            "LeNet5": CNN
        }
        assert name_backbone in [
            'ResNet18', 'LeNet5'], "program not support model {}".format(name_backbone)
        self.model = backbone[name_backbone](
            3, self.num_classes).to(device=self.cuda)

        self.train_loader, self.val_loader = self.load_data(data_dir)
        self.train()
        accept_save = input("Save model: [yes/no]:")

        if accept_save == "yes":
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            torch.save(self.model.state_dict(),
                       '{}/pytorch_{}.pth'.format(save_model_dir, timestamp))
            print("Saved!")
