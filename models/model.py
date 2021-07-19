import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim

class Model(pl.LightningModule):
    def __init__(self, model):
        super().__init__()

        self.mode = model.mode
        if self.mode == 'classifier':
            self.loss_function = nn.CrossEntropyLoss()
        elif self.mode == 'segmentator':
            self.loss_function = nn.L1Loss()
        else:
            raise NotImplementedError

        self.model = model

    def forward(self, x):
        logits = self.model(x)

        if self.mode == 'classifier':
            pred = torch.softmax(logits, dim=-1)
            pred = pred.argmax(-1)
        elif self.mode == 'segmentator':
            pred = torch.where(logits < 0.5, 0, 1)
        else:
            raise NotImplementedError

        return logits, pred

    def training_step(self, batch, batch_idx):
        x, y = batch

        if self.mode == 'segmentator':
            y = torch.where(y < 0.5, 0, 1)

        logits, pred = self.forward(x)
        loss = self.loss_function(logits, y)

        y = y.flatten()
        pred = pred.flatten()

        outs = {"loss" : loss}
        with torch.no_grad():
            outs["acc"] = (y == pred).float().mean()

        return outs

    def training_epoch_end(self, outs):
        loss = torch.stack([m['loss'] for m in outs]).mean()
        self.log('train_loss', loss, prog_bar=True)
        acc = torch.stack([m['acc'] for m in outs]).mean()
        self.log('train_acc', acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch

        if self.mode == 'segmentator':
            y = torch.where(y < 0.5, 0, 1)

        logits, pred = self.forward(x)
        loss = self.loss_function(logits, y)

        y = y.flatten()
        pred = pred.flatten()

        outs = {"loss" : loss}
        outs["acc"] = (y == pred).float().mean()

        return outs

    def validation_epoch_end(self, outs):
        loss = torch.stack([m['loss'] for m in outs]).mean()
        self.log('valid_loss', loss, prog_bar=True)
        acc = torch.stack([m['acc'] for m in outs]).mean()
        self.log('valid_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

        decay = 0.1
        step = 40

        lr_lambda = lambda epoch: decay ** (epoch // step)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {"optimizer" : optimizer, "scheduler" : scheduler}
