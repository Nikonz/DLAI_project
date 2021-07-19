import torch
import torchvision
import torchvision.transforms as transforms
import pytorch_lightning as pl

from models.model import Model
from datamodule import CifarDatamodule, StarDatamodule, CarsDatamodule

from models.AllCNNC import AllCNNC, AllCNNCChanPool
from models.GAllCNNC import GAllCNNC, GAllCNNCRotPool
from models.Unet import Unet
from models.GUnet import GUnet, GUnetGSkipConn
from models.ResNet18 import ResNet18

EPOCHS = 100

callbacks = [
        pl.callbacks.ModelCheckpoint(
            filename='{epoch}-{valid_acc}',
            monitor='valid_acc',
            mode='max',
            save_last=True)
]

#datamodule = CifarDatamodule()
#datamodule = StarDatamodule()
datamodule = CarsDatamodule(batch_size=32)

#model = Model(ResNet18(1))
#model = Model(AllCNNC(1, 2))
#model = Model(AllCNNCChanPool(1, 2))
#model = Model(GAllCNNC(1, 2))
#model = Model(GAllCNNCRotPool(1, 2))
#model = Model(Unet(3, 1, 2, 28))
#model = Model(GUnet(3, 1, 2, 13))
model = Model(GUnetGSkipConn(3, 1, 2, 13))

trainer = pl.Trainer(gpus=None, val_check_interval=1.0, max_epochs=EPOCHS, callbacks=callbacks)
trainer.fit(model, datamodule=datamodule)
