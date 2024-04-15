import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, AUROC
from torch.optim.lr_scheduler import CosineAnnealingLR


class SimpleConvNet(nn.Module):
    def __init__(
        self,
        num_classes,
        feature_dim,
        out_channels,
        kernel_size=5,
        stride=1,
        padding=0,
        output_size=(10, 10),
    ):
        super(SimpleConvNet, self).__init__()
        # Define a single convolutional layer
        # Assuming input tensor shape is (N, feature_dim, H, W) where N is the batch size
        self.conv1 = nn.Conv2d(
            in_channels=feature_dim,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        # Define a pooling layer that outputs a fixed size
        self.pool = nn.AdaptiveAvgPool2d(output_size)

        # Calculate the number of features going into the linear layer from the output size
        self.num_features_before_fc = out_channels * output_size[0] * output_size[1]

        # Define a linear layer that dynamically adapts to the output of the pooling layer
        self.fc1 = nn.Linear(self.num_features_before_fc, num_classes)

    def forward(self, x):
        # Apply convolutional layer with ReLU activation
        x = F.relu(self.conv1(x))

        # Apply adaptive pooling
        x = self.pool(x)

        # Flatten the output from the pooling layer
        x = x.view(x.size(0), -1)  # Flatten all dimensions except batch

        # Apply linear layer to produce final outputs
        x = self.fc1(x)

        return x


class SimpleConvNetModelPL(pl.LightningModule):
    def __init__(
        self,
        num_classes=2,
        feature_dim=3,
        out_channels=16,
        kernel_size=3,
        stride=1,
        padding=0,
        lr=1e-3,
        num_epochs=10,
    ):
        super().__init__()
        # Model Configuration
        self.save_hyperparameters()

        # Model architecture
        self.model = SimpleConvNet(
            num_classes=num_classes,
            feature_dim=feature_dim,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        # Metrics
        task = "binary" if num_classes == 2 else "multiclass"
        self.train_accuracy = Accuracy(
            num_classes=num_classes, average="macro", multiclass=True
        )
        self.val_accuracy = Accuracy(
            num_classes=num_classes, average="macro", multiclass=True
        )
        self.train_auroc = AUROC(
            num_classes=num_classes, average="macro", compute_on_step=False
        )
        self.val_auroc = AUROC(
            num_classes=num_classes, average="macro", compute_on_step=False
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        self.train_accuracy(y_hat, y)
        self.train_auroc(y_hat, y)
        self.log("train_acc", self.train_accuracy, on_step=False, on_epoch=True)
        self.log("train_auroc", self.train_auroc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.val_accuracy(y_hat, y)
        self.val_auroc(y_hat, y)
        self.log("val_acc", self.val_accuracy, on_step=False, on_epoch=True)
        self.log("val_auroc", self.val_auroc, on_step=False, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        self.log("val_acc_epoch", self.val_accuracy.compute())
        self.log("val_auroc_epoch", self.val_auroc.compute())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        scheduler = CosineAnnealingLR(
            optimizer, T_max=self.hparams.num_epochs, eta_min=0
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
