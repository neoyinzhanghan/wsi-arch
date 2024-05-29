import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.models import resnet50
from torchmetrics import Accuracy, AUROC
from torch.optim.lr_scheduler import CosineAnnealingLR

class CustomResNet50(nn.Module):
    def __init__(self, num_classes, feature_dim):
        super(CustomResNet50, self).__init__()
        # Load a pre-trained ResNet50 model
        original_resnet50 = resnet50(pretrained=True)
        
        # Modify the first convolution layer to accept feature_dim channels
        self.features = nn.Sequential(
            nn.Conv2d(feature_dim, 64, kernel_size=7, stride=2, padding=3, bias=False),
            *list(original_resnet50.children())[1:-2]  # Exclude the original first conv layer and the final fully connected layer
        )
        
        # Replace the last fully connected layer with one that outputs num_classes
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class ResNet50ModelPL(pl.LightningModule):
    def __init__(
        self,
        num_classes=2,
        feature_dim=1024,
        lr=1e-3,
        num_epochs=10,
    ):
        super().__init__()
        # Save hyperparameters
        self.save_hyperparameters()
        
        # Model architecture
        self.model = CustomResNet50(num_classes=num_classes, feature_dim=feature_dim)

        # Metrics
        task = "binary" if num_classes == 2 else "multiclass"
        self.train_accuracy = Accuracy(num_classes=num_classes, average="macro", multiclass=True)
        self.val_accuracy = Accuracy(num_classes=num_classes, average="macro", multiclass=True)
        self.train_auroc = AUROC(num_classes=num_classes, average="macro", compute_on_step=False)
        self.val_auroc = AUROC(num_classes=num_classes, average="macro", compute_on_step=False)

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
        scheduler = CosineAnnealingLR(optimizer, T_max=self.hparams.num_epochs, eta_min=0)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
