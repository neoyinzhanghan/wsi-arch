import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score, AUROC
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_lightning.loggers import TensorBoardLogger
from wsiarch.data.dataloaders import (
    FeatureImageDataset,
    create_data_loaders,
    FeatureImageDataModule,
)


class MultiHeadAttentionClassifier(nn.Module):
    def __init__(
        self, d_model=2048, num_heads=8, num_classes=2, height_max=445, width_max=230, use_flash_attention=True
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_flash_attention = use_flash_attention
        self.num_classes = num_classes
        self.height_max = height_max
        self.width_max = width_max

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.class_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.classifier = nn.Linear(d_model, num_classes)

    def get_positional_encoding(self):
        # Initialize a positional encoding tensor with zeros, shape (height, width, d_model)
        height = self.height_max
        width = self.width_max
        pos_encoding = torch.zeros(height, width, self.d_model)
        assert pos_encoding.shape == (
            height,
            width,
            self.d_model,
        ), f"pos_encoding shape mismatch: {pos_encoding.shape}"

        # Generate a range of positions for height and width
        y_pos = (
            torch.arange(height).unsqueeze(1).float()
        )  # Unsqueeze to make it a column vector
        assert y_pos.shape == (height, 1), f"y_pos shape mismatch: {y_pos.shape}"

        x_pos = (
            torch.arange(width).unsqueeze(1).float()
        )  # Unsqueeze to make it a row vector
        assert x_pos.shape == (width, 1), f"x_pos shape mismatch: {x_pos.shape}"

        # Calculate the divisor term for the positional encoding formula
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2).float()
            * -(math.log(10000.0) / self.d_model)
        )
        assert div_term.shape == (
            self.d_model // 2,
        ), f"div_term shape mismatch: {div_term.shape}"

        # Apply the sine function to the y positions and expand to match (height, 1, d_model // 2)
        pos_encoding_y_sin = (
            torch.sin(y_pos * div_term)
            .unsqueeze(1)
            .expand(height, width, self.d_model // 2)
        )
        assert pos_encoding_y_sin.shape == (
            height,
            width,
            self.d_model // 2,
        ), f"pos_encoding_y_sin shape mismatch: {pos_encoding_y_sin.shape}"

        # Apply the cosine function to the y positions and expand to match (height, 1, d_model // 2)
        pos_encoding_y_cos = (
            torch.cos(y_pos * div_term)
            .unsqueeze(1)
            .expand(height, width, self.d_model // 2)
        )
        assert pos_encoding_y_cos.shape == (
            height,
            width,
            self.d_model // 2,
        ), f"pos_encoding_y_cos shape mismatch: {pos_encoding_y_cos.shape}"

        # Apply the sine function to the x positions and expand to match (1, width, d_model // 2)

        pos_encoding_x_sin = (
            torch.sin(x_pos * div_term)
            .unsqueeze(0)
            .expand(height, width, self.d_model // 2)
        )
        assert pos_encoding_x_sin.shape == (
            height,
            width,
            self.d_model // 2,
        ), f"pos_encoding_x_sin shape mismatch: {pos_encoding_x_sin.shape}"

        # Apply the cosine function to the x positions and expand to match (1, width, d_model // 2)
        pos_encoding_x_cos = (
            torch.cos(x_pos * div_term)
            .unsqueeze(0)
            .expand(height, width, self.d_model // 2)
        )
        assert pos_encoding_x_cos.shape == (
            height,
            width,
            self.d_model // 2,
        ), f"pos_encoding_x_cos shape mismatch: {pos_encoding_x_cos.shape}"

        # Combine the positional encodings
        pos_encoding[:, :, 0::2] = pos_encoding_y_sin + pos_encoding_x_sin
        pos_encoding[:, :, 1::2] = pos_encoding_y_cos + pos_encoding_x_cos

        assert pos_encoding.shape == (
            height,
            width,
            self.d_model,
        ), f"pos_encoding shape mismatch: {pos_encoding.shape}"

        return pos_encoding

    def forward(self, x):
        batch_size, d_model, height, width = x.shape

        x = x.permute(0, 2, 3, 1).view(batch_size, height * width, d_model)
        pos_encoding = self.get_positional_encoding().to(x.device)
        x = x + pos_encoding.view(height * width, d_model)

        class_tokens = self.class_token.expand(batch_size, -1, -1)
        x = torch.cat([class_tokens, x], dim=1)

        q = (
            self.q_proj(x)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        if self.use_flash_attention:
            attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        else:
            attn_output = torch.matmul(
                F.softmax(
                    torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim),
                    dim=-1,
                ),
                v,
            )

        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        )
        output = self.out_proj(attn_output)

        class_token_output = output[:, 0]
        logits = self.classifier(class_token_output)
        return logits


class MultiHeadAttentionClassifierPL(pl.LightningModule):
    def __init__(
        self, d_model, num_heads, num_classes, use_flash_attention=True, num_epochs=10
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = MultiHeadAttentionClassifier(
            d_model, num_heads, num_classes, use_flash_attention
        )

        self.train_accuracy = Accuracy(num_classes=num_classes, task="multiclass")
        self.val_accuracy = Accuracy(num_classes=num_classes, task="multiclass")
        self.test_accuracy = Accuracy(num_classes=num_classes, task="multiclass")

        self.train_f1 = F1Score(num_classes=num_classes, task="multiclass")
        self.val_f1 = F1Score(num_classes=num_classes, task="multiclass")
        self.test_f1 = F1Score(num_classes=num_classes, task="multiclass")

        self.train_auroc = AUROC(num_classes=num_classes, task="multiclass")
        self.val_auroc = AUROC(num_classes=num_classes, task="multiclass")
        self.test_auroc = AUROC(num_classes=num_classes, task="multiclass")

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss)
        self.log("train_accuracy", self.train_accuracy(logits, y))
        self.log("train_f1", self.train_f1(logits, y))
        self.log("train_auroc", self.train_auroc(logits, y))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss)
        self.log("val_accuracy", self.val_accuracy(logits, y))
        self.log("val_f1", self.val_f1(logits, y))
        self.log("val_auroc", self.val_auroc(logits, y))

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("test_loss", loss)
        self.log("test_accuracy", self.test_accuracy(logits, y))
        self.log("test_f1", self.test_f1(logits, y))
        self.log("test_auroc", self.test_auroc(logits, y))

    def on_train_epoch_end(self):
        scheduler = self.lr_schedulers()
        current_lr = scheduler.get_last_lr()[0]
        self.log("lr", current_lr)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.0001)
        scheduler = CosineAnnealingLR(
            optimizer, T_max=self.hparams.num_epochs, eta_min=0
        )
        return [optimizer], [scheduler]


def train_model(data_dir, num_gpus=3, num_epochs=10):
    data_module = FeatureImageDataModule(
        root_dir=data_dir,
        metadata_file=os.path.join(data_dir, "metadata.csv"),
        width_max=230,
        height_max=445,
        batch_size=1,
        num_workers=24,
    )

    model = MultiHeadAttentionClassifierPL(
        d_model=1024,
        num_heads=8,
        num_classes=2,
        use_flash_attention=True,
        num_epochs=num_epochs,
    )

    logger = TensorBoardLogger("lightning_logs", name="multihead_attention")

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        logger=logger,
        devices=num_gpus,
        accelerator="gpu",
    )
    trainer.fit(model, data_module)
    trainer.test(model, data_module.test_dataloader())


if __name__ == "__main__":
    data_dir = "/media/hdd1/neo/LUAD-LUSC_FI_ResNet"
    train_model(data_dir=data_dir)
