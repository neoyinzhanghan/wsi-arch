import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score, AUROC
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_lightning.loggers import TensorBoardLogger
from wsiarch.data.dataloader_seq import H5DataModule


class Attn(
    nn.Module
):  # the main purpose of having this particular function is to have a named layer in the architecture for the flash attention
    def __init__(self, head_dim, use_flash_attention):
        super(Attn, self).__init__()
        self.head_dim = head_dim
        self.use_flash_attention = use_flash_attention

    def forward(self, q, k, v):
        if self.use_flash_attention:
            attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        else:
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(
                self.head_dim
            )
            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_output = torch.matmul(attn_probs, v)
        return attn_output


class MultiHeadAttentionClassifier(nn.Module):
    def __init__(
        self,
        d_model=2048,
        num_heads=8,
        num_classes=2,
        length_max=58182,
        use_flash_attention=True,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_flash_attention = use_flash_attention
        self.num_classes = num_classes
        self.length_max = length_max

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        head_dim = d_model // num_heads

        self.attn = Attn(head_dim=head_dim, use_flash_attention=use_flash_attention)

        self.class_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x, p):
        batch_size, d_model, length = x.shape

        assert d_model == self.d_model, "Input feature depth must be equal to d_model"
        assert length == self.length_max, "Input length must be equal to length_max"

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

        attn_output = self.attn(q, k, v)

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

    def forward(self, x, p):
        return self.model(x, p)

    def training_step(self, batch, batch_idx):
        x, p, y = batch
        logits = self(x, p)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss)
        self.log("train_accuracy", self.train_accuracy(logits, y))
        self.log("train_f1", self.train_f1(logits, y))
        self.log("train_auroc", self.train_auroc(logits, y))
        return loss

    def validation_step(self, batch, batch_idx):
        x, p, y = batch
        logits = self(x, p)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss)
        self.log("val_accuracy", self.val_accuracy(logits, y))
        self.log("val_f1", self.val_f1(logits, y))
        self.log("val_auroc", self.val_auroc(logits, y))

    def test_step(self, batch, batch_idx):
        x, p, y = batch
        logits = self(x, p)
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


def train_model(metadata_path, num_gpus=3, num_epochs=10):
    data_module = H5DataModule(
        metadata_path=metadata_path,
        length_max=58182,
        batch_size=1,
        num_workers=24,
    )

    model = MultiHeadAttentionClassifierPL(
        d_model=2048,
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
    metadata_path = "/media/hdd1/neo/LUAD-LUSC_UNI_metadata.csv"
    train_model(metadata_path=metadata_path)
