import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics import Accuracy, F1Score, AUROC
from wsiarch.models.components.hyena import HyenaOperator2D
from wsiarch.data.dataloaders import (
    FeatureImageDataset,
    create_data_loaders,
    FeatureImageDataModule,
)


# OptimModule definition
class OptimModule(nn.Module):
    def register(self, name, tensor, lr=None):
        if lr is not None:
            param = nn.Parameter(tensor)
            self.register_parameter(name, param)
            setattr(self, name + "_lr", lr)
        else:
            self.register_buffer(name, tensor)


# PositionalEmbedding2D definition
class PositionalEmbedding2D(OptimModule):
    def __init__(
        self, emb_dim: int, height: int, width: int, lr_pos_emb: float = 1e-5, **kwargs
    ):
        super().__init__()

        self.width = width
        self.height = height

        t_width = torch.linspace(0, 1, self.width)[None, :, None]
        t_height = torch.linspace(0, 1, self.height)[None, :, None]

        assert (
            emb_dim % 2 == 0 and emb_dim >= 6
        ), "emb_dim must be even and greater or equal to 6 (time_x, sine_x and cosine_x, time_y, sine_y, cosine_y)"

        if emb_dim > 1:
            bands = (emb_dim - 2) // 4

        t_rescaled_width = torch.linspace(0, width - 1, width)[None, :, None]
        t_rescaled_height = torch.linspace(0, height - 1, height)[None, :, None]

        w_width = 2 * math.pi * t_rescaled_width / width
        w_height = 2 * math.pi * t_rescaled_height / height

        f = torch.linspace(1e-4, bands - 1, bands)[None, None]
        z_width = torch.exp(-1j * f * w_width)
        z_height = torch.exp(-1j * f * w_height)

        z_width = torch.cat([t_width, z_width.real, z_width.imag], dim=-1)
        z_height = torch.cat([t_height, z_height.real, z_height.imag], dim=-1)

        self.register("z_width", z_width, lr=lr_pos_emb)
        self.register("z_height", z_height, lr=lr_pos_emb)
        self.register("t_width", t_width, lr=0.0)
        self.register("t_height", t_height, lr=0.0)

    def forward(self, x, y):
        return (
            self.z_height[:, :x],
            self.z_width[:, :y],
            self.t_height[:, :x],
            self.t_width[:, :y],
        )


# MultiHeadAttentionClassifier definition
class MultiHeadAttentionClassifier(nn.Module):
    def __init__(
        self, d_model, num_heads, num_classes, height, width, use_flash_attention=True
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_flash_attention = use_flash_attention

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.class_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.classifier = nn.Linear(d_model, num_classes)

        self.pos_embedding = PositionalEmbedding2D(d_model, height, width)

    def forward(self, x):
        batch_size, d_model, height, width = x.shape

        # Reshape input
        x = x.permute(0, 2, 3, 1).view(batch_size, height * width, d_model)

        # Add positional encoding
        z_height, z_width, _, _ = self.pos_embedding(height, width)
        pos_encoding = torch.cat(
            [
                z_height.expand(width, -1, -1).transpose(0, 1),
                z_width.expand(height, -1, -1),
            ],
            dim=-1,
        )
        pos_encoding = pos_encoding.view(height * width, -1).unsqueeze(0)
        x = x + pos_encoding

        # Add class token
        class_tokens = self.class_token.expand(batch_size, -1, -1)
        x = torch.cat([class_tokens, x], dim=1)

        # Multi-head attention
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

        # Extract class token and classify
        class_token_output = output[:, 0]
        logits = self.classifier(class_token_output)
        probs = F.softmax(logits, dim=-1)

        return probs


# PyTorch Lightning module
class MultiHeadAttentionClassifierPL(pl.LightningModule):
    def __init__(
        self,
        d_model,
        num_heads,
        num_classes,
        height,
        width,
        use_flash_attention=True,
        num_epochs=10,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = MultiHeadAttentionClassifier(
            d_model=d_model,
            num_heads=num_heads,
            num_classes=num_classes,
            height=height,
            width=width,
            use_flash_attention=use_flash_attention,
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

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.001)
        scheduler = CosineAnnealingLR(
            optimizer, T_max=self.hparams.num_epochs, eta_min=0
        )
        return [optimizer], [scheduler]


# Main training loop
def train_model(data_dir, num_gpus=3, num_epochs=10):
    data_module = FeatureImageDataModule(
        root_dir=data_dir,
        metadata_file=os.path.join(data_dir, "metadata.csv"),
        width_max=230,
        height_max=445,
        batch_size=1,
        num_workers=9,
    )

    model = MultiHeadAttentionClassifierPL(
        d_model=256,
        num_heads=8,
        num_classes=10,
        height=24,
        width=24,
        use_flash_attention=True,
        num_epochs=num_epochs,
    )

    # Logger
    logger = TensorBoardLogger("lightning_logs", name="attention")

    # Trainer configuration for distributed training
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        logger=logger,
        devices=num_gpus,
        accelerator="gpu",  # 'ddp' for DistributedDataParallel
    )
    trainer.fit(model, data_module)
    trainer.test(model, data_module.test_dataloader())


if __name__ == "__main__":
    data_dir = "/media/hdd1/neo/LUAD-LUSC_FI"
    train_model(data_dir=data_dir)
