import pytorch_lightning as pl
from wsiarch.models.components.hyena import HyenaOperator2D


class HyenaModelPL(pl.LightningModule):
    """PyTorch Lightning Module for Hyena model.
    It is applied to 2D data with any depth of tokens.

    The model directly applies the Hyena operator to the input data.
    """

    def __init__(
        self,
        d_model,
        width_max,
        height_max,
        order=2,
        filter_order=64,
        dropout=0.0,
        filter_dropout=0.0,
    ):
        super().__init__()
        # Save hyperparameters
        self.save_hyperparameters()

        # first setup the HYENA layer
        hyena_layer = HyenaOperator2D(
            d_model=d_model,
            width_max=width_max,
            height_max=height_max,
            order=order,
            filter_order=filter_order,
            dropout=dropout,
            filter_dropout=filter_dropout,
        )

    pass
