"""
Simplified standalone version of Hyena: https://arxiv.org/abs/2302.10866, designed for quick experimentation.
A complete version is available under `src.models.sequence.hyena`.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def fftconv(u, k, D):
    seqlen = u.shape[-1]
    fft_size = 2 * seqlen

    k_f = torch.fft.rfft(k, n=fft_size) / fft_size
    u_f = torch.fft.rfft(u.to(dtype=k.dtype), n=fft_size)

    if len(u.shape) > 3:
        k_f = k_f.unsqueeze(1)
    y = torch.fft.irfft(u_f * k_f, n=fft_size, norm="forward")[..., :seqlen]

    out = y + u * D.unsqueeze(-1)
    return out.to(dtype=u.dtype)


def fftconv2d(u, k, D):
    """
    Convolutional layer using FFT with a bias term.
    We first compute the FFT of the input and the kernel, then we multiply them element-wise and compute the inverse FFT.

    Parameters:
    u: the input tensor, with shape (batch, channels, height, width)
    k: the convolutional kernel tensor, with shape (channels, kernel_height, kernel_width)
    D: a kind of bias tensor applied element-wise multiplication to the input to obtain a bias term added to the convolutional output. Should be broadcastable to the input tensor.


    """

    # add some assertion statements to check that u and D have the same number of channels
    assert (
        u.shape[1] == D.shape[1]
    ), f"Number of channels in u ({u.shape[1]}) and D ({D.shape[1]}) must be the same"

    img_width = u.shape[-1]
    img_height = u.shape[-2]

    fft_height = 2 * img_height
    fft_width = 2 * img_width

    k_expanded = k.unsqueeze(0).repeat(u.shape[0], 1, 1, 1)

    # now we invoke the convolution theorem, first we compute the FFT of the kernel and the input using torch.fft.rfftn
    k_f = torch.fft.rfftn(k_expanded, s=(fft_height, fft_width), dim=(2, 3)) / (
        fft_height * fft_width
    )
    u_f = torch.fft.rfftn(u, s=(fft_height, fft_width), dim=(2, 3))

    # now make an element-wise multiplication of the FFT of the input and the kernel
    y = torch.fft.irfftn(u_f * k_f, s=(fft_height, fft_width), dim=(2, 3))[
        ..., :img_height, :img_width
    ]

    # add the bias term
    out = y + u * D.unsqueeze(-1).unsqueeze(-1)

    return out.to(dtype=u.dtype)


@torch.jit.script
def mul_sum(q, y):
    return (q * y).sum(dim=1)


class OptimModule(nn.Module):
    """Interface for Module that allows registering buffers/parameters with configurable optimizer hyperparameters.
    NOTE: This should not be any different for 2D Hyena Operator. This is just a helper class for the HyenaFilter class.
    """

    def register(self, name, tensor, lr=None, wd=0.0):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {}
            if lr is not None:
                optim["lr"] = lr
            if wd is not None:
                optim["weight_decay"] = wd
            setattr(getattr(self, name), "_optim", optim)


class Sin(nn.Module):
    """
    A PyTorch module that applies a sinusoidal activation function to its input, with the optional of having a trainable frequency parameter.

    Parameters:
    dim: the dimension of the input tensor
    """

    def __init__(self, dim, w=10, train_freq=True):
        super().__init__()

        # you want to initialize the frequency parameter as a trainable parameter if train_freq is True
        # else you want to initialize it as a constant tensor
        self.freq = (
            nn.Parameter(w * torch.ones(1, dim))
            if train_freq
            else w * torch.ones(1, dim)
        )

    def forward(self, x):

        # apply the sinusoidal activation function to the input tensor element-wise
        return torch.sin(self.freq * x)


class PositionalEmbedding(OptimModule):
    def __init__(self, emb_dim: int, seq_len: int, lr_pos_emb: float = 1e-5, **kwargs):
        """Complex exponential positional embeddings for Hyena filters.
        The embedding consists of [x, sin(f * x), cos(f * x)] for where f spans over the frequency bands.
        """
        super().__init__()

        self.seq_len = seq_len
        # The time embedding fed to the filteres is normalized so that t_f = 1
        t = torch.linspace(0, 1, self.seq_len)[None, :, None]  # 1, L, 1

        if emb_dim > 1:
            bands = (
                emb_dim - 1
            ) // 2  # the number bands is by default equal to half of the embedding dimension - 1 =
        # To compute the right embeddings we use the "proper" linspace
        t_rescaled = torch.linspace(0, seq_len - 1, seq_len)[None, :, None]
        w = (
            2 * math.pi * t_rescaled / seq_len
        )  # 1, L, 1 (this is the scaling factor inside the sinusoidal function)

        f = torch.linspace(1e-4, bands - 1, bands)[
            None, None
        ]  # bands is the number of bands
        z = torch.exp(
            -1j * f * w
        )  # the shaoe if z here should be 1, L, bands (L here is the sequence length) after broadcasting
        z = torch.cat(
            [t, z.real, z.imag], dim=-1
        )  # The resulting z will therefore have the shape [1, seq_len, 1 + 2 * bands]
        self.register(
            "z", z, lr=lr_pos_emb
        )  # z is learnable and will be optimized during training
        self.register(
            "t", t, lr=0.0
        )  # the t is not trainable throughout the training process by setting the learning rate to 0

    def forward(
        self, L
    ):  # what we did here is to precompute a positional encoding sequences
        # this forward function retrieves the positional encoding for the first L elements of the sequence across the bands

        # this is strange behavior because if the sequence is length L < seq_len, then the normalization factor should not have been seq_len, but L
        # based on how the paper described the positional encoding.
        # I think for now it is fine let's just leave it as it is, maybe we will see the reason for this later
        return self.z[:, :L], self.t[:, :L]

        # Actually now I think about it this makes sense, the paper just wasn't too clear about this
        # You HAVE TO set a maximum sequence length for the positional encoding and for the convolutional filter implicit parametrization


class PositionalEmbedding2D(OptimModule):
    def __init__(
        self, emb_dim: int, width: int, height: int, lr_pos_emb: float = 1e-5, **kwargs
    ):
        """Complex exponential positional embeddings for Hyena filters for 2D input.
        We essentially apply positional embedding 1D for x and y dimensions separately and then concatenate them.
        """
        super().__init__()

        self.width = width
        self.height = height

        t_width = torch.linspace(0, 1, self.width)[None, :, None]
        t_height = torch.linspace(0, 1, self.height)[None, :, None]

        if emb_dim > 1:
            bands = (emb_dim - 1) // 2

        t_rescaled_width = torch.linspace(0, width - 1, width)[None, :, None]
        t_rescaled_height = torch.linspace(0, height - 1, height)[
            None, :, None
        ]  # note that this may seen redundant but it is necessary for logging different lr

        w_width = 2 * math.pi * t_rescaled_width / width
        w_height = (
            2 * math.pi * t_rescaled_height / height
        )  # note that this may seen redundant but it is necessary for logging different lr

        f = torch.linspace(1e-4, bands - 1, bands)[None, None]
        z_width = torch.exp(-1j * f * w_width)
        z_height = torch.exp(-1j * f * w_height)

        z_width = torch.cat([t_width, z_width.real, z_width.imag], dim=-1)
        z_height = torch.cat([t_height, z_height.real, z_height.imag], dim=-1)

        self.register("z_width", z_width, lr=lr_pos_emb)
        self.register("z_height", z_height, lr=lr_pos_emb)
        self.register("t_width", t_width, lr=0.0)

    def forward(self, x, y):
        return (
            self.z_width[:, :x],
            self.z_height[:, :y],
            self.t_width[:, :x],
            self.t_height[:, :y],
        )


class ExponentialModulation(OptimModule):
    def __init__(
        self,
        d_model,
        fast_decay_pct=0.3,
        slow_decay_pct=1.5,
        target=1e-2,
        modulation_lr=0.0,
        modulate: bool = True,
        shift: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.modulate = modulate
        self.shift = shift
        max_decay = math.log(target) / fast_decay_pct
        min_decay = math.log(target) / slow_decay_pct
        deltas = torch.linspace(min_decay, max_decay, d_model)[
            None, None
        ]  # so you linearly increment the decay factor for each channel, but these are learnable
        self.register("deltas", deltas, lr=modulation_lr)

    def forward(self, t, x):
        if self.modulate:
            decay = torch.exp(-t * self.deltas.abs())
            x = x * (decay + self.shift)
        return x


class GaussianModulation2D(
    OptimModule
):  # this is the Gaussian modulation for 2D inputs
    def __init__(
        self,
        d_model,
        max_sigma=0.3,  # NOTE that the image coordinates are normalized to the range [0, 1] the sigmas need to be adjusted accordingly
        min_sigma=0.1,
        modulation_lr=0.0,
        modulate: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.modulate = modulate
        deltas = torch.linspace(min_sigma, max_sigma, d_model)[None, None]
        self.register("deltas", deltas, lr=modulation_lr)

    def forward(self, x, y, input):
        """First we compute the value of the Gaussian function at a given location (the covariance matrix is sigma^2 * I)
        Then we multiply the input by the value of the Gaussian function at that location
        """

        if self.modulate:
            scaler = torch.exp(
                -((x - input.size(-1) / 2) ** 2 + (y - input.size(-2) / 2) ** 2)
                / (2 * self.deltas.abs() ** 2)
            )
            output = input * scaler
        return output


class HyenaFilter(OptimModule):
    def __init__(
        self,
        d_model,
        emb_dim=3,  # dim of input to MLP, augments with positional encoding
        order=16,  # width of the implicit MLP
        fused_fft_conv=False,
        seq_len=1024,
        lr=1e-3,
        lr_pos_emb=1e-5,
        dropout=0.0,
        w=1,  # frequency of periodic activations
        wd=0,  # weight decay of kernel parameters
        bias=True,
        num_inner_mlps=2,
        normalized=False,
        **kwargs,
    ):
        """
        Implicit long filter with modulation.

        Args:
            d_model: number of channels in the input
            emb_dim: dimension of the positional encoding (`emb_dim` - 1) // 2 is the number of bands
            order: width of the FFN
            num_inner_mlps: number of inner linear layers inside filter MLP
        """
        super().__init__()
        self.d_model = d_model  # number of channels
        self.use_bias = bias
        self.fused_fft_conv = fused_fft_conv
        self.bias = nn.Parameter(torch.randn(self.d_model))
        self.dropout = nn.Dropout(dropout)

        act = Sin(dim=order, w=w)
        self.emb_dim = emb_dim
        assert (
            emb_dim % 2 != 0 and emb_dim >= 3
        ), "emb_dim must be odd and greater or equal to 3 (time, sine and cosine)"
        self.seq_len = seq_len

        self.pos_emb = PositionalEmbedding(emb_dim, seq_len, lr_pos_emb)

        self.implicit_filter = nn.Sequential(
            nn.Linear(emb_dim, order),
            act,
        )
        for i in range(num_inner_mlps):
            self.implicit_filter.append(
                nn.Linear(order, order)
            )  # the internal layers of the MLP have a fixed width of `order`
            self.implicit_filter.append(act)

        self.implicit_filter.append(nn.Linear(order, d_model, bias=False))

        self.modulation = ExponentialModulation(d_model, **kwargs)

        self.normalized = normalized
        for c in self.implicit_filter.children():
            for name, v in c.state_dict().items():
                optim = {"weight_decay": wd, "lr": lr}
                setattr(getattr(c, name), "_optim", optim)

    def filter(self, L, *args, **kwargs):
        z, t = self.pos_emb(L)
        h = self.implicit_filter(z)
        h = self.modulation(t, h)
        return h  # this is the filter kernel function, and indeed you truncate the first L elements of the positional encoding

    def forward(self, x, L, k=None, bias=None, *args, **kwargs):
        if k is None:
            k = self.filter(L)

        # Ensure compatibility with filters that return a tuple
        k = k[0] if type(k) is tuple else k

        y = fftconv(x, k, bias)
        return y


class HyenaFilter2D(OptimModule):
    def __init__(
        self,
        d_model,
        emb_dim=6,  # dim of input to MLP, augments with positional encoding
        order=16,  # width of the implicit MLP, this is confusing because later, order refers to the depth of the Hyena recurrence
        fused_fft_conv=False,
        width=1024,
        height=1024,
        lr=1e-3,
        lr_pos_emb=1e-5,
        dropout=0.0,
        w=1,  # frequency of periodic activations
        wd=0,  # weight decay of kernel parameters
        bias=True,
        num_inner_mlps=2,
        normalized=False,
        **kwargs,
    ):
        """
        Implicit long filter with modulation for 2D inputs.

        Args:
            d_model: number of channels in the input
            emb_dim: dimension of the positional encoding (`emb_dim` - 1) // 2 is the number of bands
            order: width of the FFN
            num_inner_mlps: number of inner linear layers inside filter MLP
        """
        super().__init__()
        self.d_model = d_model
        self.use_bias = bias
        self.fused_fft_conv = fused_fft_conv
        self.bias = nn.Parameter(torch.randn(self.d_model))
        self.dropout = nn.Dropout(dropout)

        act = Sin(dim=order, w=w)
        self.emb_dim = emb_dim
        assert (
            emb_dim % 2 == 0 and emb_dim >= 6
        ), "emb_dim must be even and greater or equal to 6 (time_x, sine_x and cosine_x, time_y, sine_y, cosine_y)"
        # Note that in the 2D case because we have two dimensions, the emb_dim must be even, essentially double of the 1D case
        self.width = width
        self.height = height

        self.pos_emb = PositionalEmbedding2D(emb_dim, width, height, lr_pos_emb)

        self.implicit_filter = nn.Sequential(
            nn.Linear(emb_dim, order),
            act,
        )
        for i in range(num_inner_mlps):
            self.implicit_filter.append(nn.Linear(order, order))
            self.implicit_filter.append(act)

        self.implicit_filter.append(nn.Linear(order, d_model, bias=False))

        self.modulation = GaussianModulation2D(d_model, **kwargs)

        self.normalized = normalized
        for c in self.implicit_filter.children():
            for name, v in c.state_dict().items():
                optim = {"weight_decay": wd, "lr": lr}
                setattr(getattr(c, name), "_optim", optim)

    def filter(self, x, y, *args, **kwargs):
        # This will give you a filter that is x wide and y high
        z_x, z_y, t_x, t_y = self.pos_emb(x, y)
        z = torch.cat([z_x, z_y], dim=-1)
        t = torch.cat([t_x, t_y], dim=-1)
        h = self.implicit_filter(z)
        h = self.modulation(x, y, h)
        return h

    def forward(self, input, x, y, k=None, bias=None, *args, **kwargs):
        if k is None:
            k = self.filter(x, y)

        # Ensure compatibility with filters that return a tuple
        k = k[0] if type(k) is tuple else k

        y = fftconv2d(input, k, bias)
        return y


class CustomProjection2D(nn.Module):
    def __init__(self, d_model, inner_width):
        super().__init__()
        self.in_proj = nn.Linear(d_model, inner_width)

    def forward(self, u):
        b, d, h, w = u.shape

        # Reshape u to (b * h * w, d), combining batches, height, and width
        u = u.permute(0, 2, 3, 1).reshape(b * h * w, d)

        # Apply the linear transformation
        u_transformed = self.in_proj(u)

        # Reshape back to (b, h, w, inner_width)
        u_transformed = u_transformed.view(b, h, w, -1).permute(0, 3, 1, 2)

        return u_transformed


class HyenaOperator2D(nn.Module):
    def __init__(
        self,
        d_model,
        width_max,
        height_max,
        order=2,  # here this order refers to the depth of the Hyena recurrence
        filter_order=64,
        dropout=0.0,
        filter_dropout=0.0,
        **filter_args,
    ):
        """2 Dimensional extension of the Hyena operator for 1D sequences."""

        super().__init__()
        self.d_model = d_model  # d_model refers to the dimension of the input token embedding for each token or patch
        self.width_max = width_max
        self.height_max = height_max
        self.order = order
        inner_width = d_model * (order + 1)
        self.dropout = nn.Dropout(dropout)
        self.in_proj = CustomProjection2D(
            d_model, inner_width
        )  # you are not changing the dimension of the input token embedding
        self.out_proj = CustomProjection2D(d_model, d_model)

        # This part is what goes into the initial projection before the Hyena recurrence
        short_kernel_size = 3
        short_kernel_padding = (short_kernel_size - 1) // 2

        self.short_filter = nn.Conv2d(
            inner_width,
            inner_width,
            short_kernel_size,
            padding=short_kernel_padding,
            groups=inner_width,
        )  # 3 here is the kernel size of the convolutional layer

        self.filter_fn = HyenaFilter2D(
            d_model * (order - 1),
            order=filter_order,
            width=width_max,
            height=height_max,
            channels=1,
            dropout=filter_dropout,
            **filter_args,
        )

    def forward(self, u, *args, **kwargs):
        width = u.size(-2)
        height = u.size(-3)
        width_filter = min(width, self.width_max)
        height_filter = min(height, self.height_max)
        u = self.in_proj(u)
        u = rearrange(u, "b h w d -> b d h w")

        uc = self.short_filter(u)[..., :height_filter, :width_filter]
        *x, v = uc.split(self.d_model, dim=1)

        k = self.filter_fn.filter(width_filter, height_filter)[0]
        k = rearrange(k, "d (h w) -> d w h", w=height_filter)
        bias = rearrange(self.filter_fn.bias, "(h w) -> w h", w=height_filter)

        for o, x_i in enumerate(reversed(x[1:])):
            v = self.dropout(v * x_i)  # it seems like the default dropout is 0.0
            v = self.filter_fn(v, width_filter, height_filter, k=k[o], bias=bias[o])

        y = rearrange(v * x[0], "b d h w -> b h w d")

        y = self.out_proj(y)
        return y


class HyenaOperator(nn.Module):
    def __init__(
        self,
        d_model,
        l_max,
        order=2,
        filter_order=64,
        dropout=0.0,
        filter_dropout=0.0,
        **filter_args,
    ):
        r"""
        Hyena operator described in the paper https://arxiv.org/pdf/2302.10866.pdf

        Args:
            d_model (int): Dimension of the input and output embeddings (width of the layer)
            l_max: (int): Maximum input sequence length. Defaults to None
            order: (int): Depth of the Hyena recurrence. Defaults to 2
            dropout: (float): Dropout probability. Defaults to 0.0
            filter_dropout: (float): Dropout probability for the filter. Defaults to 0.0
        """
        super().__init__()
        self.d_model = d_model
        self.l_max = l_max
        self.order = order
        inner_width = d_model * (order + 1)
        self.dropout = nn.Dropout(dropout)
        self.in_proj = nn.Linear(d_model, inner_width)
        self.out_proj = nn.Linear(d_model, d_model)

        self.short_filter = nn.Conv1d(
            inner_width, inner_width, 3, padding=2, groups=inner_width
        )
        self.filter_fn = HyenaFilter(
            d_model * (order - 1),
            order=filter_order,
            seq_len=l_max,  # Indeed, this confirms my understanding earlier that this should be the maximum sequence length
            channels=1,
            dropout=filter_dropout,
            **filter_args,
        )

    def forward(self, u, *args, **kwargs):
        l = u.size(-2)
        l_filter = min(
            l, self.l_max
        )  # here is how we decide the length of the filter kernel
        u = self.in_proj(u)
        u = rearrange(u, "b l d -> b d l")

        uc = self.short_filter(u)[..., :l_filter]
        *x, v = uc.split(self.d_model, dim=1)

        k = self.filter_fn.filter(l_filter)[0]
        k = rearrange(k, "l (o d) -> o d l", o=self.order - 1)
        bias = rearrange(self.filter_fn.bias, "(o d) -> o d", o=self.order - 1)

        for o, x_i in enumerate(reversed(x[1:])):
            v = self.dropout(v * x_i)
            v = self.filter_fn(v, l_filter, k=k[o], bias=bias[o])

        y = rearrange(v * x[0], "b d l -> b l d")

        y = self.out_proj(y)
        return y


if __name__ == "__main__":
    layer = HyenaOperator(d_model=512, l_max=1024, order=2, filter_order=64)
    x = torch.randn(1, 1024, 512, requires_grad=True)
    y = layer(x)

    print(x.shape, y.shape)

    grad = torch.autograd.grad(y[:, 10, :].sum(), x)[0]
    print('Causality check: gradients should not flow "from future to past"')
    print(grad[0, 11, :].sum(), grad[0, 9, :].sum())
