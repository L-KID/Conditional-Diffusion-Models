import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """
    Apply QKV self-attention with a reidual connection.

    Input:
        x: tensor of shape (N, channels, H, W)
    Output:
        tensor of shape (N, channels, H, W)
    Args:
        channels (int): number of input channels
    """
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


class DoubleConv(nn.Module):
    """
    Applies two conv blocks with/without residual connection. 

    Input:
        x: tensor of shape (N, in_channels, H, W)
    Output:
        tensor of shape (N, out_channels, H, W)
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        mid_channels (int or None): number of middle channels
        residual (True or False): with or without residual connection
    """
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    """
    Downsample a given tensor by a factor of 2 (MaxPool), and an embedding layer for timestep and class condition.

    Input: 
        x: tensor of shape (N, in_channels, H, W)
        t: timestep embedding or the embedding of timestep and class
    Output:
        tensor of shape (N, in_channels, H // 2, W // 2)
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        emb_dim (int): embedding dimension
    """
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    """
    Upsample a given tensor by a factor of 2.

    Input:
        x: tensor of shape (N, in_channels, H, W)
        skip_x: tensor that the upsample will match the size of it
        t: timestep embedding or the embedding of timestep and class
    Output:
        tensor of shape (N, out_channels, H * 2, W * 2)
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        emb_dim (int): embedding dimesion
    """
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        # Padding the upsampled x if there's a size difference from the target size (e.g. odd input height and width)
        diffY = skip_x.size()[2] - x.size()[2]
        diffX = skip_x.size()[3] - x.size()[3]

        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UNet_conditional(nn.Module):
    """
    UNet model used to estimae noise.

    Input:
        x: tensor of shape (N, c_in, H, W)
        t: timesteps tensor of shape (N)
        y: classes tensor of shape (N) or None if the block doesn't use class conditioning
    Output:
        tensor of shape (N, c_out, H, W)
    Args:
        c_in (int): number of image channels
        c_out (int): number of image channels
        time_dim (int): number of time embedding dimension
        num_classes (int): number of dataset classes
        device (string): use CPU or GPU
    """
    def __init__(self, c_in=3, c_out=3, time_dim=256, num_classes=None, device="cpu"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.sa1 = SelfAttention(256, 7)
        self.down3 = Down(256, 256)
        self.sa2 = SelfAttention(256, 3)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa3 = SelfAttention(128, 7)
        self.up2 = Up(256, 64)
        self.up3 = Up(128, 64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

    def pos_encoding(self, t, channels):
        """
        Compute a positional embedding of timesteps (Tansfomer sinusoidal position embedding).

        Args:
            t (tensor of shape (N)): timesteps
            channels: embedding dimension

        Returns:
            pos_enc (tensor of shape (N, channels)): embedded timesteps
        """
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, y):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        if y is not None:
            t += self.label_emb(y)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x3 = self.down2(x2, t)
        x3 = self.sa1(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa2(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa3(x)
        x = self.up2(x, x2, t)
        x = self.up3(x, x1, t)
        output = self.outc(x)
        return output


if __name__ == '__main__':
    net = UNet_conditional(c_in=1, c_out=1, num_classes=10, device="cpu")
    print(sum([p.numel() for p in net.parameters()]))
    x = torch.randn(3, 1, 28, 28)
    t = x.new_tensor([500] * x.shape[0]).long()
    y = x.new_tensor([1] * x.shape[0]).long()
    print(net(x, t, y).shape)
