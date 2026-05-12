from typing import Tuple, List, Any, Tuple
from einops import rearrange

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from ezdl.models.layers import DropPath, ConvModule, MultiDropPath
from transformers import SegformerModel, SegformerConfig


CHANNEL_PRETRAIN = {'R': 0, 'G': 1, 'B': 2, 'NIR': 3, 'RE': 4}


class Attention(nn.Module):
    def __init__(self, dim, head, sr_ratio):
        super().__init__()
        self.head = head
        self.sr_ratio = sr_ratio
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, sr_ratio, sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, H, W) -> Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x = x.permute(0, 2, 1).reshape(B, C, H, W)
            x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(x)

        k, v = self.kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: Tensor, H, W) -> Tensor:
        B, _, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)


class MLP(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.fc2 = nn.Linear(c2, c1)

    def forward(self, x: Tensor, H, W) -> Tensor:
        return self.fc2(F.gelu(self.dwconv(self.fc1(x), H, W)))


class PatchEmbed(nn.Module):
    def __init__(self, c1=3, c2=32, patch_size=7, stride=4):
        super().__init__()
        self.proj = nn.Conv2d(c1, c2, patch_size, stride, patch_size // 2)  # padding=(ps[0]//2, ps[1]//2)
        self.norm = nn.LayerNorm(c2)

    def forward(self, x: Tensor) -> Tuple[Any, Any, Any]:
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class Block(nn.Module):
    def __init__(self, dim, head, sr_ratio=1, dpr=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, head, sr_ratio)
        self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * 4))

    def forward(self, x: Tensor, H, W) -> Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


mit_settings = {
    'B0': [[32, 64, 160, 256], [2, 2, 2, 2]],  # [embed_dims, depths]
    'B1': [[64, 128, 320, 512], [2, 2, 2, 2]],
    'B2': [[64, 128, 320, 512], [3, 4, 6, 3]],
    'B3': [[64, 128, 320, 512], [3, 4, 18, 3]],
    'B4': [[64, 128, 320, 512], [3, 8, 27, 3]],
    'B5': [[64, 128, 320, 512], [3, 6, 40, 3]],

    'LD': [[16, 32, 80, 128], [2, 2, 2, 2]], # Lightweight deep
    # 'LT': [[32, 64, 160, 256], [1, 1, 1, 1]], # Lightweight thick
    'L0': [[16, 32, 80, 128], [1, 1, 1, 1]],
    'L1': [[8, 16, 40, 64], [1, 1, 1, 1]],
    'L2': [[4, 8, 20, 32], [1, 1, 1, 1]],
}


class MiT(nn.Module):
    pretrained_url = ("nvidia/segformer-", "-finetuned-ade-512-512")

    def __init__(self, model_name: str = 'B0', input_channels=3, n_blocks=4, pretrained=False):
        super().__init__()
        if not pretrained:
            assert model_name in mit_settings.keys(), f"MiT model name should be in {list(mit_settings.keys())}"
            embed_dims, depths = mit_settings[model_name]
            drop_path_rate = 0.1
            self.channels = embed_dims

            # patch_embed
            patch_sizes = [7, 3, 3, 3]
            paddings = [4, 2, 2, 2]
            for i in range(n_blocks):
                c_in = input_channels if i == 0 else embed_dims[i-1]
                setattr(self, f"patch_embed{i+1}", PatchEmbed(c_in, embed_dims[i], patch_sizes[i], paddings[i]))

            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

            heads = [1, 2, 5, 8]
            ratios = [8, 4, 2, 1]
            cur = 0
            for i in range(n_blocks):
                setattr(self, f"block{i+1}",
                        nn.ModuleList([Block(embed_dims[i], heads[i], ratios[i], dpr[cur + k]) for k in range(depths[i])]))
                setattr(self, f"norm{i+1}", nn.LayerNorm(embed_dims[i]))
                cur += depths[i]
            self.forward = self.base_forward
            self.partial_forward = self.base_partial_forward
            self.n_blocks = n_blocks
        else:
            url = self.pretrained_url[0] + model_name.lower() + self.pretrained_url[1]
            self.url = url
            config = SegformerConfig().from_pretrained(url)
            config.num_encoder_blocks = n_blocks
            config.num_channels = input_channels
            self.config = config
            self.channels = config.hidden_sizes
            self.encoder = SegformerModel(config)
            self.forward = self.hug_forward
            self.partial_forward = self.hug_partial_forward
            self.n_blocks = n_blocks

    def init_pretrained_weights(self, weights=None, channels_to_load=None, channel_scales=None):
        """Load pretrained weights with optional per-channel scaling.

        Args:
            weights: pretrained state_dict. If None, loads from self.url.
            channels_to_load: list of channel labels (R/G/B/NIR/RE) ή None for all.
            channel_scales: optional list of floats (same length as channels_to_load).
                            Multiplies the first-conv weights per channel — useful για
                            I3D-style scaling όταν αρχικοποιούμε non-RGB channels από
                            RGB pretrained (π.χ. NIR ← 0.6·G_pretrained).
                            None ή λίστα με 1.0s → καμία scaling (original behavior).
        """
        first_conv = 'encoder.patch_embeddings.0.proj.weight'
        weights = SegformerModel.from_pretrained(self.url).state_dict() if weights is None else weights

        if channels_to_load is None:
            channels_to_load = slice(self.config.num_channels)
        else:
            channels_to_load = [CHANNEL_PRETRAIN[x] for x in channels_to_load]

        if list(weights.keys())[0][:len("encoder.encoder")] == "encoder.encoder": # Remove extra encoder.
            weights = {k[len("encoder."):]: v for k, v in weights.items()}
        keys = weights.keys()
        fkeys = [k for k in keys if int(k.split('.')[2]) < self.n_blocks]
        weights = {k: weights[k] for k in fkeys}

        num_to_load = len(channels_to_load) if isinstance(channels_to_load, list) else channels_to_load.stop

        weights[first_conv] = weights[first_conv][:, channels_to_load]

        # I3D-style per-channel scaling (optional)
        if channel_scales is not None and isinstance(channels_to_load, list):
            if len(channel_scales) != len(channels_to_load):
                raise ValueError(
                    f"channel_scales length ({len(channel_scales)}) μη συμβατό με "
                    f"channels_to_load length ({len(channels_to_load)})"
                )
            for i, scale in enumerate(channel_scales):
                if scale != 1.0:
                    weights[first_conv][:, i] = weights[first_conv][:, i] * float(scale)

        if num_to_load < self.config.num_channels:
            remaining = self.config.num_channels - num_to_load
            weights[first_conv] = torch.cat([
                weights[first_conv],
                self.encoder.state_dict()[first_conv][:, -remaining:]], dim=1)

        self.encoder.load_state_dict(weights)

    def hug_forward(self, x):
        return self.encoder(x, output_hidden_states=True).hidden_states

    def base_partial_forward(self, x: Tensor, block_slice) -> Tensor:
        B = x.shape[0]

        outputs = ()
        for i in range(self.n_blocks)[block_slice]:
            x, H, W = getattr(self, f"patch_embed{i+1}")(x)
            for blk in getattr(self, f"block{i+1}"):
                x = blk(x, H, W)
            x = getattr(self, f"norm{i+1}")(x).reshape(B, H, W, -1).permute(0, 3, 1, 2)
            outputs = outputs + (x,)
        return outputs

    def base_forward(self, x):
        return self.base_partial_forward(x, slice(self.n_blocks))

    def hug_partial_forward(self, pixel_values, block_slice):
        batch_size = pixel_values.shape[0]
        all_hidden_states = ()
        hidden_states = pixel_values
        output_attentions = False
        for idx, x in list(enumerate(zip(
                self.encoder.encoder.patch_embeddings,
                self.encoder.encoder.block,
                self.encoder.encoder.layer_norm
        )))[block_slice]:
            embedding_layer, block_layer, norm_layer = x
            # first, obtain patch embeddings
            hidden_states, height, width = embedding_layer(hidden_states)
            # second, send embeddings through blocks
            for i, blk in enumerate(block_layer):
                layer_outputs = blk(hidden_states, height, width, output_attentions)
                hidden_states = layer_outputs[0]
            # third, apply layer norm
            hidden_states = norm_layer(hidden_states)
            # fourth, optionally reshape back to (batch_size, num_channels, height, width)
            if idx != len(self.encoder.encoder.patch_embeddings) - 1 or (
                idx == len(self.encoder.encoder.patch_embeddings) - 1 and self.encoder.encoder.config.reshape_last_stage
            ):
                hidden_states = hidden_states.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
            all_hidden_states = all_hidden_states + (hidden_states,)
        return all_hidden_states


class HSwish(nn.Module):
    """h-swish activation (Howard et al., MobileNetV3) — used στο Coordinate Attention paper."""
    def forward(self, x):
        return x * F.relu6(x + 3.0, inplace=False) / 6.0


class CoordinateAttention(nn.Module):
    """Coordinate Attention (Hou et al., CVPR 2021 — arXiv 2103.02907).

    Channel + spatial-position attention. Pool feature map κατά H και W
    separately (instead of global pool used by SE), encode location info,
    και επιστρέφει per-position attention weights.

    Args:
        in_channels: input feature channels
        reduction: bottleneck reduction ratio (paper default 32)
    """
    def __init__(self, in_channels: int, reduction: int = 32):
        super().__init__()
        mid = max(8, in_channels // reduction)
        # Direction-specific pooling (key insight of paper)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # (B, C, H, 1)
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # (B, C, 1, W)
        # Shared bottleneck after concat (eq. 7 paper)
        self.conv1 = nn.Conv2d(in_channels, mid, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.act = HSwish()
        # Per-direction projections (eq. 8 paper)
        self.conv_h = nn.Conv2d(mid, in_channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(mid, in_channels, kernel_size=1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        _, _, H, W = x.shape
        # Pool κατά H (keep height) και W (keep width)
        x_h = self.pool_h(x)                          # (B, C, H, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)      # (B, C, W, 1)
        # Concat κατά spatial dim και κάνε bottleneck
        y = torch.cat([x_h, x_w], dim=2)              # (B, C, H+W, 1)
        y = self.act(self.bn1(self.conv1(y)))
        # Split back & project ανά direction
        x_h, x_w = torch.split(y, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)                 # back to (B, mid, 1, W)
        a_h = torch.sigmoid(self.conv_h(x_h))         # (B, C, H, 1)
        a_w = torch.sigmoid(self.conv_w(x_w))         # (B, C, 1, W)
        # Apply attention (broadcasts to full H×W)
        return identity * a_h * a_w


class FusionBlock(nn.Module):
    def __init__(self, channels, fusion_type="conv_sum", p_local=0.1, p_glob=0.5,
                 coord_attn_reduction: int = 32):
        super().__init__()
        if fusion_type == "squeeze_excite":
            self.conv = ConvModule(channels * 2, channels, k=1, p=0)
            self.squeeze = nn.AdaptiveAvgPool2d(1)
            self.excite = nn.Sequential(
                nn.Linear(channels * 2, channels // 4),
                nn.ReLU(inplace=True),
                nn.Linear(channels // 4, channels * 2),
                nn.Sigmoid()
            )
            self.forward = self.squeeze_forward
        elif fusion_type == "coord_attn":
            # Channel + spatial-position attention (Hou et al., CVPR 2021).
            # Drop-in replacement for SE με extra spatial awareness.
            self.conv = ConvModule(channels * 2, channels, k=1, p=0)
            self.coord_attn = CoordinateAttention(
                in_channels=channels * 2,
                reduction=coord_attn_reduction,
            )
            self.forward = self.coord_attn_forward
        elif fusion_type == "conv_sum_drop" or fusion_type == "conv_sum":
            self.conv1 = ConvModule(channels, channels, k=1, p=0)
            self.conv2 = ConvModule(channels, channels, k=1, p=0)
            if fusion_type == "conv_sum_drop":
                self.multi_drop = MultiDropPath(num_inputs=2, p=p_glob)
                self.drop = DropPath(p_local)
                self.forward = self.drop_forward
            else:
                self.forward = self.base_forward

    def base_forward(self, x1, x2):
        return self.conv1(x1) + self.conv2(x2)

    def drop_forward(self, x1, x2):
        y1 = x1 + self.drop(self.conv1(x1))
        y2 = x2 + self.drop(self.conv2(x2))
        return sum(self.multi_drop([y1, y2]))

    def squeeze_forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        gap = self.squeeze(x)
        gap = rearrange(gap, 'b c h w -> b (h w c)')
        gap = self.excite(gap)
        gap = rearrange(gap, 'b (h w c) -> b c h w', h=1, w=1)
        x = x * gap + x
        return self.conv(x)

    def coord_attn_forward(self, x1, x2):
        """Concat 2 streams → Coordinate Attention → project back to single channel dim."""
        x = torch.cat([x1, x2], dim=1)
        x = self.coord_attn(x) + x   # residual (same as SE forward pattern)
        return self.conv(x)


class MiTFusion(nn.Module):
    def __init__(self, channel_dims: list, p_local=0.1, p_glob=0.5, fusion_type="conv_sum"):
        super().__init__()
        self.channel_dims = channel_dims
        for i in range(len(channel_dims)):
            setattr(self, f'fusion_{i}', FusionBlock(channel_dims[i],
                                                     fusion_type=fusion_type,
                                                     p_local=p_local,
                                                     p_glob=p_glob))

    def forward(self, x: Tuple[List]) -> List:
        y = [getattr(self, f"fusion_{i}")(f1, f2)
             for i, (f1, f2) in enumerate(zip(*x))]
        return y


if __name__ == '__main__':
    model = MiT('B0')
    x = torch.zeros(1, 3, 224, 224)
    outs = model(x)
    for y in outs:
        print(y.shape)
