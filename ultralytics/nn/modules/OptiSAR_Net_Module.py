import einops
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch.nn.parameter import Parameter
from .conv import Conv, autopad

__all__ = (
    "DAAM",
    "BSPPF",
    "VSSA"
)

class EnhancedConvolutionalBlock(nn.Module):
    """
    Enhanced Convolutional Block (ECB) that applies two 1x1 convolutions with optional residual connection.
    """

    def __init__(self, input_channels, hidden_channels=None, output_channels=None,
                 dropout_rate=0., use_residual=True):
        super().__init__()

        # If not specified, keep the number of channels constant
        output_channels = output_channels or input_channels
        hidden_channels = hidden_channels or input_channels

        # First 1x1 convolution layer
        self.conv_reduce = Conv(input_channels, hidden_channels, k=1)

        # Second 1x1 convolution layer
        self.conv_expand = Conv(hidden_channels, output_channels, k=1)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout_rate)

        # Flag to control residual connection
        self.use_residual = use_residual

    def forward(self, x):
        # Store the input for potential residual connection
        residual = x

        # First convolution
        x = self.conv_reduce(x)
        x = self.dropout(x)

        # Second convolution
        x = self.conv_expand(x)
        x = self.dropout(x)

        # Add residual connection if enabled
        if self.use_residual:
            return x + residual
        else:
            return x


class DualAdaptiveAttention(nn.Module):
    """
    Dual Adaptive Attention (DAA).
    """
    def __init__(self, dim):
        super().__init__()

        # Input projection
        self.input_proj = Conv(dim, dim, k=1)
        self.activation = nn.GELU()

        # Output projection
        self.output_proj = Conv(dim, dim, k=1)

        # Depth-wise convolutions for local and global feature extraction
        self.local_conv = Conv(dim, dim, k=3, p=1, g=dim)
        self.global_conv = Conv(dim, dim, k=3, p=3, g=dim, d=3)

        # Channel reduction for attention computation
        self.channel_reducer_local = Conv(dim, dim // 2, k=1)
        self.channel_reducer_global = Conv(dim, dim // 2, k=1)

        # Attention squeeze operation
        self.attention_squeeze = Conv(2, 2, k=7, p=3)

        # Final channel mixing
        self.channel_mixer = Conv(dim // 2, dim, k=1)

    def forward(self, x):
        # Store input for residual connection
        residual = x.clone()

        # Input projection and activation
        x = self.input_proj(x)
        x = self.activation(x)

        # Local feature extraction
        local_features = self.local_conv(x)

        # Global feature extraction
        global_features = self.global_conv(local_features)

        # Compute attention for local and global features
        attn_local = self.channel_reducer_local(local_features)
        attn_global = self.channel_reducer_global(global_features)

        # Concatenate local and global attention
        attn_combined = torch.cat([attn_local, attn_global], dim=1)

        # Compute average and max attention
        attn_avg = torch.mean(attn_combined, dim=1, keepdim=True)
        attn_max, _ = torch.max(attn_combined, dim=1, keepdim=True)

        # Aggregate average and max attention
        attn_pooled = torch.cat([attn_avg, attn_max], dim=1)

        # Apply attention squeeze and sigmoid activation
        attn_weights = self.attention_squeeze(attn_pooled).sigmoid()

        # Compute weighted attention
        weighted_attn = (
            attn_local * attn_weights[:, 0, :, :].unsqueeze(1) +
            attn_global * attn_weights[:, 1, :, :].unsqueeze(1)
        )

        # Mix channels in the attention
        attn_mixed = self.channel_mixer(weighted_attn)

        # Apply attention to global features
        x = global_features * attn_mixed

        # Output projection
        x = self.output_proj(x)

        # Add residual connection
        x = x + residual

        return x


class DAAM(nn.Module):
    """
    Dual Attention Adaptive Module (DAAM) that combines attention and ECB
    with optional layer scaling.
    """

    def __init__(self, dim, use_auto_layer_scaling=True, layer_scale_init_value=1e-2):
        super().__init__()

        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.daa = DualAdaptiveAttention(dim)
        self.ecb = EnhancedConvolutionalBlock(dim, dim)

        self.use_auto_layer_scaling = use_auto_layer_scaling
        if use_auto_layer_scaling:
            self.layer_scale_daa = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_ecb = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        else:
            self.layer_scale_daa = None
            self.layer_scale_ecb = None

    def forward(self, x):
        # Apply attention
        attention_output = self.daa(self.norm1(x))
        if self.use_auto_layer_scaling:
            attention_output = self.layer_scale_daa.unsqueeze(-1).unsqueeze(-1) * attention_output
        x = x + attention_output

        # Apply ECB
        ecb_output = self.ecb(self.norm2(x))
        if self.use_auto_layer_scaling:
            ecb_output = self.layer_scale_ecb.unsqueeze(-1).unsqueeze(-1) * ecb_output
        x = x + ecb_output

        return x


class TopkRouting(nn.Module):
    """
    Differentiable top-k routing with scaling, adapted from bi-level routing attention.
    Args:
        qk_dim: int, feature dimension of query and key
        topk: int, the 'topk'
        qk_scale: int or None, temperature (multiply) of softmax activation
        with_param: bool, wether inorporate learnable params in routing unit
        diff_routing: bool, wether make routing differentiable
        soft_routing: bool, wether make output value multiplied by routing weights
    """

    def __init__(self, qk_dim, topk=4, qk_scale=None):
        super().__init__()
        self.topk = topk
        self.qk_dim = qk_dim
        self.scale = qk_scale or qk_dim ** -0.5
        # routing activation
        self.routing_act = nn.Softmax(dim=-1)

    def forward(self, query: Tensor, key: Tensor) -> Tuple[Tensor]:
        """
        Args:
            q, k: (n, p^2, c) tensor
        Return:
            r_weight, topk_index: (n, p^2, topk) tensor
        """

        query, key = query.detach(), key.detach()
        attn_logit = (query * self.scale) @ key.transpose(-2, -1)  # (n, p^2, p^2)
        topk_attn_logit, topk_index = torch.topk(attn_logit, k=self.topk, dim=-1)  # (n, p^2, k), (n, p^2, k)
        r_weight = self.routing_act(topk_attn_logit)  # (n, p^2, k)

        return r_weight, topk_index


class KVGather(nn.Module):
    """
       KVGather module for efficient key-value pair selection based on routing indices.
       This module is part of the bi-level routing attention mechanism.
    """

    def __init__(self):
        super().__init__()

    def forward(self, r_idx: Tensor, r_weight: Tensor, kv: Tensor):
        """
        r_idx: (n, p^2, topk) tensor
        r_weight: (n, p^2, topk) tensor
        kv: (n, p^2, w^2, c_kq+c_v)
        Return:
            (n, p^2, topk, w^2, c_kq+c_v) tensor
        """
        # select kv according to routing index
        n, p2, w2, c_kv = kv.size()
        topk = r_idx.size(-1)
        # FIXME: gather consumes much memory (topk times redundancy), write cuda kernel?
        topk_kv = torch.gather(kv.view(n, 1, p2, w2, c_kv).expand(-1, p2, -1, -1, -1),
                               # (n, p^2, p^2, w^2, c_kv) without mem cpy
                               dim=2,
                               index=r_idx.view(n, p2, topk, 1, 1).expand(-1, -1, -1, w2, c_kv)
                               # (n, p^2, k, w^2, c_kv)
                               )

        return topk_kv


class BiLevelRoutingDeformableAttention(nn.Module):
    """
    Bi-Level Routing Deformable Attention module.

    This module combines local attention with global routing and deformable convolutions
    for enhanced feature extraction in computer vision tasks.
    """

    def __init__(self, dim, num_windows=11, num_heads=4, qk_dim=None, qk_scale=None,
                 kv_per_window=4, kv_downsample_mode='ada_maxpool', topk=4,
                 side_conv=3, use_deformable=True, off_conv=7
                ):
        """
        Initialize the Bi-Level Routing Deformable Attention.

        Args:
            dim (int): Number of input channels.
            num_windows (int): Number of windows in each dimension for local attention.
            num_heads (int): Number of attention heads.
            qk_dim (int): Dimension of query and key vectors. If None, set to dim. Default is None.
            qk_scale (float): Scaling factor for query-key dot product. If None, set to 1/sqrt(qk_dim).
            kv_per_window (int): Number of key-value pairs per window for downsampling.
            kv_downsample_mode (str): Mode for downsampling key-value pairs. Options: 'ada_avgpool', 'ada_maxpool'.
            topk (int): Number of top attention scores to consider in routing.
            side_dwconv (int): Kernel size for depthwise convolution in LEPE. Set to 0 to disable.
            use_deformable (bool):
            auto_pad (bool): Whether to automatically pad input to match window size.
        """
        super().__init__()
        self.dim = dim
        self.num_windows = num_windows
        self.num_heads = num_heads
        self.qk_dim = qk_dim or dim
        assert self.qk_dim % num_heads == 0 and self.dim % num_heads == 0, 'qk_dim and dim must be divisible by num_heads!'
        self.scale = qk_scale or self.qk_dim ** -0.5

        # Side-enhanced convolution (SEC)
        self.sec = nn.Conv2d(dim, dim, kernel_size=side_conv, stride=1, padding=side_conv // 2,
                              groups=dim) if side_conv > 0 else \
            lambda x: torch.zeros_like(x)

        # Global routing settings
        self.topk = topk
        self.router = TopkRouting(qk_dim=self.qk_dim, qk_scale=self.scale, topk=self.topk)
        self.kv_gather = KVGather()

        # Query, Key, Value projections
        self.query_proj = Conv(dim, dim, 1)
        self.kv_proj = Conv(dim, dim * 2, 1)

        # Key-Value downsampling
        self.kv_downsample_mode = kv_downsample_mode
        self.kv_per_window = kv_per_window
        if self.kv_downsample_mode == 'ada_avgpool':
            self.kv_down = nn.AdaptiveAvgPool2d(self.kv_per_window)
        elif self.kv_downsample_mode == 'ada_maxpool':
            self.kv_down = nn.AdaptiveMaxPool2d(self.kv_per_window)
        else:
            self.kv_down = nn.Identity()

        self.attn_act = nn.Softmax(dim=-1)
        self.use_deformable = use_deformable
        self.off_conv = off_conv

        # Offset prediction
        self.offset_predictor = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, kernel_size=self.off_conv, stride=1, padding=self.off_conv // 2, groups=self.dim),
            nn.BatchNorm2d(dim),
            nn.Conv2d(self.dim, 2, kernel_size=1, stride=1, padding=0, bias=False)
        )



    @torch.no_grad()
    def _get_reference_points(self, height, width, batch_size, dtype, device):
        """
        Generate reference points for deformable attention.

        Args:
            height (int): Height of the feature map.
            width (int): Width of the feature map.
            batch_size (int): Batch size.
            dtype (torch.dtype): Data type of the tensor.
            device (torch.device): Device to create the tensor on.

        Returns:
            torch.Tensor: Reference points of shape (batch_size, height, width, 2).
        """
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, height - 0.5, height, dtype=dtype, device=device),
            torch.linspace(0.5, width - 0.5, width, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(width - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(height - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(batch_size, -1, -1, -1)
        return ref

    def forward(self, x):
        x = rearrange(x, "n c h w -> n h w c")

        # Auto-padding
        batch_size, height_in, width_in, channels = x.size()
        pad_left = pad_top = 0
        pad_right = (self.num_windows - width_in % self.num_windows) % self.num_windows
        pad_bottom = (self.num_windows - height_in % self.num_windows) % self.num_windows
        x = F.pad(x, (0, 0, pad_left, pad_right, pad_top, pad_bottom))
        _, height, width, _ = x.size()  # padded size

        # Reshape input for window-based processing
        x = rearrange(x, "n (j h) (i w) c -> n c (j h) (i w)", j=self.num_windows, i=self.num_windows)

        # Query projection
        query = self.query_proj(x)
        query_offset = query
        query = rearrange(query, "n c (j h) (i w) -> n (j i) h w c", j=self.num_windows, i=self.num_windows)

        # Deformable offset calculation
        offset = self.offset_predictor(query_offset).contiguous()
        height_key, width_key = offset.size(2), offset.size(3)
        offset = einops.rearrange(offset, 'b p h w -> b h w p')
        dtype, device = x.dtype, x.device
        batch_size, _, _, _ = offset.size()
        reference = self._get_reference_points(height_key, width_key, batch_size, dtype, device)

        pos = (offset + reference).clamp(-0.05, +0.05)

        # Apply deformable sampling
        if self.use_deformable:
            x_sampled = F.grid_sample(
                input=x,
                grid=pos[..., (1, 0)],  # y, x -> x, y
                mode='bilinear', align_corners=True)
        else:
            x_sampled = x

        # Key-Value projection
        kv = self.kv_proj(x_sampled)
        kv = rearrange(kv, "n c (j h) (i w) -> n (j i) h w c", j=self.num_windows, i=self.num_windows)


        # Reshape for pixel-wise and window-wise operations
        query_pixel = rearrange(query, 'n p2 h w c -> n p2 (h w) c')
        kv_pixel = rearrange(kv, 'n p2 h w c -> (n p2) c h w')

        # Downsample key-value pairs
        kv_pixel = self.kv_down(kv_pixel)
        kv_pixel = rearrange(kv_pixel, '(n j i) c h w -> n (j i) (h w) c', j=self.num_windows, i=self.num_windows)

        # Window-wise query and key
        query_window, key_window = query.mean([2, 3]), kv[..., 0:self.qk_dim].mean([2, 3])

        # Side-enhanced convolution (SEC)
        sec = self.sec(rearrange(kv[..., self.qk_dim:], 'n (j i) h w c -> n c (j h) (i w)', j=self.num_windows,
                                   i=self.num_windows).contiguous())
        sec = rearrange(sec, 'n c (j h) (i w) -> n (j h) (i w) c', j=self.num_windows, i=self.num_windows)

        # Global routing
        routing_weights, routing_indices = self.router(query_window, key_window)

        # Gather key-value pairs based on routing
        kv_pixel_selected = self.kv_gather(r_idx=routing_indices, r_weight=routing_weights, kv=kv_pixel)
        key_pixel_selected, value_pixel_selected = kv_pixel_selected.split([self.qk_dim, self.dim], dim=-1)

        # Reshape for multi-head attention
        key_pixel_selected = rearrange(key_pixel_selected, 'n p2 k w2 (m c) -> (n p2) m c (k w2)', m=self.num_heads)
        value_pixel_selected = rearrange(value_pixel_selected, 'n p2 k w2 (m c) -> (n p2) m (k w2) c', m=self.num_heads)
        query_pixel = rearrange(query_pixel, 'n p2 w2 (m c) -> (n p2) m w2 c', m=self.num_heads)

        # Compute attention weights and apply attention
        attn_weights = (query_pixel * self.scale) @ key_pixel_selected
        attn_weights = self.attn_act(attn_weights)
        out = attn_weights @ value_pixel_selected

        # Reshape output and add SEC
        out = rearrange(out, '(n j i) m (h w) c -> n (j h) (i w) (m c)', j=self.num_windows, i=self.num_windows,
                        h=height // self.num_windows, w=width // self.num_windows)
        out = out + sec

        # Remove padding if applied
        out = out[:, :height_in, :width_in, :].contiguous()

        return rearrange(out, "n h w c -> n c h w")


class BSPPF(nn.Module):
    # The Bi-Level Routing Deformable Spatial Pyramid Pooling - Fast (BSPPF) layer
    # is used to dynamically adjust the allocation of multi-scale feature space.

    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.brda = BiLevelRoutingDeformableAttention(c_)

    def forward(self, x):
        x = self.cv1(x)

        # Apply Bi-Level Routing Deformable Attention with residual connection
        x = x + self.brda(x)

        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class SpatialShuffleAttention(nn.Module):
    """
    Spatial Shuffle Attention (SSA).
    """

    def __init__(self, dim, groups=8, dropout_rate=0.1):
        """
        Initialize the Spatial Shuffle Attention.

        Args:
            dim (int): Number of input channels. Default is 512.
            groups (int): Number of groups for channel shuffling. Default is 8.
            dropout_rate (float): Dropout rate for regularization. Default is 0.1.
        """
        super().__init__()
        self.groups = groups
        self.dim = dim

        # Pooling layers
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Learnable parameters for spatial attention
        self.weight_max = Parameter(torch.zeros(1, dim // (2 * groups), 1, 1))
        self.bias_max = Parameter(torch.ones(1, dim // (2 * groups), 1, 1))
        self.weight_avg = Parameter(torch.zeros(1, dim // (2 * groups), 1, 1))
        self.bias_avg = Parameter(torch.ones(1, dim // (2 * groups), 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_rate)

    @staticmethod
    def channel_shuffle(x, groups):
        """
        Perform channel shuffling operation.

        Args:
            x (torch.Tensor): Input tensor.
            groups (int): Number of groups for shuffling.

        Returns:
            torch.Tensor: Channel shuffled tensor.
        """
        batch_size, channels, height, width = x.shape
        channels_per_group = channels // groups

        # Reshape and transpose for shuffling
        x = x.view(batch_size, groups, channels_per_group, height, width)
        x = x.transpose(1, 2).contiguous()

        # Flatten the grouped channels
        x = x.view(batch_size, -1, height, width)

        return x

    def forward(self, x):
        b, c, h, w = x.size()

        # Group the input into subfeatures
        x = x.view(b * self.groups, -1, h, w)  # (b*groups, c//groups, h, w)

        # Apply initial channel shuffle
        x = self.channel_shuffle(x, 2)

        # Partitioning subspace
        x_1, x_2 = x.chunk(2, dim=1)

        # Apply pooling operations
        avg_pool = self.avg_pool(x_1)  # (batch_size*groups, channels//(2*groups), 1, 1)
        max_pool = self.max_pool(x_2)  # (batch_size*groups, channels//(2*groups), 1, 1)

        # Embedding global and key information
        avg_attention = self.weight_avg * avg_pool + self.bias_avg
        max_attention = self.weight_max * max_pool + self.bias_max

        # Dual-path spatial attention fusion
        channel_attention = torch.cat((max_attention, avg_attention), dim=1)
        channel_attention = self.sigmoid(channel_attention)

        # Apply attention and dropout
        x = x * self.dropout(channel_attention)

        # Reshape back to original dimensions
        out = x.contiguous().view(b, -1, h, w)

        # Final channel shuffle
        out = self.channel_shuffle(out, 2)

        return out


class GSConv(nn.Module):
    # GSConv https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, None, g, 1, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, 1, act)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)

        b, n, h, w = x2.data.size()
        b_n = b * n // 2
        y = x2.reshape(b_n, 2, h * w)
        y = y.permute(1, 0, 2)
        y = y.reshape(2, -1, n // 2, h, w)

        return torch.cat((y[0], y[1]), 1)


class GSBottleneck(nn.Module):
    # GS Bottleneck https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=3, s=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        # for lighting
        self.conv_lighting = nn.Sequential(
            GSConv(c1, c_, 3, 1),
            GSConv(c_, c2, 3, 1, act=False))

        self.shortcut = Conv(c1, c2, 1, 1, act=False)

    def forward(self, x):
        return self.conv_lighting(x) + self.shortcut(x)


class VSSA(nn.Module):
    """
    VoVGSCSP module with Spatial Shuffle Attention (VSSA).
    """

    def __init__(self, in_channels, out_channels, num_gsb=4, expansion_factor=0.5, dropout_rate=0.05):
        """
        Initialize the VSSA module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            num_gsb (int): Number of GSBottleneck layers. Default is 4.
            expansion_factor (float): Factor to determine the number of hidden channels. Default is 0.5.
        """
        super().__init__()
        hidden_channels = int(out_channels * expansion_factor)  # Calculate hidden channels
        self.conv_input_1 = Conv(in_channels, hidden_channels, k=1, s=1)
        self.conv_input_2 = Conv(in_channels, hidden_channels, k=1, s=1)

        # GSBottleneck sequence
        self.gsb_sequence = nn.Sequential(
            *(GSBottleneck(hidden_channels, hidden_channels, e=1.0) for _ in range(num_gsb))
        )
        self.conv_output = Conv(2 * hidden_channels, out_channels, k=1)

        # Spatial Shuffle Attention
        self.spatial_shuffle_attention = SpatialShuffleAttention(in_channels)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Apply Spatial Shuffle Attention and dropout
        attended_features = self.dropout(self.spatial_shuffle_attention(x))

        # Process through GSBottleneck sequence
        gsb_output = self.gsb_sequence(self.conv_input_1(attended_features))

        # Direct path through second input convolution
        direct_path = self.conv_input_2(attended_features)

        # Concatenate and process through output convolution
        combined_features = torch.cat((direct_path, gsb_output), dim=1)
        output = self.conv_output(combined_features)

        return output