import torch
import torch.nn as nn


def _conv_out_size(size: int, kernel_size: int, stride: int, padding: int) -> int:
    return ((size + 2 * padding - kernel_size) // stride) + 1


class AsymmetricMAMBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        history_len: int,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.history_len = history_len

        self.temporal_scale_embed = nn.Parameter(
            torch.randn(1, history_len, 1, dim) * 0.02
        )
        self.template_type_embed = nn.Parameter(
            torch.randn(1, 1, 1, dim) * 0.02
        )
        self.search_type_embed = nn.Parameter(
            torch.randn(1, history_len + 1, 1, dim) * 0.02
        )

        self.template_query_norm = nn.LayerNorm(dim)
        self.template_kv_norm = nn.LayerNorm(dim)
        self.search_query_norm = nn.LayerNorm(dim)
        self.search_kv_norm = nn.LayerNorm(dim)

        self.template_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.search_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.output_proj = nn.Linear(dim, dim)
        self.output_norm = nn.LayerNorm(dim)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, init_tokens, search_groups, return_attention=True):
        batch_size, num_search_groups, num_patches, dim = search_groups.shape
        if num_search_groups != self.history_len + 1:
            raise ValueError(
                f"Expected {self.history_len + 1} search groups, got {num_search_groups}"
            )

        hist_groups = (
            search_groups[:, : self.history_len]
            + self.temporal_scale_embed
            + self.search_type_embed[:, : self.history_len]
        )
        det_group = (
            search_groups[:, self.history_len : self.history_len + 1]
            + self.search_type_embed[:, self.history_len : self.history_len + 1]
        )
        init_tokens_typed = init_tokens + self.template_type_embed.squeeze(1)
        search_groups_typed = torch.cat([hist_groups, det_group], dim=1)
        search_tokens_typed = search_groups_typed.reshape(
            batch_size,
            num_search_groups * num_patches,
            dim,
        )

        template_q = self.template_query_norm(init_tokens_typed)
        template_kv = self.template_kv_norm(init_tokens_typed)
        template_context, template_attn = self.template_attention(
            template_q,
            template_kv,
            template_kv,
            need_weights=return_attention,
            average_attn_weights=False,
        )

        search_q = self.search_query_norm(search_tokens_typed)
        search_kv_input = torch.cat([init_tokens_typed, search_tokens_typed], dim=1)
        search_kv = self.search_kv_norm(search_kv_input)
        search_context, search_attn = self.search_attention(
            search_q,
            search_kv,
            search_kv,
            need_weights=return_attention,
            average_attn_weights=False,
        )

        fused_context = torch.cat([template_context, search_context], dim=1)
        residual_input = torch.cat([init_tokens, search_groups.reshape(batch_size, -1, dim)], dim=1)
        output_tokens = self.output_norm(self.output_proj(fused_context) + residual_input)
        output_tokens = output_tokens + self.mlp(self.mlp_norm(output_tokens))

        template_output = output_tokens[:, :num_patches, :]
        search_output = output_tokens[:, num_patches:, :].reshape(
            batch_size,
            num_search_groups,
            num_patches,
            dim,
        )
        return template_output, search_output, template_attn, search_attn


class MultiStageMAMStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        grid_h: int,
        grid_w: int,
        num_heads: int,
        depth: int,
        history_len: int,
        dropout: float = 0.0,
        mlp_ratio: int = 4,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.embed = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.num_patches = grid_h * grid_w
        self.out_channels = out_channels
        self.history_len = history_len
        self.patch_pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, out_channels) * 0.02
        )
        self.blocks = nn.ModuleList(
            [
                AsymmetricMAMBlock(
                    dim=out_channels,
                    num_heads=num_heads,
                    history_len=history_len,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

    def _map_to_tokens(self, feature_map):
        batch_size, channels, height, width = feature_map.shape
        if height != self.grid_h or width != self.grid_w:
            raise ValueError(
                f"Stage feature map mismatch: expected {(self.grid_h, self.grid_w)}, "
                f"got {(height, width)}"
            )
        tokens = feature_map.flatten(2).transpose(1, 2)
        return tokens + self.patch_pos_embed

    def _tokens_to_map(self, tokens):
        batch_size, num_patches, channels = tokens.shape
        return tokens.transpose(1, 2).reshape(
            batch_size,
            channels,
            self.grid_h,
            self.grid_w,
        )

    def forward(self, init_map, hist_maps, det_map, return_attention=True):
        batch_size = init_map.size(0)
        hist_len = hist_maps.size(1)
        if hist_len != self.history_len:
            raise ValueError(
                f"Expected history_len={self.history_len}, got {hist_len}"
            )

        init_embed = self.embed(init_map)
        det_embed = self.embed(det_map)

        hist_flat = hist_maps.reshape(
            batch_size * hist_len,
            hist_maps.size(2),
            hist_maps.size(3),
            hist_maps.size(4),
        )
        hist_embed = self.embed(hist_flat).reshape(
            batch_size,
            hist_len,
            self.out_channels,
            self.grid_h,
            self.grid_w,
        )

        init_tokens = self._map_to_tokens(init_embed)
        det_tokens = self._map_to_tokens(det_embed).unsqueeze(1)
        hist_tokens = self._map_to_tokens(
            hist_embed.reshape(
                batch_size * hist_len,
                self.out_channels,
                self.grid_h,
                self.grid_w,
            )
        ).reshape(batch_size, hist_len, self.num_patches, self.out_channels)
        search_groups = torch.cat([hist_tokens, det_tokens], dim=1)

        last_template_attn = None
        last_search_attn = None
        for block in self.blocks:
            init_tokens, search_groups, last_template_attn, last_search_attn = block(
                init_tokens,
                search_groups,
                return_attention=return_attention,
            )

        init_next = self._tokens_to_map(init_tokens)
        hist_next = self._tokens_to_map(
            search_groups[:, :hist_len].reshape(
                batch_size * hist_len,
                self.num_patches,
                self.out_channels,
            )
        ).reshape(
            batch_size,
            hist_len,
            self.out_channels,
            self.grid_h,
            self.grid_w,
        )
        det_next = self._tokens_to_map(search_groups[:, hist_len])
        return init_next, hist_next, det_next, last_template_attn, last_search_attn


class TemporalAttentionScorer(nn.Module):
    """Multi-stage patch-token temporal matcher for MOT pair scoring."""

    DEFAULT_STAGE_DIMS = (64, 192, 384)
    DEFAULT_STAGE_HEADS = (4, 3, 6)
    DEFAULT_STAGE_DEPTHS = (1, 4, 8)
    DEFAULT_STAGE_KERNELS = (7, 3, 3)
    DEFAULT_STAGE_STRIDES = (4, 2, 2)

    def __init__(
        self,
        image_height=256,
        image_width=128,
        patch_size=16,
        hidden_dim=256,
        num_heads=4,
        dropout=0.0,
        history_len=3,
        num_stages=3,
        stage_dims=None,
        stage_heads=None,
        stage_depths=None,
        stage_kernels=None,
        stage_strides=None,
    ):
        super().__init__()
        if history_len < 2:
            raise ValueError("history_len must be at least 2")

        self.image_height = image_height
        self.image_width = image_width
        self.patch_size = patch_size
        self.dropout = dropout
        self.history_len = history_len
        self.num_stages = int(num_stages)

        if self.num_stages < 1 or self.num_stages > 3:
            raise ValueError("num_stages must be in {1,2,3}")

        if stage_dims is None:
            if self.num_stages == 1:
                stage_dims = [hidden_dim]
            else:
                stage_dims = list(self.DEFAULT_STAGE_DIMS[: self.num_stages])
        if stage_heads is None:
            if self.num_stages == 1:
                stage_heads = [num_heads]
            else:
                stage_heads = list(self.DEFAULT_STAGE_HEADS[: self.num_stages])
        if stage_depths is None:
            if self.num_stages == 1:
                stage_depths = [1]
            else:
                stage_depths = list(self.DEFAULT_STAGE_DEPTHS[: self.num_stages])
        if stage_kernels is None:
            if self.num_stages == 1:
                stage_kernels = [patch_size]
            else:
                stage_kernels = list(self.DEFAULT_STAGE_KERNELS[: self.num_stages])
        if stage_strides is None:
            if self.num_stages == 1:
                stage_strides = [patch_size]
            else:
                stage_strides = list(self.DEFAULT_STAGE_STRIDES[: self.num_stages])

        if not (
            len(stage_dims)
            == len(stage_heads)
            == len(stage_depths)
            == len(stage_kernels)
            == len(stage_strides)
            == self.num_stages
        ):
            raise ValueError("Stage configuration lengths must all match num_stages")

        self.stage_dims = [int(x) for x in stage_dims]
        self.stage_heads = [int(x) for x in stage_heads]
        self.stage_depths = [int(x) for x in stage_depths]
        self.stage_kernels = [int(x) for x in stage_kernels]
        self.stage_strides = [int(x) for x in stage_strides]

        for dim, heads in zip(self.stage_dims, self.stage_heads):
            if dim % heads != 0:
                raise ValueError(f"Stage dim {dim} must be divisible by heads {heads}")

        self.hidden_dim = self.stage_dims[-1]
        self.num_heads = self.stage_heads[-1]

        stage_grids = []
        current_h = image_height
        current_w = image_width
        for kernel_size, stride in zip(self.stage_kernels, self.stage_strides):
            padding = kernel_size // 2
            current_h = _conv_out_size(current_h, kernel_size, stride, padding)
            current_w = _conv_out_size(current_w, kernel_size, stride, padding)
            stage_grids.append((current_h, current_w))
        self.stage_grids = stage_grids
        self.grid_h, self.grid_w = stage_grids[-1]
        self.num_patches = self.grid_h * self.grid_w

        in_channels = [3] + self.stage_dims[:-1]
        self.stages = nn.ModuleList(
            [
                MultiStageMAMStage(
                    in_channels=in_channels[i],
                    out_channels=self.stage_dims[i],
                    kernel_size=self.stage_kernels[i],
                    stride=self.stage_strides[i],
                    grid_h=self.stage_grids[i][0],
                    grid_w=self.stage_grids[i][1],
                    num_heads=self.stage_heads[i],
                    depth=self.stage_depths[i],
                    history_len=history_len,
                    dropout=dropout,
                    mlp_ratio=4,
                )
                for i in range(self.num_stages)
            ]
        )

        self.scorer = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def _validate_inputs(self, init_crop, det_crop, hist_crops):
        if init_crop.dim() != 4:
            raise ValueError(
                f"init_crop must have shape (B, 3, H, W), got {tuple(init_crop.shape)}"
            )
        if det_crop.dim() != 4:
            raise ValueError(
                f"det_crop must have shape (B, 3, H, W), got {tuple(det_crop.shape)}"
            )
        if hist_crops.dim() != 5:
            raise ValueError(
                f"hist_crops must have shape (B, T, 3, H, W), got {tuple(hist_crops.shape)}"
            )
        if init_crop.size(0) != det_crop.size(0) or det_crop.size(0) != hist_crops.size(0):
            raise ValueError("init_crop, det_crop and hist_crops must share the same batch size")
        if init_crop.size(1) != 3 or det_crop.size(1) != 3 or hist_crops.size(2) != 3:
            raise ValueError("all crops must have 3 channels")
        if init_crop.size(-2) != self.image_height or init_crop.size(-1) != self.image_width:
            raise ValueError(
                "init_crop spatial size mismatch: "
                f"expected {(self.image_height, self.image_width)}, "
                f"got {(init_crop.size(-2), init_crop.size(-1))}"
            )
        if det_crop.size(-2) != self.image_height or det_crop.size(-1) != self.image_width:
            raise ValueError(
                "det_crop spatial size mismatch: "
                f"expected {(self.image_height, self.image_width)}, "
                f"got {(det_crop.size(-2), det_crop.size(-1))}"
            )
        if hist_crops.size(-2) != self.image_height or hist_crops.size(-1) != self.image_width:
            raise ValueError(
                "hist_crops spatial size mismatch: "
                f"expected {(self.image_height, self.image_width)}, "
                f"got {(hist_crops.size(-2), hist_crops.size(-1))}"
            )
        if hist_crops.size(1) < self.history_len:
            raise ValueError(
                f"hist_crops must have at least {self.history_len} history states, got {hist_crops.size(1)}"
            )

    def forward(self, init_crop, det_crop, hist_crops, return_attention=True):
        self._validate_inputs(init_crop, det_crop, hist_crops)
        hist_maps = hist_crops[:, : self.history_len]
        init_map = init_crop
        det_map = det_crop

        last_template_attn = None
        last_search_attn = None
        for stage in self.stages:
            init_map, hist_maps, det_map, last_template_attn, last_search_attn = stage(
                init_map,
                hist_maps,
                det_map,
                return_attention=return_attention,
            )

        batch_size = init_map.size(0)
        template_tokens = init_map.flatten(2).transpose(1, 2)
        det_tokens = det_map.flatten(2).transpose(1, 2)
        template_pooled = template_tokens.mean(dim=1)
        det_pooled = det_tokens.mean(dim=1)

        fused = torch.cat([template_pooled, det_pooled], dim=-1)
        score = self.scorer(fused).squeeze(-1)

        if return_attention:
            return score, {
                "template_attn": last_template_attn,
                "search_attn": last_search_attn,
            }
        return score
