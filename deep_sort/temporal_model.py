import torch
import torch.nn as nn


class TemporalAttentionScorer(nn.Module):
    """Single-stage patch-token temporal matcher for MOT pair scoring."""

    def __init__(
        self,
        image_height=256,
        image_width=128,
        patch_size=16,
        hidden_dim=256,
        num_heads=4,
        dropout=0.0,
        history_len=3,
        num_stages=1,
        stage_dims=None,
        stage_heads=None,
        stage_depths=None,
        stage_kernels=None,
        stage_strides=None,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if history_len < 2:
            raise ValueError("history_len must be at least 2")
        if image_height % patch_size != 0 or image_width % patch_size != 0:
            raise ValueError("image size must be divisible by patch_size")
        if int(num_stages) != 1:
            raise ValueError("This rollback version only supports num_stages=1")

        self.image_height = image_height
        self.image_width = image_width
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.history_len = history_len
        self.num_stages = 1

        self.grid_h = image_height // patch_size
        self.grid_w = image_width // patch_size
        self.num_patches = self.grid_h * self.grid_w

        self.patch_embed = nn.Conv2d(
            3,
            hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.patch_pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, hidden_dim) * 0.02
        )
        self.temporal_scale_embed = nn.Parameter(
            torch.randn(1, history_len, 1, hidden_dim) * 0.02
        )
        self.template_type_embed = nn.Parameter(
            torch.randn(1, 1, 1, hidden_dim) * 0.02
        )
        self.search_type_embed = nn.Parameter(
            torch.randn(1, history_len + 1, 1, hidden_dim) * 0.02
        )

        self.template_query_norm = nn.LayerNorm(hidden_dim)
        self.template_kv_norm = nn.LayerNorm(hidden_dim)
        self.search_query_norm = nn.LayerNorm(hidden_dim)
        self.search_kv_norm = nn.LayerNorm(hidden_dim)
        self.template_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.search_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
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

    def _patchify(self, x):
        tokens = self.patch_embed(x)
        tokens = tokens.flatten(2).transpose(1, 2)
        return tokens + self.patch_pos_embed

    def build_tokens(self, init_crop, det_crop, hist_crops):
        self._validate_inputs(init_crop, det_crop, hist_crops)
        batch_size = init_crop.size(0)
        hist_crops = hist_crops[:, : self.history_len]

        init_tokens = self._patchify(init_crop).unsqueeze(1)
        det_tokens = self._patchify(det_crop).unsqueeze(1)

        hist_flat = hist_crops.reshape(
            batch_size * self.history_len,
            3,
            self.image_height,
            self.image_width,
        )
        hist_tokens = self._patchify(hist_flat)
        hist_tokens = hist_tokens.reshape(
            batch_size,
            self.history_len,
            self.num_patches,
            self.hidden_dim,
        )
        init_tokens = init_tokens + self.template_type_embed
        search_groups = torch.cat(
            [
                hist_tokens + self.temporal_scale_embed + self.search_type_embed[:, : self.history_len],
                det_tokens + self.search_type_embed[:, self.history_len : self.history_len + 1],
            ],
            dim=1,
        )
        return init_tokens.squeeze(1), search_groups

    def forward(self, init_crop, det_crop, hist_crops, return_attention=True):
        init_tokens, search_groups = self.build_tokens(
            init_crop, det_crop, hist_crops
        )
        batch_size = init_tokens.size(0)
        num_search_groups = search_groups.size(1)
        search_tokens = search_groups.reshape(
            batch_size,
            num_search_groups * self.num_patches,
            self.hidden_dim,
        )

        template_q = self.template_query_norm(init_tokens)
        template_kv = self.template_kv_norm(init_tokens)
        template_context, template_attn = self.template_attention(
            template_q,
            template_kv,
            template_kv,
            need_weights=return_attention,
            average_attn_weights=False,
        )

        search_q = self.search_query_norm(search_tokens)
        search_kv_input = torch.cat([init_tokens, search_tokens], dim=1)
        search_kv = self.search_kv_norm(search_kv_input)
        search_context, search_attn = self.search_attention(
            search_q,
            search_kv,
            search_kv,
            need_weights=return_attention,
            average_attn_weights=False,
        )

        fused_context = torch.cat([template_context, search_context], dim=1)
        residual_input = torch.cat([init_tokens, search_tokens], dim=1)
        projected_tokens = self.output_proj(fused_context)
        output_tokens = self.output_norm(projected_tokens + residual_input)

        template_output = output_tokens[:, : self.num_patches, :]
        search_output = output_tokens[:, self.num_patches :, :].reshape(
            batch_size,
            num_search_groups,
            self.num_patches,
            self.hidden_dim,
        )

        template_pooled = template_output.mean(dim=1)
        det_pooled = search_output[:, -1].mean(dim=1)

        fused = torch.cat([template_pooled, det_pooled], dim=-1)
        score = self.scorer(fused).squeeze(-1)

        if return_attention:
            return score, {
                "template_attn": template_attn,
                "search_attn": search_attn,
            }
        return score
