import torch
import torch.nn as nn


class TemporalAttentionScorer(nn.Module):
    """Patch-token temporal matcher for MOT pair scoring.

    Inputs
    ------
    init_crop:
        Tensor of shape (B, 3, H, W), the initial template crop for the track.
    det_crop:
        Tensor of shape (B, 3, H, W), the current detection crop.
    hist_crops:
        Tensor of shape (B, T, 3, H, W), the ordered online updated template crops.
        Typical ordering is [t, t-i, t-2i].

    Outputs
    -------
    score:
        Tensor of shape (B,), scalar matching score per pair.
    attn:
        Attention weights of shape (B, heads, Nq, T * Nh), where Nq is the
        number of query patches in the current detection and Nh is the number
        of patches per history crop.
    """

    def __init__(
        self,
        image_height=256,
        image_width=128,
        patch_size=16,
        hidden_dim=256,
        num_heads=4,
        dropout=0.0,
        history_len=3,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if history_len < 2:
            raise ValueError("history_len must be at least 2")
        if image_height % patch_size != 0 or image_width % patch_size != 0:
            raise ValueError("image size must be divisible by patch_size")

        self.image_height = image_height
        self.image_width = image_width
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.history_len = history_len

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

        init_tokens = self._patchify(init_crop)
        det_tokens = self._patchify(det_crop)

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
        online_tokens = (
            hist_tokens
            + self.temporal_scale_embed
            + self.template_type_embed[:, 1:]
        )
        init_tokens = init_tokens.unsqueeze(1) + self.template_type_embed[:, :1]

        template_tokens = torch.cat([init_tokens, online_tokens], dim=1)
        template_tokens = template_tokens.reshape(
            batch_size,
            (self.history_len + 1) * self.num_patches,
            self.hidden_dim,
        )
        return init_tokens.squeeze(1), template_tokens, det_tokens

    def forward(self, init_crop, det_crop, hist_crops, return_attention=True):
        init_tokens, template_tokens, det_tokens = self.build_tokens(
            init_crop, det_crop, hist_crops
        )

        template_q = self.template_query_norm(template_tokens)
        template_kv = self.template_kv_norm(template_tokens)
        template_context, template_attn = self.template_attention(
            template_q,
            template_kv,
            template_kv,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        template_projected = self.output_proj(template_context)
        template_output = self.output_norm(template_projected + template_tokens)

        search_q = self.search_query_norm(det_tokens)
        search_kv = self.search_kv_norm(torch.cat([det_tokens, template_tokens], dim=1))
        search_context, search_attn = self.search_attention(
            search_q,
            search_kv,
            search_kv,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        search_projected = self.output_proj(search_context)
        search_output = self.output_norm(search_projected + det_tokens)

        template_pooled = template_output.mean(dim=1)
        det_pooled = search_output.mean(dim=1)

        fused = torch.cat([template_pooled, det_pooled], dim=-1)
        score = self.scorer(fused).squeeze(-1)

        if return_attention:
            return score, {
                "template_attn": template_attn,
                "search_attn": search_attn,
            }
        return score
