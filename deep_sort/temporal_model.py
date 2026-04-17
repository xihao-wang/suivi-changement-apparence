import torch
import torch.nn as nn


class TemporalAttentionScorer(nn.Module):
    """A lightweight temporal scorer based on cross-attention.

    Inputs
    ------
    det_feat:
        Tensor of shape (B, F), the current detection feature.
    hist_feat:
        Tensor of shape (B, T, F), the ordered matched-detection history features.
        A typical setting is T=3 with [df_t, df_{t-i}, df_{t-2i}].

    Outputs
    -------
    score:
        Tensor of shape (B,), a scalar matching score per pair.
    attn:
        Tensor of shape (B, num_heads, 1, T), attention weights for debugging.
    """

    def __init__(self, feature_dim, hidden_dim=256, num_heads=4, dropout=0.0):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.q_proj = nn.Linear(feature_dim, hidden_dim)
        self.k_proj = nn.Linear(feature_dim, hidden_dim)
        self.v_proj = nn.Linear(feature_dim, hidden_dim)
        self.q_norm = nn.LayerNorm(hidden_dim)
        self.k_norm = nn.LayerNorm(hidden_dim)

        self.hist_pos_embed = nn.Parameter(torch.randn(1, 3, hidden_dim) * 0.02)

        # Use batch_first so the module accepts (B, T, C).
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, det_feat, hist_feat, return_attention=True):
        if det_feat.dim() != 2:
            raise ValueError(
                f"det_feat must have shape (B, F), got {tuple(det_feat.shape)}"
            )
        if hist_feat.dim() != 3:
            raise ValueError(
                f"hist_feat must have shape (B, T, F), got {tuple(hist_feat.shape)}"
            )
        if det_feat.size(0) != hist_feat.size(0):
            raise ValueError("det_feat and hist_feat must share the same batch size")
        if det_feat.size(-1) != self.feature_dim or hist_feat.size(-1) != self.feature_dim:
            raise ValueError(
                "feature dimension mismatch: "
                f"expected {self.feature_dim}, got det={det_feat.size(-1)} "
                f"hist={hist_feat.size(-1)}"
            )

        if hist_feat.size(1) < 3:
            raise ValueError(
                f"hist_feat must have at least 3 ordered states, got {hist_feat.size(1)}"
            )

        # query  = current detection feature
        # tokens = [df_t, df_t-i, df_t-2i]
        temporal_tokens = hist_feat[:, :3, :]

        q = self.q_norm(self.q_proj(det_feat)).unsqueeze(1)  # (B, 1, H)
        query = q.squeeze(1)  # (B, H)

        k = self.k_norm(
            self.k_proj(temporal_tokens) + self.hist_pos_embed[:, :temporal_tokens.size(1), :]
        )
        v = self.v_proj(temporal_tokens) + self.hist_pos_embed[:, :temporal_tokens.size(1), :]

        context, attn = self.attention(
            q,
            k,
            v,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        context = context.squeeze(1)  # (B, H)

        ## Option 1: use context alone for scoring.
        # score = self.mlp(context).squeeze(-1)           # (B,)

        ## Option 2: concatenate query and context for scoring.
        score = self.mlp(torch.cat([query, context], dim=-1)).squeeze(-1)  # (B,)

        if return_attention:
            return score, attn
        return score


def build_temporal_input(det_feat, df_t, df_t_i, df_t_2i):
    """Build the ordered temporal input tensor.

    Parameters
    ----------
    det_feat : Tensor, shape (B, F)
    df_t : Tensor, shape (B, F)
    df_t_i : Tensor, shape (B, F)
    df_t_2i : Tensor, shape (B, F)

    Returns
    -------
    hist_feat : Tensor, shape (B, 3, F)
    """
    if not (det_feat.dim() == df_t.dim() == df_t_i.dim() == df_t_2i.dim() == 2):
        raise ValueError("all input features must have shape (B, F)")
    return torch.stack([df_t, df_t_i, df_t_2i], dim=1)
