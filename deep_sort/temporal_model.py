import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttentionScorer(nn.Module):
    """Feature-level temporal scorer with short/long memory split.

    The scorer keeps the current detector / tracker pipeline intact and only
    restructures the temporal branch:

    - short-term memory encoder:
      current detection queries the recent matched-detection history
    - long-term memory encoder:
      a dedicated learnable query aggregates a sparse long memory
    - long-term gate:
      the long-term summary is down-weighted when it conflicts with the
      current observation
    - fusion scorer:
      combines current / short / gated-long representations into the final
      matching score

    Inputs
    ------
    det_feat:
        Tensor of shape (B, F), the current detection feature.
    hist_feat:
        Tensor of shape (B, Ts, F), ordered recent history features used by the
        short-term branch. A typical setting is Ts=3 with
        [df_t, df_{t-i}, df_{t-2i}].
    long_hist_feat:
        Optional tensor of shape (B, Tl, F), sparse long-term memory features.
        If omitted, the scorer falls back to a lightweight proxy built from
        hist_feat so the training pipeline remains usable without new labels.
    """

    def __init__(
        self,
        feature_dim,
        hidden_dim=256,
        num_heads=4,
        dropout=0.0,
        history_len=3,
        long_history_len=3,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if history_len < 2:
            raise ValueError("history_len must be at least 2")
        if long_history_len < 1:
            raise ValueError("long_history_len must be at least 1")

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.history_len = history_len
        self.long_history_len = long_history_len

        self.input_proj = nn.Linear(feature_dim, hidden_dim)

        self.short_pos_embed = nn.Parameter(
            torch.randn(1, history_len, hidden_dim) * 0.02
        )
        self.short_temporal_embed = nn.Parameter(
            torch.randn(1, history_len, hidden_dim) * 0.02
        )
        self.short_type_embed = nn.Parameter(
            torch.randn(1, history_len, hidden_dim) * 0.02
        )

        self.long_pos_embed = nn.Parameter(
            torch.randn(1, long_history_len, hidden_dim) * 0.02
        )
        self.long_temporal_embed = nn.Parameter(
            torch.randn(1, long_history_len, hidden_dim) * 0.02
        )
        self.long_type_embed = nn.Parameter(
            torch.randn(1, long_history_len, hidden_dim) * 0.02
        )
        self.long_query_token = nn.Parameter(
            torch.randn(1, 1, hidden_dim) * 0.02
        )

        self.short_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.long_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.det_norm = nn.LayerNorm(hidden_dim)
        self.short_query_norm = nn.LayerNorm(hidden_dim)
        self.short_kv_norm = nn.LayerNorm(hidden_dim)
        self.long_query_norm = nn.LayerNorm(hidden_dim)
        self.long_kv_norm = nn.LayerNorm(hidden_dim)

        self.short_out = nn.Linear(hidden_dim, hidden_dim)
        self.long_out = nn.Linear(hidden_dim, hidden_dim)
        self.short_out_norm = nn.LayerNorm(hidden_dim)
        self.long_out_norm = nn.LayerNorm(hidden_dim)

        gate_hidden = max(hidden_dim // 2, 64)
        self.long_gate = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 2, gate_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(gate_hidden, 1),
        )

        fusion_in_dim = hidden_dim * 3 + 5
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.last_long_gate = None
        self.last_short_similarity = None
        self.last_long_similarity = None
        self.last_long_attention = None

    def _validate_inputs(self, det_feat, hist_feat, long_hist_feat):
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
        if hist_feat.size(1) < self.history_len:
            raise ValueError(
                f"hist_feat must have at least {self.history_len} ordered states, got {hist_feat.size(1)}"
            )
        if long_hist_feat is not None:
            if long_hist_feat.dim() != 3:
                raise ValueError(
                    f"long_hist_feat must have shape (B, T, F), got {tuple(long_hist_feat.shape)}"
                )
            if long_hist_feat.size(0) != det_feat.size(0):
                raise ValueError("long_hist_feat must share the same batch size")
            if long_hist_feat.size(-1) != self.feature_dim:
                raise ValueError(
                    f"long_hist_feat feature dim mismatch: expected {self.feature_dim}, got {long_hist_feat.size(-1)}"
                )
            if long_hist_feat.size(1) < self.long_history_len:
                raise ValueError(
                    f"long_hist_feat must have at least {self.long_history_len} states, got {long_hist_feat.size(1)}"
                )

    def _build_long_proxy(self, hist_feat):
        # Oldest -> newest ordering gives the long branch a slower, coarser
        # temporal view even when no dedicated long-term memory is available.
        reversed_hist = torch.flip(hist_feat[:, : self.history_len, :], dims=[1])
        if reversed_hist.size(1) >= self.long_history_len:
            return reversed_hist[:, : self.long_history_len, :]
        pad_count = self.long_history_len - reversed_hist.size(1)
        pad = reversed_hist[:, -1:, :].expand(-1, pad_count, -1)
        return torch.cat([reversed_hist, pad], dim=1)

    def build_input_tokens(self, det_feat, hist_feat, long_hist_feat=None):
        self._validate_inputs(det_feat, hist_feat, long_hist_feat)
        if long_hist_feat is None:
            long_hist_feat = self._build_long_proxy(hist_feat)

        det_token = self.input_proj(det_feat)

        short_tokens = self.input_proj(hist_feat[:, : self.history_len, :])
        short_tokens = (
            short_tokens
            + self.short_pos_embed
            + self.short_temporal_embed
            + self.short_type_embed
        )

        long_tokens = self.input_proj(long_hist_feat[:, : self.long_history_len, :])
        long_tokens = (
            long_tokens
            + self.long_pos_embed
            + self.long_temporal_embed
            + self.long_type_embed
        )
        return det_token, short_tokens, long_tokens

    def encode_short_memory(
        self,
        det_feat,
        hist_feat,
        long_hist_feat=None,
        input_tokens=None,
        return_attention=True,
    ):
        if input_tokens is None:
            input_tokens = self.build_input_tokens(det_feat, hist_feat, long_hist_feat)
        det_token, short_tokens, _ = input_tokens
        short_query = self.short_query_norm(det_token.unsqueeze(1))
        short_kv = self.short_kv_norm(short_tokens)

        short_context, short_attn = self.short_attention(
            short_query,
            short_kv,
            short_kv,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        short_vec = self.short_out_norm(
            self.short_out(short_context.squeeze(1)) + 0.1 * det_token
        )
        if return_attention:
            return short_vec, short_attn
        return short_vec

    def encode_long_memory(
        self,
        det_feat,
        hist_feat,
        long_hist_feat=None,
        input_tokens=None,
        return_attention=True,
    ):
        if input_tokens is None:
            input_tokens = self.build_input_tokens(det_feat, hist_feat, long_hist_feat)
        _, _, long_tokens = input_tokens
        batch_size = long_tokens.size(0)
        long_query = self.long_query_token.expand(batch_size, -1, -1)
        long_query = self.long_query_norm(long_query)
        long_kv = self.long_kv_norm(long_tokens)

        long_context, long_attn = self.long_attention(
            long_query,
            long_kv,
            long_kv,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        long_vec = self.long_out_norm(
            self.long_out(long_context.squeeze(1)) + 0.1 * self.long_query_token.squeeze(0)
        )
        if return_attention:
            return long_vec, long_attn
        return long_vec

    def _compute_long_gate(self, det_vec, short_vec, long_vec):
        det_unit = F.normalize(det_vec, dim=-1)
        short_unit = F.normalize(short_vec, dim=-1)
        long_unit = F.normalize(long_vec, dim=-1)

        sim_short = (det_unit * short_unit).sum(dim=-1, keepdim=True)
        sim_long = (det_unit * long_unit).sum(dim=-1, keepdim=True)
        gate_input = torch.cat([det_vec, short_vec, long_vec, sim_short, sim_long], dim=-1)
        long_gate = torch.sigmoid(self.long_gate(gate_input))
        return long_gate, sim_short, sim_long

    def forward(self, det_feat, hist_feat, long_hist_feat=None, return_attention=True):
        input_tokens = self.build_input_tokens(det_feat, hist_feat, long_hist_feat)
        det_vec = self.det_norm(input_tokens[0])

        if return_attention:
            short_vec, short_attn = self.encode_short_memory(
                det_feat,
                hist_feat,
                long_hist_feat=long_hist_feat,
                input_tokens=input_tokens,
                return_attention=True,
            )
            long_vec, long_attn = self.encode_long_memory(
                det_feat,
                hist_feat,
                long_hist_feat=long_hist_feat,
                input_tokens=input_tokens,
                return_attention=True,
            )
        else:
            short_vec = self.encode_short_memory(
                det_feat,
                hist_feat,
                long_hist_feat=long_hist_feat,
                input_tokens=input_tokens,
                return_attention=False,
            )
            long_vec = self.encode_long_memory(
                det_feat,
                hist_feat,
                long_hist_feat=long_hist_feat,
                input_tokens=input_tokens,
                return_attention=False,
            )
            short_attn = None
            long_attn = None

        long_gate, sim_short, sim_long = self._compute_long_gate(det_vec, short_vec, long_vec)
        gated_long_vec = long_gate * long_vec

        fusion_input = torch.cat(
            [
                det_vec,
                short_vec,
                gated_long_vec,
                long_gate,
                sim_short,
                sim_long,
                sim_short - sim_long,
                (short_vec * gated_long_vec).mean(dim=-1, keepdim=True),
            ],
            dim=-1,
        )
        fused = self.fusion(fusion_input)
        score = self.scorer(fused).squeeze(-1)

        self.last_long_gate = long_gate.detach()
        self.last_short_similarity = sim_short.detach()
        self.last_long_similarity = sim_long.detach()
        self.last_long_attention = None if long_attn is None else long_attn.detach()

        if return_attention:
            return score, short_attn
        return score


def build_temporal_input(det_feat, df_t, df_t_i, df_t_2i):
    """Build the ordered short-term temporal input tensor.

    Returns
    -------
    hist_feat : Tensor, shape (B, 3, F)
        Ordered as [df_t, df_t-i, df_t-2i].
    """
    if not (det_feat.dim() == df_t.dim() == df_t_i.dim() == df_t_2i.dim() == 2):
        raise ValueError("all input features must have shape (B, F)")
    return torch.stack([df_t, df_t_i, df_t_2i], dim=1)
