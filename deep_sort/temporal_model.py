import torch
import torch.nn as nn


class TemporalAttentionScorer(nn.Module):
    """Temporal scorer placeholder.

    The previous prototype attention logic has been removed.  The replacement
    architecture should be added step by step so the implementation matches the
    design being tested.

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
        Not implemented until the next temporal attention design is added.
    """

    def __init__(self, feature_dim, hidden_dim=256, num_heads=4, dropout=0.0, history_len=3):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if history_len < 2:
            raise ValueError("history_len must be at least 2")

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.history_len = history_len
        self.input_proj = nn.Linear(feature_dim, hidden_dim)
        self.history_pos_embed = nn.Parameter(torch.randn(1, history_len, hidden_dim) * 0.02)
        # Explicitly encode temporal distance:
        # df_t -> 0 step ago, df_t-i -> 1 step ago, ...
        self.temporal_scale_embed = nn.Parameter(torch.randn(1, history_len, hidden_dim) * 0.02)
        # Explicitly encode token role:
        # slot 0 is the most recent history token, the remaining slots are older history.
        self.history_type_embed = nn.Parameter(torch.randn(1, history_len, hidden_dim) * 0.02)
        self.det_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.history_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.history_query_norm = nn.LayerNorm(hidden_dim)
        self.history_kv_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * (2 + history_len) + history_len, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def _validate_inputs(self, det_feat, hist_feat):
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

    def build_input_tokens(self, det_feat, hist_feat):
        """Project raw feature tokens into the hidden attention space.

        Feature-level mapping used for MOT matching:
        - template/reference token: prototype built from history
        - current query token: det_feat
        - temporal memory tokens: the first history_len ordered states

        With nn.MultiheadAttention, Q/K/V projections are created internally
        from the query/key/value inputs passed to each attention branch.
        """
        self._validate_inputs(det_feat, hist_feat)

        # Use a stable feature-level prototype as the template/reference token.
        prototype_feat = hist_feat[:, : self.history_len, :].mean(dim=1)
        template_token = self.input_proj(prototype_feat).unsqueeze(1)
        query_token = self.input_proj(det_feat).unsqueeze(1)
        history_tokens = self.input_proj(hist_feat[:, : self.history_len, :])
        history_tokens = (
            history_tokens
            + self.history_pos_embed
            + self.temporal_scale_embed
            + self.history_type_embed
        )
        return template_token, query_token, history_tokens

    def build_det_self_attention(
        self, det_feat, hist_feat, input_tokens=None, return_attention=True
    ):
        """Apply the formula-4 Attention_t pattern to the reference token.

        In the original paper, Attention_t is applied to the initial template.
        Here the template/reference role is assigned to a prototype built from
        the recent matched-detection history.
        """
        if input_tokens is None:
            input_tokens = self.build_input_tokens(det_feat, hist_feat)
        template_token, _, _ = input_tokens

        template_context, template_attn = self.det_attention(
            template_token,
            template_token,
            template_token,
            need_weights=return_attention,
            average_attn_weights=False,
        )

        if return_attention:
            return template_context, template_attn
        return template_context

    def build_history_attention(
        self, det_feat, hist_feat, input_tokens=None, return_attention=True
    ):
        """Apply the formula-4 Attention_s pattern to history tokens.

        Feature-level mapping of the paper's notation:
        - q_s1: Q of det_feat
        - the template/reference branch is handled separately by the prototype
        - k_s1...k_sn and v_s1...v_sn: K/V of the first history_len ordered states

        This keeps the asymmetric matching structure:
        the current detection query attends to the candidate track history.
        """
        if input_tokens is None:
            input_tokens = self.build_input_tokens(det_feat, hist_feat)
        _, query_token, history_tokens = input_tokens
        query_token = self.history_query_norm(query_token)
        history_tokens = self.history_kv_norm(history_tokens)

        history_context, history_attn = self.history_attention(
            query_token,
            history_tokens,
            history_tokens,
            need_weights=return_attention,
            average_attn_weights=False,
        )

        if return_attention:
            return history_context, history_attn, history_tokens
        return history_context

    def build_projected_tokens(self, det_feat, hist_feat, return_attention=True):
        """Concatenate attention branches and apply output projection.

        This corresponds to the C + Linear Projection + residual part after
        formula-4 attention operations.  The output token order is:
        [reference(df_t), current_query(det_feat)].
        """
        input_tokens = self.build_input_tokens(det_feat, hist_feat)
        template_token, query_token, _ = input_tokens
        branch_inputs = torch.cat([template_token, query_token], dim=1)

        template_context, template_attn = self.build_det_self_attention(
            det_feat, hist_feat, input_tokens=input_tokens, return_attention=True
        )
        history_context, history_attn, history_tokens = self.build_history_attention(
            det_feat, hist_feat, input_tokens=input_tokens, return_attention=True
        )

        attention_tokens = torch.cat([template_context, history_context], dim=1)
        projected_tokens = self.output_proj(attention_tokens)
        output_tokens = self.output_norm(projected_tokens + 0.1 * branch_inputs)
        # output_tokens = self.output_norm(projected_tokens)


        if return_attention:
            return output_tokens, template_attn, history_attn, history_tokens
        return output_tokens

    def forward(self, det_feat, hist_feat, return_attention=True):
        if return_attention:
            output_tokens, _, history_attn, history_tokens = self.build_projected_tokens(
                det_feat, hist_feat, return_attention=True
            )
        else:
            output_tokens = self.build_projected_tokens(
                det_feat, hist_feat, return_attention=False
            )

        template_output = output_tokens[:, 0, :]
        query_output = output_tokens[:, 1, :]
        if return_attention:
            attn_mean = history_attn.mean(dim=1).squeeze(1)
            per_history_contrib = attn_mean.unsqueeze(-1) * history_tokens
            fused = torch.cat(
                [template_output, query_output]
                + [per_history_contrib[:, i, :] for i in range(self.history_len)]
                + [attn_mean],
                dim=-1,
            )
        else:
            # Recompute the minimal history path needed by the scorer without
            # returning attention tensors to the caller.
            _, query_token, history_tokens = self.build_input_tokens(det_feat, hist_feat)
            query_token = self.history_query_norm(query_token)
            history_tokens = self.history_kv_norm(history_tokens)
            _, history_attn = self.history_attention(
                query_token,
                history_tokens,
                history_tokens,
                need_weights=True,
                average_attn_weights=False,
            )
            attn_mean = history_attn.mean(dim=1).squeeze(1)
            per_history_contrib = attn_mean.unsqueeze(-1) * history_tokens
            fused = torch.cat(
                [template_output, query_output]
                + [per_history_contrib[:, i, :] for i in range(self.history_len)]
                + [attn_mean],
                dim=-1,
            )
        score = self.scorer(fused).squeeze(-1)

        if return_attention:
            return score, history_attn
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
