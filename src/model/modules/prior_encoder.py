import math

import torch
import torch.nn as nn

from .layers import LayerNorm
from ..utils import convert_pad_shape


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        channel,
        out_channel,
        n_heads,
        window_size=None,
        heads_share=True,
        p_dropout=0.0,
        proximal_bias=False,
        proximal_init=False,
    ):
        super(MultiHeadAttention, self).__init__()
        assert channel % n_heads == 0

        self.channel = channel
        self.out_channel = out_channel
        self.n_heads = n_heads
        self.window_size = window_size
        self.heads_share = heads_share
        self.proximal_bias = proximal_bias
        self.p_dropout = p_dropout
        self.attn = None

        self.k_channel = channel // n_heads
        self.conv_q = torch.nn.Conv1d(channel, channel, 1)
        self.conv_k = torch.nn.Conv1d(channel, channel, 1)
        self.conv_v = torch.nn.Conv1d(channel, channel, 1)
        if window_size is not None:
            n_heads_rel = 1 if heads_share else n_heads
            rel_stddev = self.k_channel**-0.5
            self.emb_rel_k = torch.nn.Parameter(
                torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channel)
                * rel_stddev
            )
            self.emb_rel_v = torch.nn.Parameter(
                torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channel)
                * rel_stddev
            )
        self.conv_o = torch.nn.Conv1d(channel, out_channel, 1)
        self.drop = torch.nn.Dropout(p_dropout)

        torch.nn.init.xavier_uniform_(self.conv_q.weight)
        torch.nn.init.xavier_uniform_(self.conv_k.weight)
        if proximal_init:
            self.conv_k.weight.data.copy_(self.conv_q.weight.data)
            self.conv_k.bias.data.copy_(self.conv_q.bias.data)
        torch.nn.init.xavier_uniform_(self.conv_v.weight)

    def forward(self, x, c, attn_mask=None):
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)

        x, self.attn = self.attention(q, k, v, mask=attn_mask)

        x = self.conv_o(x)
        return x

    def attention(self, query, key, value, mask=None):
        b, d, t_s, t_t = (*key.size(), query.size(2))
        query = query.view(b, self.n_heads, self.k_channel, t_t).transpose(2, 3)
        key = key.view(b, self.n_heads, self.k_channel, t_s).transpose(2, 3)
        value = value.view(b, self.n_heads, self.k_channel, t_s).transpose(2, 3)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.k_channel)
        if self.window_size is not None:
            assert (
                t_s == t_t
            ), "Relative attention is only available for self-attention."
            key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, t_s)
            rel_logits = self._matmul_with_relative_keys(query, key_relative_embeddings)
            rel_logits = self._relative_position_to_absolute_position(rel_logits)
            scores_local = rel_logits / math.sqrt(self.k_channel)
            scores = scores + scores_local
        if self.proximal_bias:
            assert t_s == t_t, "Proximal bias is only available for self-attention."
            scores = scores + self._attention_bias_proximal(t_s).to(
                device=scores.device, dtype=scores.dtype
            )
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
        p_attn = torch.nn.functional.softmax(scores, dim=-1)
        p_attn = self.drop(p_attn)
        output = torch.matmul(p_attn, value)
        if self.window_size is not None:
            relative_weights = self._absolute_position_to_relative_position(p_attn)
            value_relative_embeddings = self._get_relative_embeddings(
                self.emb_rel_v, t_s
            )
            output = output + self._matmul_with_relative_values(
                relative_weights, value_relative_embeddings
            )
        output = output.transpose(2, 3).contiguous().view(b, d, t_t)
        return output, p_attn

    def _matmul_with_relative_values(self, x, y):
        ret = torch.matmul(x, y.unsqueeze(0))
        return ret

    def _matmul_with_relative_keys(self, x, y):
        ret = torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))
        return ret

    def _get_relative_embeddings(self, relative_embeddings, length):
        pad_length = max(length - (self.window_size + 1), 0)
        slice_start_position = max((self.window_size + 1) - length, 0)
        slice_end_position = slice_start_position + 2 * length - 1
        if pad_length > 0:
            padded_relative_embeddings = torch.nn.functional.pad(
                relative_embeddings,
                convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]]),
            )
        else:
            padded_relative_embeddings = relative_embeddings
        used_relative_embeddings = padded_relative_embeddings[
            :, slice_start_position:slice_end_position
        ]
        return used_relative_embeddings

    def _relative_position_to_absolute_position(self, x):
        B, H, L, _ = x.size()
        x = torch.nn.functional.pad(
            x, convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, 1]])
        )
        x_flat = x.view([B, H, L * 2 * L])
        x_flat = torch.nn.functional.pad(
            x_flat, convert_pad_shape([[0, 0], [0, 0], [0, L - 1]])
        )
        x_final = x_flat.view([B, H, L + 1, 2 * L - 1])[:, :, :L, L - 1 :]  # noqa
        return x_final

    def _absolute_position_to_relative_position(self, x):
        batch, heads, length, _ = x.size()
        x = torch.nn.functional.pad(
            x, convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, length - 1]])
        )
        x_flat = x.view([batch, heads, length**2 + length * (length - 1)])
        x_flat = torch.nn.functional.pad(
            x_flat, convert_pad_shape([[0, 0], [0, 0], [length, 0]])
        )
        x_final = x_flat.view([batch, heads, length, 2 * length])[:, :, :, 1:]
        return x_final

    def _attention_bias_proximal(self, length):
        r = torch.arange(length, dtype=torch.float32)
        diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
        return torch.unsqueeze(torch.unsqueeze(-torch.log1p(torch.abs(diff)), 0), 0)


class FFN(nn.Module):
    def __init__(self, in_channel, h_channel, out_channel, kernel_size, p_dropout=0.0):
        super(FFN, self).__init__()
        self.conv_1 = torch.nn.Conv1d(
            in_channel, h_channel, kernel_size, padding=kernel_size // 2
        )
        self.conv_2 = torch.nn.Conv1d(
            h_channel, out_channel, kernel_size, padding=kernel_size // 2
        )
        self.drop = torch.nn.Dropout(p_dropout)

    def forward(self, x, x_mask):
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        return x * x_mask


class PhonemeEncoder(nn.Module):
    def __init__(
        self,
        num_vocab,
        channels,
        num_head,
        num_layers,
        kernel_size=1,
        dropout=0.0,
        window_size=4,
    ):
        super().__init__()
        self.channels = channels
        self.num_layers = num_layers
        # Embedding
        self.emb = torch.nn.Embedding(num_vocab, channels)
        torch.nn.init.normal_(self.emb.weight, 0.0, channels**-0.5)

        # FFT Block
        self.drop = torch.nn.Dropout(dropout)
        self.attn_layers = torch.nn.ModuleList()
        self.norm_layers_1 = torch.nn.ModuleList()
        self.ffn_layers = torch.nn.ModuleList()
        self.norm_layers_2 = torch.nn.ModuleList()
        for _ in range(self.num_layers):
            self.attn_layers.append(
                MultiHeadAttention(
                    channels,
                    channels,
                    num_head,
                    window_size=window_size,
                    p_dropout=dropout,
                )
            )
            self.norm_layers_1.append(LayerNorm(channels))
            self.ffn_layers.append(
                FFN(
                    channels,
                    channels * 4,
                    channels,
                    kernel_size,
                    p_dropout=dropout,
                )
            )
            self.norm_layers_2.append(LayerNorm(channels))
        # Postnet
        self.postnet = torch.nn.Conv1d(channels, channels * 2, 1)

    def forward(self, x, mask):
        x = self.emb(x) * math.sqrt(self.channels)
        x = x.transpose(-1, -2)
        attn_mask = mask.unsqueeze(2) * mask.unsqueeze(-1)
        for i in range(self.num_layers):
            x = x * mask
            y = self.attn_layers[i](x, x, attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)
            y = self.ffn_layers[i](x, mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)
        x = x * mask
        stats = self.postnet(x) * mask
        m, log_s = stats.split([self.channels] * 2, dim=1)
        return x, m, log_s
